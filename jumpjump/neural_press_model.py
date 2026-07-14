from __future__ import annotations

import json
import hashlib
import itertools
import math
import os
import random
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import neural_press_model_config
from .press_model import (
    failure_press_cap_ms,
    minimum_press_ms_for_distance,
    short_hop_press_cap_ms,
)
from .training_data import uses_current_landing_measurement, valid_training_samples
from .types import DependencyError, DetectionResult, JumpAutoError
from .utils import clamp


FEATURE_VERSION = 4
FEATURE_NAMES = (
    "dx_over_width",
    "dy_over_height",
    "distance_over_diagonal",
    "abs_dx_over_width",
    "abs_dy_over_height",
    "piece_width_over_width",
    "piece_height_over_height",
    "target_width_over_width",
    "target_height_over_height",
    "piece_scale_ratio",
    "game_score_log",
    "stage_press_scale",
    "confidence",
    "legacy_press_seconds",
)


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(
            "PyTorch is required for neural press training/inference. "
            "Install it from https://pytorch.org/get-started/locally/."
        ) from exc
    return torch


def _network(torch: Any, input_size: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def feature_vector(sample: dict[str, Any]) -> list[float]:
    width = max(1.0, _finite(sample.get("viewport_width_px"), 1.0))
    height = max(1.0, _finite(sample.get("viewport_height_px"), 1.0))
    diagonal = math.hypot(width, height)
    dx = _finite(sample.get("dx_px"))
    dy = _finite(sample.get("dy_px"))
    return [
        dx / width,
        dy / height,
        _finite(sample.get("effective_distance_px"), math.hypot(dx, dy)) / diagonal,
        abs(dx) / width,
        abs(dy) / height,
        _finite(sample.get("piece_width_px")) / width,
        _finite(sample.get("piece_height_px")) / height,
        _finite(sample.get("target_width_px")) / width,
        _finite(sample.get("target_height_px")) / height,
        clamp(_finite(sample.get("piece_scale_ratio"), 1.0), 0.25, 2.0),
        math.log1p(max(0.0, _finite(sample.get("game_score")))) / math.log(1001.0),
        clamp(_finite(sample.get("stage_press_scale"), 1.0), 0.25, 2.0),
        clamp(_finite(sample.get("confidence")), 0.0, 1.0),
        _finite(sample.get("legacy_press_ms")) / 1000.0,
    ]


def inference_sample(
    result: DetectionResult,
    viewport_size: tuple[int, int],
    legacy_press_ms: float,
) -> dict[str, Any]:
    return {
        "viewport_width_px": viewport_size[0],
        "viewport_height_px": viewport_size[1],
        "dx_px": result.dx_px,
        "dy_px": result.dy_px,
        "effective_distance_px": result.effective_distance_px,
        "piece_width_px": result.piece_bbox[2],
        "piece_height_px": result.piece_bbox[3],
        "target_width_px": result.target_bbox[2],
        "target_height_px": result.target_bbox[3],
        "piece_scale_ratio": result.piece_scale_ratio or 1.0,
        "game_score": result.game_score,
        "stage_press_scale": result.stage_press_scale or 1.0,
        "confidence": result.confidence,
        "legacy_press_ms": legacy_press_ms,
    }


def direction_name(sample: dict[str, Any]) -> str:
    return "left" if _finite(sample.get("dx_px")) < 0 else "right"


def coverage_key(sample: dict[str, Any], bin_size_px: float) -> str:
    distance = max(0.0, _finite(sample.get("effective_distance_px")))
    return f"{direction_name(sample)}:{int(distance // max(1.0, bin_size_px))}"


def build_coverage_bins(
    samples: Iterable[dict[str, Any]],
    bin_size_px: float,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = coverage_key(sample, bin_size_px)
        counts[key] = counts.get(key, 0) + 1
    bins: list[dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        if count < minimum_samples:
            continue
        direction, index_text = key.split(":", 1)
        index = int(index_text)
        bins.append(
            {
                "key": key,
                "direction": direction,
                "distance_min_px": index * bin_size_px,
                "distance_max_px": (index + 1) * bin_size_px,
                "samples": count,
            }
        )
    return bins


def coverage_bin_for_sample(
    sample: dict[str, Any],
    coverage_bins: Iterable[dict[str, Any]],
    bin_size_px: float,
) -> dict[str, Any] | None:
    key = coverage_key(sample, bin_size_px)
    return next((item for item in coverage_bins if item.get("key") == key), None)


def coverage_strength(sample_count: int, minimum_samples: int) -> float:
    if sample_count < minimum_samples:
        return 0.0
    return clamp((sample_count - minimum_samples + 1) / max(1.0, minimum_samples), 0.25, 1.0)


def apply_safety_limits(
    press_ms: float,
    result: DetectionResult,
    config: dict[str, Any],
) -> float:
    model = config["press_model"]
    short_cap = short_hop_press_cap_ms(result.effective_distance_px, model)
    if short_cap is not None:
        press_ms = min(press_ms, short_cap)
    failure_cap = failure_press_cap_ms(result.effective_distance_px, model, config)
    if failure_cap is not None:
        press_ms = min(press_ms, failure_cap)
    minimum_press = minimum_press_ms_for_distance(result.effective_distance_px, model, config)
    return clamp(press_ms, minimum_press, float(config["max_press_ms"]))


@dataclass
class Prediction:
    press_ms: float
    source: str
    legacy_press_ms: float
    correction_ratio: float = 0.0
    fallback_reason: str | None = None
    model_id: str | None = None


class NeuralPressPredictor:
    def __init__(self, torch: Any, model: Any, metadata: dict[str, Any], device: Any):
        self.torch = torch
        self.model = model
        self.metadata = metadata
        self.device = device

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> "NeuralPressPredictor":
        torch = import_torch()
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise JumpAutoError(f"Could not load neural model metadata '{metadata_path}': {exc}") from exc
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise JumpAutoError(
                f"Neural model feature version {metadata.get('feature_version')!r} is incompatible "
                f"with runtime version {FEATURE_VERSION}."
            )
        if metadata.get("feature_names") != list(FEATURE_NAMES):
            raise JumpAutoError("Neural model feature list is incompatible with this runtime.")
        expected_hash = metadata.get("weights_sha256")
        if not isinstance(expected_hash, str) or _file_sha256(model_path) != expected_hash:
            raise JumpAutoError("Neural model weights do not match their metadata.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _network(torch, len(FEATURE_NAMES))
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:
            raise JumpAutoError(f"Could not load neural model weights '{model_path}': {exc}") from exc
        model.to(device)
        model.eval()
        return cls(torch, model, metadata, device)

    def predict(
        self,
        result: DetectionResult,
        viewport_size: tuple[int, int],
        legacy_press_ms: float,
        config: dict[str, Any],
    ) -> Prediction:
        sample = inference_sample(result, viewport_size, legacy_press_ms)
        settings = neural_press_model_config(config)
        bin_size = float(self.metadata.get("coverage_bin_size_px", settings["coverage_bin_size_px"]))
        coverage = coverage_bin_for_sample(
            sample,
            self.metadata.get("coverage_bins", []),
            bin_size,
        )
        if coverage is None:
            return Prediction(
                legacy_press_ms,
                "legacy",
                legacy_press_ms,
                fallback_reason="outside_neural_coverage",
            )
        vector = feature_vector(sample)
        means = self.metadata["feature_means"]
        scales = self.metadata["feature_scales"]
        normalized = [(value - means[i]) / scales[i] for i, value in enumerate(vector)]
        with self.torch.no_grad():
            tensor = self.torch.tensor([normalized], dtype=self.torch.float32, device=self.device)
            raw = float(self.model(tensor).reshape(-1)[0].item())
        max_ratio = min(
            float(self.metadata["max_correction_ratio"]),
            float(settings["max_correction_ratio"]),
            float(settings["runtime_max_correction_ratio"]),
        )
        minimum = int(self.metadata.get("min_samples_per_coverage_bin", settings["min_samples_per_coverage_bin"]))
        strength = coverage_strength(int(coverage.get("samples", 0)), minimum)
        correction = math.tanh(raw) * max_ratio * strength
        corrected = apply_safety_limits(legacy_press_ms * (1.0 + correction), result, config)
        return Prediction(
            corrected,
            "neural",
            legacy_press_ms,
            correction,
            model_id=str(self.metadata.get("model_id")) if self.metadata.get("model_id") else None,
        )


def legacy_prediction(press_ms: float) -> Prediction:
    return Prediction(press_ms, "legacy", press_ms, 0.0)


def _split_samples(
    samples: list[dict[str, Any]],
    min_validation: int,
    bin_size_px: float,
    min_validation_per_bin: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    desired = max(min_validation, int(math.ceil(len(samples) * 0.20)))
    sessions: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        sessions.setdefault(str(sample.get("session_id", "unknown")), []).append(sample)
    session_keys = sorted(sessions, key=lambda key: (hashlib_sha(key), key))
    bucket_keys = {coverage_key(sample, bin_size_px) for sample in samples}
    if 1 < len(session_keys) <= 16:
        best: tuple[float, list[dict[str, Any]], list[dict[str, Any]]] | None = None
        for count in range(1, len(session_keys)):
            for selected in itertools.combinations(session_keys, count):
                selected_set = set(selected)
                validation = [sample for key in selected for sample in sessions[key]]
                training = [sample for key in session_keys if key not in selected_set for sample in sessions[key]]
                if len(validation) < min_validation or len(training) < min_validation:
                    continue
                valid_counts = {key: 0 for key in bucket_keys}
                train_counts = {key: 0 for key in bucket_keys}
                for sample in validation:
                    valid_counts[coverage_key(sample, bin_size_px)] += 1
                for sample in training:
                    train_counts[coverage_key(sample, bin_size_px)] += 1
                if any(valid_counts[key] < min_validation_per_bin for key in bucket_keys):
                    continue
                if any(train_counts[key] < min_validation_per_bin for key in bucket_keys):
                    continue
                score = abs(len(validation) - desired) + count * 0.001
                if best is None or score < best[0]:
                    best = (score, training, validation)
            if best is not None and best[0] < 1.0:
                break
        if best is not None:
            return best[1], best[2], "session_grouped_coverage_stratified"

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(coverage_key(sample, bin_size_px), []).append(sample)
    validation = []
    training = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda item: (hashlib_sha(str(item.get("sample_id", item.get("timestamp", "")))), str(item.get("timestamp", ""))),
        )
        take = max(min_validation_per_bin, int(round(len(ordered) * 0.20)))
        take = min(take, max(1, len(ordered) - min_validation_per_bin))
        validation.extend(ordered[:take])
        training.extend(ordered[take:])
    if len(validation) < min_validation:
        movable = sorted(training, key=lambda item: hashlib_sha(str(item.get("sample_id", ""))))
        for sample in movable:
            key = coverage_key(sample, bin_size_px)
            remaining = sum(coverage_key(item, bin_size_px) == key for item in training)
            if remaining <= min_validation_per_bin:
                continue
            training.remove(sample)
            validation.append(sample)
            if len(validation) >= min_validation:
                break
    return training, validation, "distance_direction_stratified"


def hashlib_sha(value: str) -> str:
    return hashlib.sha256(f"jumpjump-2026:{value}".encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise JumpAutoError(f"Could not read neural model weights '{path}': {exc}") from exc
    return digest.hexdigest()


def _feature_stats(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [sum(row[i] for row in rows) / len(rows) for i in range(len(FEATURE_NAMES))]
    scales = []
    for i, mean in enumerate(means):
        variance = sum((row[i] - mean) ** 2 for row in rows) / max(1, len(rows) - 1)
        scales.append(max(math.sqrt(variance), 1e-6))
    return means, scales


def _metrics(predictions: list[float], samples: list[dict[str, Any]]) -> dict[str, float | None]:
    targets = [_finite(sample["target_press_ms"]) for sample in samples]
    legacy = [_finite(sample["legacy_press_ms"]) for sample in samples]
    result: dict[str, float | None] = {
        "legacy_mae_ms": sum(abs(a - b) for a, b in zip(legacy, targets)) / len(samples),
        "model_mae_ms": sum(abs(a - b) for a, b in zip(predictions, targets)) / len(samples),
    }
    for name, predicate in (("left", lambda s: _finite(s.get("dx_px")) < 0), ("right", lambda s: _finite(s.get("dx_px")) > 0)):
        indices = [index for index, sample in enumerate(samples) if predicate(sample)]
        if not indices:
            result[f"{name}_legacy_mae_ms"] = None
            result[f"{name}_model_mae_ms"] = None
            continue
        result[f"{name}_legacy_mae_ms"] = sum(abs(legacy[i] - targets[i]) for i in indices) / len(indices)
        result[f"{name}_model_mae_ms"] = sum(abs(predictions[i] - targets[i]) for i in indices) / len(indices)
    return result


def model_passes_validation_gate(
    metrics: dict[str, float | None],
    min_improvement_ratio: float,
    max_direction_regression_ratio: float,
    bucket_metrics: dict[str, dict[str, float | None]] | None = None,
    harmful_correction_rate: float | None = None,
    max_harmful_correction_rate: float = 1.0,
) -> bool:
    legacy_mae = float(metrics.get("legacy_mae_ms") or 0.0)
    model_mae = float(metrics.get("model_mae_ms") or 0.0)
    accepted = (
        legacy_mae > 0
        and model_mae <= legacy_mae * (1.0 - min_improvement_ratio)
    )
    for direction in ("left", "right"):
        base = metrics.get(f"{direction}_legacy_mae_ms")
        candidate = metrics.get(f"{direction}_model_mae_ms")
        if (
            base is not None
            and candidate is not None
            and float(candidate)
            > max(1e-6, float(base)) * (1.0 + max_direction_regression_ratio)
        ):
            accepted = False
    for values in (bucket_metrics or {}).values():
        base = float(values.get("legacy_mae_ms") or 0.0)
        candidate = float(values.get("model_mae_ms") or 0.0)
        if base > 0 and candidate > base * (1.0 + max_direction_regression_ratio):
            accepted = False
    if harmful_correction_rate is not None and harmful_correction_rate > max_harmful_correction_rate:
        accepted = False
    return accepted


def eligible_supervised_samples(all_samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        sample
        for sample in valid_training_samples(all_samples)
        if sample.get("prediction_source") == "legacy"
        and not bool(sample.get("imported_from_config"))
    ]


def failure_constraint_samples(
    all_samples: Iterable[dict[str, Any]],
    coverage_bins: list[dict[str, Any]],
    bin_size_px: float,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for sample in all_samples:
        if sample.get("prediction_source") != "legacy":
            continue
        if sample.get("result_type") != "auto_out_of_tolerance":
            continue
        if not uses_current_landing_measurement(sample):
            continue
        if _finite(sample.get("landing_label_confidence")) < 0.55:
            continue
        if _finite(sample.get("projection_ratio")) < 0.45:
            continue
        if abs(_finite(sample.get("signed_landing_error_px"))) < 1.0:
            continue
        if coverage_bin_for_sample(sample, coverage_bins, bin_size_px) is None:
            continue
        constraints.append(sample)
    return constraints


def _bucket_metrics(
    predictions: list[float],
    samples: list[dict[str, Any]],
    bin_size_px: float,
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, tuple[list[float], list[dict[str, Any]]]] = {}
    for prediction, sample in zip(predictions, samples):
        key = coverage_key(sample, bin_size_px)
        prediction_rows, sample_rows = grouped.setdefault(key, ([], []))
        prediction_rows.append(prediction)
        sample_rows.append(sample)
    return {key: _metrics(values[0], values[1]) for key, values in sorted(grouped.items())}


def _harmful_correction_metrics(
    ratios: list[float],
    samples: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    if not samples:
        return {"samples": 0, "rate": None, "mean_harmful_ratio": None}
    harmful = [
        max(0.0, ratio * (1.0 if _finite(sample.get("signed_landing_error_px")) > 0 else -1.0))
        for ratio, sample in zip(ratios, samples)
    ]
    return {
        "samples": len(samples),
        "rate": sum(value > 0.002 for value in harmful) / len(harmful),
        "mean_harmful_ratio": sum(harmful) / len(harmful),
    }


def _baseline_landing_metrics(
    all_samples: Iterable[dict[str, Any]],
    coverage_bins: list[dict[str, Any]],
    bin_size_px: float,
) -> dict[str, float | int | None]:
    measured = [
        sample
        for sample in all_samples
        if sample.get("prediction_source") == "legacy"
        and uses_current_landing_measurement(sample)
        and sample.get("landing_error_px") is not None
        and coverage_bin_for_sample(sample, coverage_bins, bin_size_px) is not None
    ]
    errors = [_finite(sample.get("landing_error_px")) for sample in measured]
    if not errors:
        return {"samples": 0, "median_error_px": None, "success_rate": None, "bins": {}}
    grouped: dict[str, list[float]] = {}
    for sample in measured:
        grouped.setdefault(coverage_key(sample, bin_size_px), []).append(
            _finite(sample.get("landing_error_px"))
        )
    return {
        "samples": len(errors),
        "median_error_px": statistics.median(errors),
        "success_rate": sum(error <= 80.0 for error in errors) / len(errors),
        "bins": {
            key: {
                "samples": len(values),
                "median_error_px": statistics.median(values),
                "success_rate": sum(error <= 80.0 for error in values) / len(values),
            }
            for key, values in sorted(grouped.items())
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def train_press_model(
    all_samples: Iterable[dict[str, Any]],
    config: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    settings = neural_press_model_config(config)
    settings["feature_version"] = FEATURE_VERSION
    all_rows = list(all_samples)
    samples = eligible_supervised_samples(all_rows)
    minimum = int(settings["min_training_samples"])
    min_validation = int(settings["min_validation_samples"])
    if len(samples) < minimum:
        raise JumpAutoError(f"Need at least {minimum} valid neural samples; found {len(samples)}.")
    bin_size = float(settings["coverage_bin_size_px"])
    min_per_bin = int(settings["min_samples_per_coverage_bin"])
    coverage_bins = build_coverage_bins(samples, bin_size, min_per_bin)
    covered_samples = [
        sample
        for sample in samples
        if coverage_bin_for_sample(sample, coverage_bins, bin_size) is not None
    ]
    minimum_covered = max(60, min_validation * 3)
    if len(covered_samples) < minimum_covered:
        raise JumpAutoError(
            f"Need at least {minimum_covered} samples inside well-covered direction/distance "
            f"bins; found {len(covered_samples)}."
        )
    training, validation, split_strategy = _split_samples(
        covered_samples,
        min_validation,
        bin_size,
        int(settings["min_validation_samples_per_bin"]),
    )
    if len(validation) < min_validation:
        raise JumpAutoError(f"Need at least {min_validation} validation samples; found {len(validation)}.")
    constraints = failure_constraint_samples(all_rows, coverage_bins, bin_size)
    validation_sessions = {str(sample.get("session_id", "unknown")) for sample in validation}
    if split_strategy.startswith("session_grouped"):
        validation_constraints = [
            sample for sample in constraints if str(sample.get("session_id", "unknown")) in validation_sessions
        ]
    else:
        validation_constraints = [
            sample
            for sample in constraints
            if int(hashlib_sha(str(sample.get("sample_id", "0")))[:8], 16) % 5 == 0
        ]
    validation_constraint_ids = {id(sample) for sample in validation_constraints}
    training_constraints = [sample for sample in constraints if id(sample) not in validation_constraint_ids]

    torch = import_torch()
    random.seed(20260712)
    torch.manual_seed(20260712)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260712)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = [feature_vector(sample) for sample in training]
    means, scales = _feature_stats(train_rows)

    def normalized(rows: list[dict[str, Any]]):
        return [[(value - means[i]) / scales[i] for i, value in enumerate(feature_vector(row))] for row in rows]

    max_ratio = min(
        float(settings["max_correction_ratio"]),
        float(settings["runtime_max_correction_ratio"]),
    )
    min_bin_samples = int(settings["min_samples_per_coverage_bin"])

    def strengths(rows: list[dict[str, Any]]) -> list[float]:
        values = []
        for row in rows:
            coverage = coverage_bin_for_sample(row, coverage_bins, bin_size)
            values.append(
                coverage_strength(int(coverage["samples"]), min_bin_samples)
                if coverage is not None
                else 0.0
            )
        return values

    x_train = torch.tensor(normalized(training), dtype=torch.float32, device=device)
    y_train = torch.tensor(
        [clamp((_finite(s["target_press_ms"]) / _finite(s["legacy_press_ms"])) - 1.0, -max_ratio, max_ratio) for s in training],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    train_strength = torch.tensor(strengths(training), dtype=torch.float32, device=device).reshape(-1, 1)
    x_validation = torch.tensor(normalized(validation), dtype=torch.float32, device=device)
    validation_strength = torch.tensor(
        strengths(validation), dtype=torch.float32, device=device
    ).reshape(-1, 1)
    if training_constraints:
        x_constraints = torch.tensor(
            normalized(training_constraints), dtype=torch.float32, device=device
        )
        constraint_strength = torch.tensor(
            strengths(training_constraints), dtype=torch.float32, device=device
        ).reshape(-1, 1)
        constraint_sign = torch.tensor(
            [1.0 if _finite(sample.get("signed_landing_error_px")) > 0 else -1.0 for sample in training_constraints],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1)
    else:
        x_constraints = constraint_strength = constraint_sign = None
    if validation_constraints:
        x_validation_constraints = torch.tensor(
            normalized(validation_constraints), dtype=torch.float32, device=device
        )
        validation_constraint_strength = torch.tensor(
            strengths(validation_constraints), dtype=torch.float32, device=device
        ).reshape(-1, 1)
        validation_constraint_sign = torch.tensor(
            [1.0 if _finite(sample.get("signed_landing_error_px")) > 0 else -1.0 for sample in validation_constraints],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1)
    else:
        x_validation_constraints = validation_constraint_strength = validation_constraint_sign = None
    model = _network(torch, len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    loss_fn = torch.nn.HuberLoss(delta=0.03)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    epochs = 0
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        prediction = torch.tanh(model(x_train)) * max_ratio * train_strength
        loss = loss_fn(prediction, y_train)
        if x_constraints is not None:
            constraint_prediction = torch.tanh(model(x_constraints)) * max_ratio * constraint_strength
            loss = loss + float(settings["failure_constraint_weight"]) * torch.relu(
                constraint_prediction * constraint_sign
            ).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_ratios = torch.tanh(model(x_validation)) * max_ratio * validation_strength
            validation_targets = torch.tensor(
                [clamp((_finite(s["target_press_ms"]) / _finite(s["legacy_press_ms"])) - 1.0, -max_ratio, max_ratio) for s in validation],
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 1)
            validation_loss = float(loss_fn(validation_ratios, validation_targets).item())
            if x_validation_constraints is not None:
                validation_constraint_prediction = (
                    torch.tanh(model(x_validation_constraints))
                    * max_ratio
                    * validation_constraint_strength
                )
                validation_loss += float(settings["failure_constraint_weight"]) * float(
                    torch.relu(
                        validation_constraint_prediction * validation_constraint_sign
                    ).mean().item()
                )
        epochs = epoch + 1
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 40:
                break
    if best_state is None:
        raise JumpAutoError("Neural training did not produce a usable checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        ratios = (
            torch.tanh(model(x_validation)) * max_ratio * validation_strength
        ).reshape(-1).cpu().tolist()
        if x_validation_constraints is not None:
            constraint_ratios = (
                torch.tanh(model(x_validation_constraints))
                * max_ratio
                * validation_constraint_strength
            ).reshape(-1).cpu().tolist()
        else:
            constraint_ratios = []
    predicted_press = [_finite(sample["legacy_press_ms"]) * (1.0 + float(ratio)) for sample, ratio in zip(validation, ratios)]
    metrics = _metrics(predicted_press, validation)
    bucket_metrics = _bucket_metrics(predicted_press, validation, bin_size)
    harmful_metrics = _harmful_correction_metrics(constraint_ratios, validation_constraints)
    accepted = model_passes_validation_gate(
        metrics,
        float(settings["min_mae_improvement_ratio"]),
        float(settings["max_direction_regression_ratio"]),
        bucket_metrics,
        harmful_metrics["rate"] if isinstance(harmful_metrics["rate"], float) else None,
        float(settings["max_harmful_correction_rate"]),
    )
    metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_means": means,
        "feature_scales": scales,
        "max_correction_ratio": max_ratio,
        "coverage_bin_size_px": bin_size,
        "min_samples_per_coverage_bin": min_bin_samples,
        "coverage_bins": coverage_bins,
        "split_strategy": split_strategy,
        "eligible_supervised_samples": len(samples),
        "covered_supervised_samples": len(covered_samples),
        "excluded_outside_coverage": len(samples) - len(covered_samples),
        "training_failure_constraints": len(training_constraints),
        "validation_failure_constraints": len(validation_constraints),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "epochs": epochs,
        "device": str(device),
        "accepted": accepted,
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "harmful_correction_metrics": harmful_metrics,
        "baseline_landing": _baseline_landing_metrics(all_rows, coverage_bins, bin_size),
    }
    if accepted:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(f".{model_path.name}.tmp")
        try:
            torch.save(best_state, temporary)
            os.replace(temporary, model_path)
        finally:
            temporary.unlink(missing_ok=True)
        metadata["weights_sha256"] = _file_sha256(model_path)
        metadata["model_id"] = metadata["weights_sha256"][:16]
        _atomic_json(metadata_path, metadata)
    else:
        settings["enabled"] = False
    if accepted:
        settings["enabled"] = True
    settings["training_metrics"] = metadata
    return metadata


def evaluate_press_model(
    all_samples: Iterable[dict[str, Any]],
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    all_rows = list(all_samples)
    predictor = NeuralPressPredictor.load(model_path, metadata_path)
    bin_size = float(predictor.metadata["coverage_bin_size_px"])
    coverage_bins = predictor.metadata["coverage_bins"]
    samples = [
        sample
        for sample in eligible_supervised_samples(all_rows)
        if coverage_bin_for_sample(sample, coverage_bins, bin_size) is not None
    ]
    if not samples:
        raise JumpAutoError("No valid samples are available for neural model evaluation.")
    predictions: list[float] = []
    for sample in samples:
        vector = feature_vector(sample)
        means = predictor.metadata["feature_means"]
        scales = predictor.metadata["feature_scales"]
        normalized = [(value - means[i]) / scales[i] for i, value in enumerate(vector)]
        with predictor.torch.no_grad():
            tensor = predictor.torch.tensor([normalized], dtype=predictor.torch.float32, device=predictor.device)
            raw = float(predictor.model(tensor).reshape(-1)[0].item())
        coverage = coverage_bin_for_sample(sample, coverage_bins, bin_size)
        strength = coverage_strength(
            int(coverage["samples"]),
            int(predictor.metadata["min_samples_per_coverage_bin"]),
        )
        ratio = math.tanh(raw) * float(predictor.metadata["max_correction_ratio"]) * strength
        predictions.append(_finite(sample["legacy_press_ms"]) * (1.0 + ratio))
    return {
        "samples": len(samples),
        **_metrics(predictions, samples),
        "validation_metrics": predictor.metadata.get("metrics", {}),
        "coverage_bins": coverage_bins,
        "landing_comparison": landing_comparison(
            all_rows,
            model_id=predictor.metadata.get("model_id"),
        ),
    }


def online_guard_decision(
    landing_errors: Iterable[float | dict[str, Any]],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    settings = neural_press_model_config(config)
    window = int(settings["online_guard_window_jumps"])
    observations: list[dict[str, Any]] = []
    for value in landing_errors:
        if isinstance(value, dict):
            error = _finite(value.get("landing_error_px"), float("nan"))
            key = value.get("coverage_key")
        else:
            error = _finite(value, float("nan"))
            key = None
        if math.isfinite(error):
            observations.append({"landing_error_px": error, "coverage_key": key})
    observations = observations[-window:]
    errors = [float(item["landing_error_px"]) for item in observations]
    minimum = int(settings["online_guard_min_jumps"])
    if len(errors) < minimum:
        return False, {"status": "collecting", "samples": len(errors), "required": minimum}
    baseline = metadata.get("baseline_landing", {})
    baseline_median = _finite(baseline.get("median_error_px"), -1.0)
    baseline_success = _finite(baseline.get("success_rate"), -1.0)
    current_median = statistics.median(errors)
    current_success = sum(error <= 80.0 for error in errors) / len(errors)
    bin_baselines = baseline.get("bins", {})
    matched_bin_rows = [
        (item, bin_baselines.get(item.get("coverage_key")))
        for item in observations
        if item.get("coverage_key") in bin_baselines
    ]
    if len(matched_bin_rows) >= minimum:
        expected_success = sum(float(values["success_rate"]) for _, values in matched_bin_rows) / len(matched_bin_rows)
        normalized_median = statistics.median(
            float(item["landing_error_px"]) / max(1.0, float(values["median_error_px"]))
            for item, values in matched_bin_rows
        )
        median_limit = 1.0 + float(settings["online_guard_max_median_regression_ratio"])
        success_floor = expected_success - float(settings["online_guard_max_success_rate_drop"])
        disable = normalized_median > median_limit or current_success < success_floor
        return disable, {
            "status": "disable" if disable else "healthy",
            "samples": len(errors),
            "median_error_px": current_median,
            "median_limit_px": None,
            "normalized_median_ratio": normalized_median,
            "normalized_median_limit": median_limit,
            "success_rate": current_success,
            "success_rate_floor": success_floor,
            "baseline_mode": "direction_distance_mix",
        }
    median_limit = (
        baseline_median * (1.0 + float(settings["online_guard_max_median_regression_ratio"]))
        if baseline_median >= 0
        else float("inf")
    )
    success_floor = (
        baseline_success - float(settings["online_guard_max_success_rate_drop"])
        if baseline_success >= 0
        else 0.0
    )
    disable = current_median > median_limit or current_success < success_floor
    return disable, {
        "status": "disable" if disable else "healthy",
        "samples": len(errors),
        "median_error_px": current_median,
        "median_limit_px": median_limit,
        "success_rate": current_success,
        "success_rate_floor": success_floor,
        "baseline_mode": "global",
    }


def landing_comparison(
    all_samples: Iterable[dict[str, Any]],
    minimum_pairs: int = 30,
    distance_tolerance_px: float = 20.0,
    model_id: str | None = None,
) -> dict[str, Any]:
    rows = [
        sample
        for sample in all_samples
        if sample.get("prediction_source") in {"legacy", "neural"}
        and _finite(sample.get("landing_error_px"), -1.0) >= 0
        and _finite(sample.get("dx_px")) != 0
    ]
    neural = [
        sample
        for sample in rows
        if sample.get("prediction_source") == "neural"
        and (model_id is None or sample.get("prediction_model_id") == model_id)
    ]
    legacy = [sample for sample in rows if sample.get("prediction_source") == "legacy"]
    unused = set(range(len(legacy)))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in neural:
        direction = -1 if _finite(candidate.get("dx_px")) < 0 else 1
        distance = _finite(candidate.get("effective_distance_px"))
        choices = [
            index
            for index in unused
            if (-1 if _finite(legacy[index].get("dx_px")) < 0 else 1) == direction
            and abs(_finite(legacy[index].get("effective_distance_px")) - distance) <= distance_tolerance_px
        ]
        if not choices:
            continue
        match = min(
            choices,
            key=lambda index: abs(_finite(legacy[index].get("effective_distance_px")) - distance),
        )
        unused.remove(match)
        pairs.append((candidate, legacy[match]))
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "matched_pairs": len(pairs),
        "minimum_pairs": minimum_pairs,
        "model_id": model_id,
    }
    if len(pairs) < minimum_pairs:
        return result
    neural_errors = [_finite(pair[0]["landing_error_px"]) for pair in pairs]
    legacy_errors = [_finite(pair[1]["landing_error_px"]) for pair in pairs]
    neural_median = statistics.median(neural_errors)
    legacy_median = statistics.median(legacy_errors)
    neural_success = sum(error <= 80.0 for error in neural_errors) / len(pairs)
    legacy_success = sum(error <= 80.0 for error in legacy_errors) / len(pairs)
    improvement = 0.0 if legacy_median <= 0 else (legacy_median - neural_median) / legacy_median
    result.update(
        {
            "status": "complete",
            "neural_median_landing_error_px": neural_median,
            "legacy_median_landing_error_px": legacy_median,
            "median_improvement_ratio": improvement,
            "neural_success_rate": neural_success,
            "legacy_success_rate": legacy_success,
            "accepted": improvement >= 0.15 and neural_success >= legacy_success,
        }
    )
    return result
