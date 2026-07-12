from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .types import DetectionResult
from .utils import timestamp


DATASET_SCHEMA_VERSION = 2


def resolve_runtime_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else config_path.parent / path


def sample_id(sample: dict[str, Any]) -> str:
    stable = {
        key: sample.get(key)
        for key in (
            "timestamp",
            "session_id",
            "dx_px",
            "dy_px",
            "executed_press_ms",
            "result_type",
        )
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave the last append incomplete. Earlier rows stay usable.
                continue
            if isinstance(value, dict):
                samples.append(value)
    return samples


def append_sample(path: Path, sample: dict[str, Any]) -> bool:
    record = dict(sample)
    record.setdefault("schema_version", DATASET_SCHEMA_VERSION)
    record.setdefault("sample_id", sample_id(record))
    if record["sample_id"] in {item.get("sample_id") for item in load_samples(path)}:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) not in {b"\n", b"\r"}
    with path.open("a", encoding="utf-8", newline="\n") as file:
        if needs_separator:
            file.write("\n")
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return True


def _bbox_dimensions(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return float(bbox[2]), float(bbox[3])


def jump_record(
    result: DetectionResult,
    *,
    session_id: str,
    viewport_size: tuple[int, int],
    legacy_press_ms: float,
    executed_press_ms: float,
    prediction_source: str,
    prediction_model_id: str | None = None,
    result_type: str,
    landing_error_px: float | None = None,
    target_press_ms: float | None = None,
    signed_landing_error_px: float | None = None,
    projection_ratio: float | None = None,
    landing_label_method: str | None = None,
    landing_label_confidence: float | None = None,
    landing_reference: tuple[int, int] | None = None,
    landing_platform_bbox: tuple[int, int, int, int] | None = None,
    trainable: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    piece_width, piece_height = _bbox_dimensions(result.piece_bbox)
    target_width, target_height = _bbox_dimensions(result.target_bbox)
    width, height = viewport_size
    record: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "timestamp": timestamp(),
        "session_id": session_id,
        "viewport_width_px": width,
        "viewport_height_px": height,
        "dx_px": result.dx_px,
        "dy_px": result.dy_px,
        "screen_distance_px": result.screen_distance_px,
        "effective_distance_px": result.effective_distance_px,
        "piece_width_px": piece_width,
        "piece_height_px": piece_height,
        "target_width_px": target_width,
        "target_height_px": target_height,
        "confidence": result.confidence,
        "legacy_press_ms": legacy_press_ms,
        "executed_press_ms": executed_press_ms,
        "target_press_ms": target_press_ms,
        "landing_error_px": landing_error_px,
        "signed_landing_error_px": signed_landing_error_px,
        "projection_ratio": projection_ratio,
        "landing_label_method": landing_label_method,
        "landing_label_confidence": landing_label_confidence,
        "landing_reference_x_px": landing_reference[0] if landing_reference else None,
        "landing_reference_y_px": landing_reference[1] if landing_reference else None,
        "landing_platform_bbox": landing_platform_bbox,
        "prediction_source": prediction_source,
        "prediction_model_id": prediction_model_id,
        "result_type": result_type,
        "trainable": bool(trainable),
    }
    if reason:
        record["reason"] = reason
    record["sample_id"] = sample_id(record)
    return record


def import_legacy_samples(path: Path, samples: Iterable[dict[str, Any]]) -> int:
    imported = 0
    for old in samples:
        try:
            press_ms = float(
                old.get("training_press_ms")
                or old.get("center_adjusted_press_ms")
                or old["press_ms"]
            )
            dx = float(old["dx_px"])
            dy = float(old["dy_px"])
            confidence = float(old.get("confidence", 0.75))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (press_ms, dx, dy, confidence)) or press_ms <= 0:
            continue
        record = {
            "schema_version": 1,
            "timestamp": str(old.get("timestamp") or timestamp()),
            "session_id": "legacy-config",
            "viewport_width_px": 1080,
            "viewport_height_px": 1920,
            "dx_px": dx,
            "dy_px": dy,
            "screen_distance_px": float(old.get("screen_distance_px", math.hypot(dx, dy))),
            "effective_distance_px": float(old.get("effective_distance_px", old.get("distance_px", math.hypot(dx, dy)))),
            "piece_width_px": float(old.get("piece_width_px", 0.0)),
            "piece_height_px": float(old.get("piece_height_px", 0.0)),
            "target_width_px": float(old.get("target_width_px", 0.0)),
            "target_height_px": float(old.get("target_height_px", 0.0)),
            "confidence": confidence,
            "legacy_press_ms": float(old.get("press_ms", press_ms)),
            "executed_press_ms": float(old.get("press_ms", press_ms)),
            "target_press_ms": press_ms,
            "landing_error_px": old.get("landing_error_px"),
            "signed_landing_error_px": old.get("signed_landing_error_px"),
            "projection_ratio": old.get("projection_ratio", 1.0),
            "prediction_source": "legacy",
            "result_type": str(old.get("result_type", "manual")),
            "trainable": str(old.get("result_type", "manual")) == "manual",
            "imported_from_config": True,
        }
        record["sample_id"] = sample_id(record)
        imported += int(append_sample(path, record))
    return imported


def valid_training_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    required = (
        "viewport_width_px",
        "viewport_height_px",
        "dx_px",
        "dy_px",
        "effective_distance_px",
        "confidence",
        "legacy_press_ms",
        "target_press_ms",
    )
    valid: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.get("trainable"):
            continue
        result_type = str(sample.get("result_type", ""))
        try:
            schema_version = int(sample.get("schema_version", 1))
        except (TypeError, ValueError):
            continue
        if result_type != "manual" and (
            schema_version < 2 or sample.get("landing_label_method") != "current_platform"
        ):
            continue
        try:
            values = [float(sample[key]) for key in required]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values) and values[0] > 0 and values[1] > 0 and values[-2] > 0 and values[-1] > 0:
            valid.append(sample)
    return valid
