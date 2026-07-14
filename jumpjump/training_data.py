from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .config import CURRENT_AUTO_FEEDBACK_VERSION, MAX_AUTOMATIC_LANDING_ERROR_PX
from .types import DetectionResult
from .utils import timestamp


DATASET_SCHEMA_VERSION = 4
CURRENT_LANDING_LABEL_METHOD = "camera_stable_horizontal_v2"


FileSignature = tuple[int, int, int, int, int] | None


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


def _iter_samples(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
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
                yield value


def load_samples(path: Path) -> list[dict[str, Any]]:
    return list(_iter_samples(path))


def _sample_record(sample: dict[str, Any]) -> dict[str, Any]:
    record = dict(sample)
    record.setdefault("schema_version", DATASET_SCHEMA_VERSION)
    record.setdefault("sample_id", sample_id(record))
    return record


def _file_signature(path: Path) -> FileSignature:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _load_sample_ids(path: Path) -> set[Any]:
    return {item.get("sample_id") for item in _iter_samples(path)}


def _append_record(path: Path, record: dict[str, Any]) -> None:
    """Append and durably flush one prepared record."""
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


class SampleIdIndex:
    """Session-scoped duplicate index for one append-only JSONL dataset."""

    def __init__(self, path: Path):
        self.path = path
        self.sample_ids = _load_sample_ids(path)
        self._signature = _file_signature(path)

    def _refresh_if_changed(self) -> None:
        signature = _file_signature(self.path)
        if signature != self._signature:
            self.sample_ids = _load_sample_ids(self.path)
            self._signature = _file_signature(self.path)

    def append(self, sample: dict[str, Any]) -> bool:
        record = _sample_record(sample)
        self._refresh_if_changed()
        if record["sample_id"] in self.sample_ids:
            return False
        _append_record(self.path, record)
        # Do not mark a row as present until its durable append succeeds.
        self.sample_ids.add(record["sample_id"])
        self._signature = _file_signature(self.path)
        return True


def append_sample(path: Path, sample: dict[str, Any]) -> bool:
    """Append with an uncached duplicate check for standalone callers."""
    record = _sample_record(sample)
    if record["sample_id"] in _load_sample_ids(path):
        return False
    _append_record(path, record)
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
    landing_label_source: str | None = None,
    landing_reference: tuple[int, int] | None = None,
    landing_platform_bbox: tuple[int, int, int, int] | None = None,
    trainable: bool = False,
    reason: str | None = None,
    jump_index: int | None = None,
    physics_unit_press_ms: float | None = None,
    effective_press_coefficient: float | None = None,
) -> dict[str, Any]:
    piece_width, piece_height = _bbox_dimensions(result.piece_bbox)
    target_width, target_height = _bbox_dimensions(result.target_bbox)
    width, height = viewport_size
    record: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
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
        "piece_scale_ratio": result.piece_scale_ratio,
        "stage_bucket": result.stage_bucket,
        "stage_press_scale": result.stage_press_scale,
        "game_score": result.game_score,
        "game_score_confidence": result.game_score_confidence,
        "raw_game_score": result.raw_game_score,
        "raw_game_score_confidence": result.raw_game_score_confidence,
        "stage_score_confirmed": result.stage_score_confirmed,
        "jump_index": jump_index,
        "target_width_px": target_width,
        "target_height_px": target_height,
        "confidence": result.confidence,
        "legacy_press_ms": legacy_press_ms,
        "executed_press_ms": executed_press_ms,
        "physics_unit_press_ms": physics_unit_press_ms,
        "effective_press_coefficient": effective_press_coefficient,
        "target_press_ms": target_press_ms,
        "landing_error_px": landing_error_px,
        "signed_landing_error_px": signed_landing_error_px,
        "projection_ratio": projection_ratio,
        "landing_label_method": landing_label_method,
        "landing_label_confidence": landing_label_confidence,
        "landing_label_source": landing_label_source,
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


def import_legacy_samples(
    path: Path,
    samples: Iterable[dict[str, Any]],
    *,
    sample_index: SampleIdIndex | None = None,
) -> int:
    index = sample_index
    imported = 0
    for old in samples:
        if not isinstance(old, dict):
            continue
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
        if index is None:
            index = SampleIdIndex(path)
        imported += int(index.append(record))
    return imported


def uses_current_landing_measurement(sample: dict[str, Any]) -> bool:
    try:
        schema_version = int(sample.get("schema_version", 1))
    except (TypeError, ValueError):
        return False
    return (
        schema_version >= DATASET_SCHEMA_VERSION
        and sample.get("feedback_version") == CURRENT_AUTO_FEEDBACK_VERSION
        and sample.get("landing_label_method") == CURRENT_LANDING_LABEL_METHOD
    )


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
        if not isinstance(sample, dict):
            continue
        if not sample.get("trainable"):
            continue
        result_type = str(sample.get("result_type", ""))
        if result_type != "manual" and not uses_current_landing_measurement(sample):
            continue
        if result_type != "manual":
            raw_landing_error = sample.get("landing_error_px")
            try:
                landing_error = float(raw_landing_error)
            except (TypeError, ValueError):
                landing_error = math.nan
            # Automatic targets outside the runtime landing tolerance are not
            # precise labels.  Keep the immutable JSONL row for diagnostics,
            # but quarantine rows produced while an unsafe 500px tolerance was
            # active so they cannot train a press model later.
            if (
                not math.isfinite(landing_error)
                or landing_error < 0
                or landing_error > MAX_AUTOMATIC_LANDING_ERROR_PX
            ):
                continue
            try:
                piece_scale = float(sample["piece_scale_ratio"])
                stage_scale = float(sample["stage_press_scale"])
                physics_unit = float(sample["physics_unit_press_ms"])
                effective_coefficient = float(sample["effective_press_coefficient"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not isinstance(sample.get("stage_bucket"), str)
                or not sample["stage_bucket"]
                or sample.get("stage_score_confirmed") is not True
                or not all(
                    math.isfinite(value) and value > 0
                    for value in (
                        piece_scale,
                        stage_scale,
                        physics_unit,
                        effective_coefficient,
                    )
                )
            ):
                continue
        try:
            values = [float(sample[key]) for key in required]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values) and values[0] > 0 and values[1] > 0 and values[-2] > 0 and values[-1] > 0:
            valid.append(sample)
    return valid
