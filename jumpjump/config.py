from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .types import ConfigError


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_DIR / "jump_config.json"
DEFAULT_DEBUG_DIR = APP_DIR / "debug"
CURRENT_SCHEMA_VERSION = 3


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": CURRENT_SCHEMA_VERSION,
    "window_title": "",
    "window_keywords": ["跳一跳", "微信", "WeChat"],
    "min_client_width": 280,
    "min_client_height": 420,
    "crop": {
        "left_ratio": 0.0,
        "right_ratio": 1.0,
        "top_ratio": 0.10,
        "bottom_ratio": 0.92,
    },
    "press_ms_per_px": None,
    "press_model": {
        "type": "weighted_euclidean",
        "base_algorithm": "physics",
        "physics_press_coefficient": 1.392,
        "physics_head_diameter_px": None,
        # wangshub's formula uses the circular head diameter.  The detected
        # desktop piece bbox is already close to that width; 1.6 made the
        # inferred diameter far too large and therefore the press too short.
        "physics_piece_width_multiplier": 1.15,
        "physics_default_head_diameter_px": 80.0,
        "linear_reference_width_px": 1080,
        "linear_reference_coefficient": 1.390,
        "curve_correction_max_ratio": 0.35,
        "x_weight": 1.0,
        "y_weight": 1.0,
        "slope_ms_per_px": None,
        "offset_ms": 0.0,
        "fit_rmse_ms": None,
        "min_samples_for_weight_fit": 3,
        "max_samples": 40,
        "curve_enabled": True,
        "curve_min_samples": 3,
        "curve_points": [],
        "segment_size_px": 7,
        "segment_corrections": [],
        "max_segment_corrections": 120,
        "failure_caps": [],
        "max_failure_caps": 24,
        "short_hop_enabled": True,
        "short_hop_min_anchor_distance_px": 80,
        "samples": [],
    },
    "neural_press_model": {
        "enabled": False,
        "dataset_path": "data/jump_samples.jsonl",
        "model_path": "models/press_residual.pt",
        "metadata_path": "models/press_residual.json",
        "feature_version": 2,
        "min_training_samples": 100,
        "min_validation_samples": 20,
        "max_correction_ratio": 0.15,
        "runtime_max_correction_ratio": 0.08,
        "coverage_bin_size_px": 100,
        "min_samples_per_coverage_bin": 12,
        "min_validation_samples_per_bin": 2,
        "failure_constraint_weight": 0.50,
        "max_harmful_correction_rate": 0.25,
        "min_mae_improvement_ratio": 0.10,
        "max_direction_regression_ratio": 0.10,
        "online_guard_min_jumps": 20,
        "online_guard_window_jumps": 30,
        "online_guard_max_median_regression_ratio": 0.15,
        "online_guard_max_success_rate_drop": 0.05,
        "training_metrics": {},
    },
    "min_press_ms": 180,
    "max_press_ms": 1800,
    "confidence_threshold": 0.45,
    "click_point": {
        "x_ratio": 0.50,
        "y_ratio": 0.82,
    },
    "piece": {
        "hsv_lower": [95, 50, 25],
        "hsv_upper": [140, 255, 200],
        "fallback_hsv_lower": [85, 45, 15],
        "fallback_hsv_upper": [165, 255, 205],
        "search_top_ratio": 0.25,
        "search_bottom_ratio": 0.98,
        "min_area": 500,
        "max_area": 16000,
        "min_width": 24,
        "min_height": 18,
        "max_width_ratio": 0.18,
        "min_height_width_ratio": 1.05,
        "edge_reject_px": 4,
        "preferred_hue": 122,
        "preferred_hue_tolerance": 38,
        "min_median_saturation": 45,
        "preferred_max_value": 155,
        "core_value_upper": 165,
        "dynamic_color_enabled": True,
        "dynamic_color_min_samples": 2,
        "dynamic_color_max_samples": 24,
        "dynamic_color_hue_margin": 14,
        "dynamic_color_saturation_margin": 55,
        "dynamic_color_value_margin": 48,
        "color_samples": [],
        "foot_offset_px": 8,
    },
    "target": {
        "diff_threshold": 14,
        "search_top_ratio": 0.18,
        "search_bottom_extra_ratio": 0.10,
        "side_gap_ratio": 0.06,
        "exclude_piece_pad_px": 28,
        "min_area": 180,
        "max_area_ratio": 0.20,
        "min_component_fill_ratio": 0.035,
        "min_width": 18,
        "min_height": 10,
        "min_distance_ratio": 0.10,
        "current_platform_exclude_pad_ratio": 0.25,
        "current_platform_edge_gap_ratio": 0.18,
        "current_platform_max_distance_ratio": 0.26,
        "current_platform_color_tolerance_lab": 28,
        "current_platform_color_max_distance_ratio": 0.14,
        "current_platform_near_distance_ratio": 0.22,
        "current_platform_horizontal_band_ratio": 0.055,
        "current_platform_risk_confidence_scale": 0.24,
        "current_platform_band_confidence_scale": 0.34,
        "current_platform_color_confidence_scale": 0.55,
        "current_platform_side_sample_px": 34,
        "current_platform_max_target_above_piece_ratio": 0.06,
        "strategy_accept_confidence": 0.45,
        "max_aspect_ratio": 3.0,
        "max_surface_aspect_ratio": 2.6,
        "max_target_y_below_piece_ratio": 0.08,
        "center_y_ratio": 0.40,
        "top_surface_seed_ratio": 0.20,
        "top_surface_max_height_ratio": 0.72,
        "top_surface_color_tolerance": 22,
        "top_surface_hue_tolerance": 18,
        "top_surface_saturation_tolerance": 72,
        "top_surface_value_tolerance": 52,
        "top_surface_min_saturation_for_hue": 24,
        "top_surface_max_height_to_width": 0.68,
        "top_surface_min_area": 60,
        "min_surface_fill_ratio": 0.12,
        "edge_touch_confidence_scale": 0.86,
        "multi_edge_touch_confidence_scale": 0.68,
        "far_edge_surface_focus_width_ratio": 0.48,
        "far_edge_surface_focus_trim_ratio": 0.30,
        "top_surface_center_y_ratio": 0.50,
    },
    "overlay": {
        "dark_gray_threshold": 88,
        "min_dark_area_ratio": 0.055,
        "min_dark_width_ratio": 0.55,
        "min_dark_height_ratio": 0.12,
        "min_dark_fill_ratio": 0.55,
    },
    "debug": {
        "auto_capture_policy": "failures_and_rechecks",
        "max_files": 200,
        "max_size_mb": 100,
    },
    "auto_tuning": {
        "enabled": True,
        "landing_tolerance_px": 80,
        "landing_platform_min_confidence": 0.55,
        "center_deadzone_px": 8,
        "center_learning_enabled": True,
        "center_learning_rate": 0.65,
        "center_max_adjustment_ratio": 0.10,
        "center_projection_min_ratio": 0.45,
        "segment_correction_enabled": True,
        "segment_correction_learning_rate": 0.45,
        "segment_correction_success_decay": 0.08,
        "segment_max_correction_ratio": 0.12,
        "segment_precision_px": 8,
        "segment_precision_hits_to_freeze": 3,
        "segment_unfreeze_error_px": 18,
        "run_confidence_floor": 0.35,
        "low_confidence_recheck_delay_s": 0.15,
        "recheck_piece_tolerance_ratio": 0.015,
        "recheck_target_tolerance_ratio": 0.025,
        "max_recognition_failures_before_pause": 3,
        "min_confidence": 0.60,
        "save_every_success": True,
        # An overlay only tells us that the jump failed; it cannot tell an
        # overshoot from an undershoot.  Turning it into an upper press limit
        # poisons all nearby distances after a single failure.
        "failure_learning_enabled": False,
        "failure_shrink_ratio": 0.92,
        "failure_cap_window_ratio": 0.16,
        "failure_cap_min_window_px": 42,
    },
}


_OPTIONAL_NUMBER_PATHS = {
    "press_ms_per_px",
    "press_model.physics_head_diameter_px",
    "press_model.slope_ms_per_px",
    "press_model.fit_rmse_ms",
}


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_known_types(value: Any, default: Any, path: str) -> None:
    if default is None:
        if value is not None and (path not in _OPTIONAL_NUMBER_PATHS or not _is_number(value)):
            raise ConfigError(f"Invalid config value at '{path}': expected a number or null.")
        return
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"Invalid config value at '{path}': expected an object.")
        for key, child_default in default.items():
            child_path = f"{path}.{key}" if path else key
            if key not in value:
                raise ConfigError(f"Missing required config value at '{child_path}'.")
            _validate_known_types(value[key], child_default, child_path)
        return
    if isinstance(default, list):
        if not isinstance(value, list):
            raise ConfigError(f"Invalid config value at '{path}': expected an array.")
        if default:
            exemplar = default[0]
            for index, item in enumerate(value):
                _validate_known_types(item, exemplar, f"{path}[{index}]")
        return
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"Invalid config value at '{path}': expected true or false.")
        return
    if isinstance(default, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"Invalid config value at '{path}': expected an integer.")
        return
    if isinstance(default, float):
        if not _is_number(value):
            raise ConfigError(f"Invalid config value at '{path}': expected a number.")
        return
    if not isinstance(value, type(default)):
        raise ConfigError(
            f"Invalid config value at '{path}': expected {type(default).__name__}."
        )


def _number_at(config: dict[str, Any], path: str) -> float:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    if not _is_number(value):
        raise ConfigError(f"Invalid config value at '{path}': expected a number.")
    return float(value)


def _require_range(
    config: dict[str, Any],
    path: str,
    low: float,
    high: float,
) -> float:
    value = _number_at(config, path)
    if value < low or value > high:
        raise ConfigError(
            f"Invalid config value at '{path}': expected {low:g} <= value <= {high:g}."
        )
    return value


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a JSON object.")
    _validate_known_types(config, DEFAULT_CONFIG, "")

    version = config.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"Unsupported config schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION}."
        )

    for path in (
        "min_client_width",
        "min_client_height",
        "min_press_ms",
        "max_press_ms",
        "debug.max_files",
        "debug.max_size_mb",
    ):
        if _number_at(config, path) <= 0:
            raise ConfigError(f"Invalid config value at '{path}': expected a positive value.")

    if _number_at(config, "min_press_ms") > _number_at(config, "max_press_ms"):
        raise ConfigError("Invalid config: min_press_ms must not exceed max_press_ms.")

    crop_left = _require_range(config, "crop.left_ratio", 0.0, 1.0)
    crop_right = _require_range(config, "crop.right_ratio", 0.0, 1.0)
    crop_top = _require_range(config, "crop.top_ratio", 0.0, 1.0)
    crop_bottom = _require_range(config, "crop.bottom_ratio", 0.0, 1.0)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        raise ConfigError("Invalid config: crop left/top ratios must be below right/bottom ratios.")

    _require_range(config, "click_point.x_ratio", 0.0, 1.0)
    _require_range(config, "click_point.y_ratio", 0.0, 1.0)
    threshold = _require_range(config, "confidence_threshold", 0.0, 1.0)
    run_floor = _require_range(config, "auto_tuning.run_confidence_floor", 0.0, 1.0)
    _require_range(config, "auto_tuning.min_confidence", 0.0, 1.0)
    _require_range(config, "auto_tuning.landing_platform_min_confidence", 0.0, 1.0)
    if run_floor > threshold:
        raise ConfigError(
            "Invalid config: auto_tuning.run_confidence_floor must not exceed "
            "confidence_threshold."
        )
    _require_range(config, "auto_tuning.low_confidence_recheck_delay_s", 0.0, 5.0)
    _require_range(config, "auto_tuning.recheck_piece_tolerance_ratio", 0.0, 0.25)
    _require_range(config, "auto_tuning.recheck_target_tolerance_ratio", 0.0, 0.25)
    _require_range(config, "overlay.min_dark_fill_ratio", 0.0, 1.0)
    _require_range(config, "neural_press_model.max_correction_ratio", 0.0, 0.50)
    _require_range(config, "neural_press_model.runtime_max_correction_ratio", 0.0, 0.25)
    _require_range(config, "neural_press_model.failure_constraint_weight", 0.0, 5.0)
    _require_range(config, "neural_press_model.max_harmful_correction_rate", 0.0, 1.0)
    _require_range(config, "neural_press_model.min_mae_improvement_ratio", 0.0, 1.0)
    _require_range(config, "neural_press_model.max_direction_regression_ratio", 0.0, 1.0)
    _require_range(config, "neural_press_model.online_guard_max_median_regression_ratio", 0.0, 2.0)
    _require_range(config, "neural_press_model.online_guard_max_success_rate_drop", 0.0, 1.0)
    for path in (
        "neural_press_model.feature_version",
        "neural_press_model.min_training_samples",
        "neural_press_model.min_validation_samples",
        "neural_press_model.coverage_bin_size_px",
        "neural_press_model.min_samples_per_coverage_bin",
        "neural_press_model.min_validation_samples_per_bin",
        "neural_press_model.online_guard_min_jumps",
        "neural_press_model.online_guard_window_jumps",
    ):
        if _number_at(config, path) <= 0:
            raise ConfigError(f"Invalid config value at '{path}': expected a positive value.")

    policy = config["debug"]["auto_capture_policy"]
    if policy not in {"failures", "failures_and_rechecks", "all"}:
        raise ConfigError(
            "Invalid config value at 'debug.auto_capture_policy': expected "
            "'failures', 'failures_and_rechecks', or 'all'."
        )

    for path in ("piece.hsv_lower", "piece.hsv_upper", "piece.fallback_hsv_lower", "piece.fallback_hsv_upper"):
        values: Any = config
        for key in path.split("."):
            values = values[key]
        if len(values) != 3 or any(not _is_number(item) for item in values):
            raise ConfigError(f"Invalid config value at '{path}': expected three numbers.")

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _migrate_config(user_config: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(user_config)
    version = migrated.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ConfigError("Configuration schema_version must be a non-negative integer.")
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"Configuration schema_version {version} is newer than supported version "
            f"{CURRENT_SCHEMA_VERSION}."
        )
    if version < 3:
        model = migrated.setdefault("press_model", {})
        tuning = migrated.setdefault("auto_tuning", {})

        # Schema 2 inferred wangshub's head_diameter as piece_width * 1.6.
        # That default is much wider than the detected piece and produces a
        # systematic under-press.  Preserve explicitly different user values.
        if model.get("physics_piece_width_multiplier") == 1.6:
            model["physics_piece_width_multiplier"] = 1.15

        # Legacy overlay-derived caps have no directional landing evidence and
        # can reduce a healthy prediction by hundreds of milliseconds.  Keep
        # successful/manual samples, curves and segment corrections, but drop
        # this unsafe learned state.
        model["failure_caps"] = []
        tuning["failure_learning_enabled"] = False
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return migrated


def _load_config_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            user_config = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read valid JSON configuration from '{path}': {exc}") from exc
    if not isinstance(user_config, dict):
        raise ConfigError(f"Configuration root in '{path}' must be a JSON object.")
    config = deep_merge(DEFAULT_CONFIG, _migrate_config(user_config))
    validate_config(config)
    return config


def load_config(path: Path) -> dict[str, Any]:
    backup_path = path.with_name(f"{path.name}.bak")
    if not path.exists() and not backup_path.exists():
        config = copy.deepcopy(DEFAULT_CONFIG)
        validate_config(config)
        return config
    try:
        return _load_config_file(path)
    except ConfigError as primary_error:
        if not backup_path.exists():
            raise
        try:
            recovered = _load_config_file(backup_path)
        except ConfigError as backup_error:
            raise ConfigError(
                f"Primary configuration is invalid ({primary_error}); backup is also invalid "
                f"({backup_error})."
            ) from primary_error
        warnings.warn(
            f"Recovered configuration from backup '{backup_path}' because '{path}' was invalid: "
            f"{primary_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return recovered


def _write_json_atomic(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(config, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def save_config(path: Path, config: dict[str, Any]) -> None:
    config["schema_version"] = CURRENT_SCHEMA_VERSION
    validate_config(config)
    backup_path = path.with_name(f"{path.name}.bak")
    try:
        valid_primary: dict[str, Any] | None = None
        valid_backup: dict[str, Any] | None = None
        if path.exists():
            try:
                valid_primary = _load_config_file(path)
            except ConfigError:
                valid_primary = None
        if backup_path.exists():
            try:
                valid_backup = _load_config_file(backup_path)
            except ConfigError:
                valid_backup = None
        if valid_primary is not None:
            _write_json_atomic(backup_path, valid_primary)
        elif valid_backup is None:
            _write_json_atomic(backup_path, config)
        _write_json_atomic(path, config)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Could not save configuration safely to '{path}': {exc}") from exc


def press_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.setdefault("press_model", {})
    defaults = DEFAULT_CONFIG["press_model"]
    for key, value in defaults.items():
        if key not in model:
            model[key] = copy.deepcopy(value)
    return model


def auto_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    tuning = config.setdefault("auto_tuning", {})
    defaults = DEFAULT_CONFIG["auto_tuning"]
    for key, value in defaults.items():
        if key not in tuning:
            tuning[key] = copy.deepcopy(value)
    return tuning


def neural_press_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.setdefault("neural_press_model", {})
    defaults = DEFAULT_CONFIG["neural_press_model"]
    for key, value in defaults.items():
        if key not in model:
            model[key] = copy.deepcopy(value)
    return model
