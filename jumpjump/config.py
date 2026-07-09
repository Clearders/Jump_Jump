from __future__ import annotations

import json
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_DIR / "jump_config.json"
DEFAULT_DEBUG_DIR = APP_DIR / "debug"


DEFAULT_CONFIG: dict[str, Any] = {
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
        "min_width": 18,
        "min_height": 10,
        "min_distance_ratio": 0.10,
        "current_platform_exclude_pad_ratio": 0.25,
        "current_platform_edge_gap_ratio": 0.18,
        "current_platform_max_distance_ratio": 0.26,
        "current_platform_color_tolerance_lab": 28,
        "current_platform_color_max_distance_ratio": 0.14,
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
        "top_surface_center_y_ratio": 0.50,
    },
    "overlay": {
        "dark_gray_threshold": 88,
        "min_dark_area_ratio": 0.055,
        "min_dark_width_ratio": 0.55,
        "min_dark_height_ratio": 0.12,
    },
    "auto_tuning": {
        "enabled": True,
        "landing_tolerance_px": 80,
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
        "min_confidence": 0.60,
        "save_every_success": True,
        "failure_learning_enabled": True,
        "failure_shrink_ratio": 0.92,
        "failure_cap_window_ratio": 0.16,
        "failure_cap_min_window_px": 42,
    },
}



def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as file:
        user_config = json.load(file)
    return deep_merge(DEFAULT_CONFIG, user_config)


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def press_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.setdefault("press_model", {})
    defaults = DEFAULT_CONFIG["press_model"]
    for key, value in defaults.items():
        if key not in model:
            model[key] = json.loads(json.dumps(value))
    return model


def auto_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    tuning = config.setdefault("auto_tuning", {})
    defaults = DEFAULT_CONFIG["auto_tuning"]
    for key, value in defaults.items():
        if key not in tuning:
            tuning[key] = json.loads(json.dumps(value))
    return tuning