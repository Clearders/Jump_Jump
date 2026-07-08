from __future__ import annotations

import math
from typing import Any

from .config import auto_tuning_config, press_model_config
from .types import DetectionResult, JumpAutoError
from .utils import clamp, timestamp


def effective_distance_from_delta(dx: float, dy: float, config: dict[str, Any]) -> float:
    model = press_model_config(config)
    x_weight = float(model.get("x_weight", 1.0))
    y_weight = float(model.get("y_weight", 1.0))
    return math.hypot(dx * x_weight, dy * y_weight)


def sample_effective_distance(sample: dict[str, Any], y_weight: float) -> float:
    return math.hypot(float(sample["dx_px"]), float(sample["dy_px"]) * y_weight)


def build_press_curve_points(
    samples: list[dict[str, Any]],
    y_weight: float,
) -> list[dict[str, float]]:
    points: list[tuple[float, float]] = []
    for sample in samples:
        try:
            distance = sample_effective_distance(sample, y_weight)
            press_ms = float(sample["press_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if distance > 0 and press_ms > 0:
            points.append((distance, press_ms))
    if not points:
        return []

    points.sort(key=lambda item: item[0])
    grouped: list[tuple[float, float]] = []
    group_distances: list[float] = []
    group_presses: list[float] = []
    group_window_px = 18.0
    for distance, press_ms in points:
        if group_distances and distance - group_distances[0] > group_window_px:
            grouped.append(
                (
                    sum(group_distances) / len(group_distances),
                    sum(group_presses) / len(group_presses),
                )
            )
            group_distances = []
            group_presses = []
        group_distances.append(distance)
        group_presses.append(press_ms)
    if group_distances:
        grouped.append(
            (
                sum(group_distances) / len(group_distances),
                sum(group_presses) / len(group_presses),
            )
        )

    monotone: list[dict[str, float]] = []
    max_press = 0.0
    for distance, press_ms in grouped:
        max_press = max(max_press, press_ms)
        monotone.append({"distance_px": distance, "press_ms": max_press})
    return monotone


def piecewise_press_ms(distance_px: float, model: dict[str, Any]) -> float | None:
    if not bool(model.get("curve_enabled", True)):
        return None
    points = [
        (float(point["distance_px"]), float(point["press_ms"]))
        for point in model.get("curve_points", [])
        if float(point.get("distance_px", 0)) > 0 and float(point.get("press_ms", 0)) > 0
    ]
    min_points = int(model.get("curve_min_samples", 3))
    if len(points) < min_points:
        return None
    points.sort(key=lambda item: item[0])

    first_distance, first_press = points[0]
    if distance_px <= first_distance:
        return distance_px * (first_press / first_distance)

    previous_distance, previous_press = points[0]
    for next_distance, next_press in points[1:]:
        if distance_px <= next_distance:
            span = max(1.0, next_distance - previous_distance)
            ratio = (distance_px - previous_distance) / span
            return previous_press + (next_press - previous_press) * ratio
        previous_distance, previous_press = next_distance, next_press

    if len(points) >= 2:
        before_distance, before_press = points[-2]
        last_distance, last_press = points[-1]
        tail_slope = (last_press - before_press) / max(1.0, last_distance - before_distance)
        linear_slope = float(model.get("slope_ms_per_px") or 0.0)
        if linear_slope > 0:
            tail_slope = clamp(tail_slope, linear_slope * 0.45, linear_slope * 1.65)
        tail_slope = max(0.05, tail_slope)
        return last_press + (distance_px - last_distance) * tail_slope

    return None


def base_press_ms_for_distance(distance_px: float, model: dict[str, Any], config: dict[str, Any]) -> float:
    curve_press = piecewise_press_ms(distance_px, model)
    if curve_press is not None:
        return curve_press
    ratio = model.get("slope_ms_per_px") or config.get("press_ms_per_px")
    if ratio is None:
        raise JumpAutoError("press_ms_per_px is not configured. Run --calibrate first.")
    return distance_px * float(ratio) + float(model.get("offset_ms", 0.0))


def local_press_slope_ms_per_px(distance_px: float, model: dict[str, Any]) -> float:
    points = [
        (float(point["distance_px"]), float(point["press_ms"]))
        for point in model.get("curve_points", [])
        if float(point.get("distance_px", 0)) > 0 and float(point.get("press_ms", 0)) > 0
    ]
    points.sort(key=lambda item: item[0])
    if len(points) >= 2:
        previous_distance, previous_press = points[0]
        for next_distance, next_press in points[1:]:
            if distance_px <= next_distance:
                return max(
                    0.05,
                    (next_press - previous_press) / max(1.0, next_distance - previous_distance),
                )
            previous_distance, previous_press = next_distance, next_press
        before_distance, before_press = points[-2]
        last_distance, last_press = points[-1]
        return max(0.05, (last_press - before_press) / max(1.0, last_distance - before_distance))
    return max(0.05, float(model.get("slope_ms_per_px") or 1.0))


def segment_bounds_for_distance(
    distance_px: float,
    model: dict[str, Any],
) -> tuple[int, float, float, float]:
    segment_size = max(4.0, float(model.get("segment_size_px", 7)))
    segment_index = int(distance_px // segment_size)
    distance_min = segment_index * segment_size
    distance_max = distance_min + segment_size
    segment_center = distance_min + segment_size / 2.0
    return segment_index, distance_min, distance_max, segment_center


def segment_correction_ms(distance_px: float, model: dict[str, Any]) -> float:
    segment_index, _, _, _ = segment_bounds_for_distance(distance_px, model)
    for correction in model.get("segment_corrections", []):
        try:
            if int(correction.get("segment_index", -1)) == segment_index:
                return float(correction.get("correction_ms", 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def segment_correction_entry(
    model: dict[str, Any],
    distance_px: float,
    create: bool = False,
) -> dict[str, Any] | None:
    segment_index, distance_min, distance_max, segment_center = segment_bounds_for_distance(
        distance_px,
        model,
    )
    corrections = model.setdefault("segment_corrections", [])
    for correction in corrections:
        if int(correction.get("segment_index", -1)) == segment_index:
            return correction
    if not create:
        return None
    correction = {
        "segment_index": segment_index,
        "distance_min_px": distance_min,
        "distance_max_px": distance_max,
        "segment_center_px": segment_center,
        "correction_ms": 0.0,
        "updates": 0,
        "stable_hits": 0,
        "frozen": False,
        "timestamp": timestamp(),
    }
    corrections.append(correction)
    max_corrections = int(model.get("max_segment_corrections", 120))
    if len(corrections) > max_corrections:
        del corrections[:-max_corrections]
    return correction


def segment_is_frozen(config: dict[str, Any], distance_px: float) -> bool:
    model = press_model_config(config)
    correction = segment_correction_entry(model, distance_px, create=False)
    return bool(correction and correction.get("frozen", False))


def mark_segment_precision_hit(config: dict[str, Any], distance_px: float, landing_error: float) -> bool:
    tuning = auto_tuning_config(config)
    model = press_model_config(config)
    correction = segment_correction_entry(model, distance_px, create=True)
    if correction is None:
        return False
    correction["stable_hits"] = int(correction.get("stable_hits", 0)) + 1
    correction["last_landing_error_px"] = landing_error
    correction["timestamp"] = timestamp()
    hits_to_freeze = int(tuning.get("segment_precision_hits_to_freeze", 3))
    if correction["stable_hits"] >= hits_to_freeze:
        correction["frozen"] = True
    return bool(correction.get("frozen", False))


def maybe_unfreeze_segment_for_error(
    config: dict[str, Any],
    distance_px: float,
    landing_error: float,
) -> bool:
    tuning = auto_tuning_config(config)
    unfreeze_error = float(tuning.get("segment_unfreeze_error_px", 18))
    if landing_error < unfreeze_error:
        return False
    model = press_model_config(config)
    correction = segment_correction_entry(model, distance_px, create=False)
    if correction is None or not bool(correction.get("frozen", False)):
        return False
    correction["frozen"] = False
    correction["stable_hits"] = 0
    correction["unfrozen_at_error_px"] = landing_error
    correction["timestamp"] = timestamp()
    return True


def record_segment_center_correction(
    config: dict[str, Any],
    distance_px: float,
    correction_delta_ms: float,
    signed_error_px: float,
    projection_ratio: float,
) -> None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("segment_correction_enabled", True)):
        return

    model = press_model_config(config)
    existing = segment_correction_entry(model, distance_px, create=False)
    if existing is not None and bool(existing.get("frozen", False)):
        return
    segment_index, distance_min, distance_max, segment_center = segment_bounds_for_distance(
        distance_px,
        model,
    )
    segment_size = max(4.0, float(model.get("segment_size_px", 7)))
    max_ratio = float(tuning.get("segment_max_correction_ratio", 0.18))
    base_press = piecewise_press_ms(segment_center, model)
    if base_press is None:
        slope = max(0.05, float(model.get("slope_ms_per_px") or 1.0))
        base_press = segment_center * slope + float(model.get("offset_ms", 0.0))
    max_abs_correction = max(8.0, abs(base_press) * max_ratio)
    correction_delta_ms = clamp(correction_delta_ms, -max_abs_correction, max_abs_correction)

    corrections = model.setdefault("segment_corrections", [])
    learning_rate = clamp(float(tuning.get("segment_correction_learning_rate", 0.55)), 0.05, 1.0)
    for correction in corrections:
        if int(correction.get("segment_index", -1)) == segment_index:
            previous = float(correction.get("correction_ms", 0.0))
            updated = previous * (1.0 - learning_rate) + correction_delta_ms * learning_rate
            correction["correction_ms"] = clamp(updated, -max_abs_correction, max_abs_correction)
            correction["updates"] = int(correction.get("updates", 0)) + 1
            correction["stable_hits"] = 0
            correction["frozen"] = False
            correction["last_signed_error_px"] = signed_error_px
            correction["last_projection_ratio"] = projection_ratio
            correction["timestamp"] = timestamp()
            return

    corrections.append(
        {
            "segment_index": segment_index,
            "distance_min_px": distance_min,
            "distance_max_px": distance_max,
            "segment_center_px": segment_center,
            "correction_ms": correction_delta_ms,
            "updates": 1,
            "stable_hits": 0,
            "frozen": False,
            "last_signed_error_px": signed_error_px,
            "last_projection_ratio": projection_ratio,
            "timestamp": timestamp(),
        }
    )
    max_corrections = int(model.get("max_segment_corrections", 120))
    if len(corrections) > max_corrections:
        del corrections[:-max_corrections]


def decay_segment_center_correction(config: dict[str, Any], distance_px: float) -> None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("segment_correction_enabled", True)):
        return
    model = press_model_config(config)
    segment_index, _, _, _ = segment_bounds_for_distance(distance_px, model)
    decay = clamp(float(tuning.get("segment_correction_success_decay", 0.20)), 0.0, 1.0)
    kept = []
    for correction in model.get("segment_corrections", []):
        if int(correction.get("segment_index", -1)) != segment_index:
            kept.append(correction)
            continue
        updated = float(correction.get("correction_ms", 0.0)) * (1.0 - decay)
        if abs(updated) >= 1.0:
            correction["correction_ms"] = updated
            correction["updates"] = int(correction.get("updates", 0)) + 1
            correction["timestamp"] = timestamp()
            kept.append(correction)
    model["segment_corrections"] = kept


def center_adjusted_press_ms(
    previous: DetectionResult,
    current_result: DetectionResult,
    press_ms: float,
    config: dict[str, Any],
) -> tuple[float, float, float] | None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("center_learning_enabled", True)):
        return None

    error_x = float(current_result.piece[0] - previous.target[0])
    error_y = float(current_result.piece[1] - previous.target[1])
    landing_error = math.hypot(error_x, error_y)
    deadzone = float(tuning.get("center_deadzone_px", 14))
    if landing_error <= deadzone:
        return None

    model = press_model_config(config)
    y_weight = float(model.get("y_weight", 1.0))
    direction_x = previous.dx_px
    direction_y = previous.dy_px * y_weight
    direction_distance = math.hypot(direction_x, direction_y)
    if direction_distance <= 1.0:
        return None

    error_effective_x = error_x
    error_effective_y = error_y * y_weight
    error_effective_distance = math.hypot(error_effective_x, error_effective_y)
    signed_error = (
        error_effective_x * direction_x + error_effective_y * direction_y
    ) / direction_distance
    projection_ratio = abs(signed_error) / max(1.0, error_effective_distance)
    if projection_ratio < float(tuning.get("center_projection_min_ratio", 0.45)):
        return None

    learning_rate = clamp(float(tuning.get("center_learning_rate", 0.65)), 0.05, 1.0)
    max_adjustment = abs(press_ms) * float(tuning.get("center_max_adjustment_ratio", 0.14))
    current_distance = previous.effective_distance_px
    desired_distance = max(1.0, current_distance - signed_error)
    current_curve_press = base_press_ms_for_distance(current_distance, model, config)
    desired_curve_press = base_press_ms_for_distance(desired_distance, model, config)
    curve_delta = desired_curve_press - current_curve_press
    adjustment = clamp(curve_delta * learning_rate, -max_adjustment, max_adjustment)
    adjusted_press = press_ms + adjustment
    return adjusted_press, signed_error, projection_ratio


def fit_line_through_origin(distances: list[float], durations: list[float]) -> tuple[float, float]:
    denominator = sum(distance * distance for distance in distances)
    if denominator <= 0:
        raise JumpAutoError("Calibration distances are invalid.")
    slope = sum(distance * duration for distance, duration in zip(distances, durations)) / denominator
    errors = [duration - slope * distance for distance, duration in zip(distances, durations)]
    rmse = math.sqrt(sum(error * error for error in errors) / max(1, len(errors)))
    return slope, rmse


def fit_line_with_offset(distances: list[float], durations: list[float]) -> tuple[float, float, float] | None:
    count = len(distances)
    if count < 2:
        return None
    sx = sum(distances)
    sy = sum(durations)
    sxx = sum(distance * distance for distance in distances)
    sxy = sum(distance * duration for distance, duration in zip(distances, durations))
    denominator = count * sxx - sx * sx
    if abs(denominator) < 1e-6:
        return None
    slope = (count * sxy - sx * sy) / denominator
    offset = (sy - slope * sx) / count
    if slope <= 0:
        return None
    errors = [
        duration - (slope * distance + offset)
        for distance, duration in zip(distances, durations)
    ]
    rmse = math.sqrt(sum(error * error for error in errors) / max(1, count))
    return slope, offset, rmse


def calibration_weight_candidates(samples: list[dict[str, Any]], current_y_weight: float) -> list[float]:
    min_ratio = 1.0
    max_ratio = 0.0
    for sample in samples:
        dx = abs(float(sample["dx_px"]))
        dy = abs(float(sample["dy_px"]))
        ratio = dy / max(1.0, dx + dy)
        min_ratio = min(min_ratio, ratio)
        max_ratio = max(max_ratio, ratio)
    if max_ratio - min_ratio < 0.16:
        return [current_y_weight]
    candidates = [round(0.55 + index * 0.025, 3) for index in range(67)]
    candidates.append(round(current_y_weight, 3))
    return sorted(set(candidates))


def fit_press_model(config: dict[str, Any]) -> dict[str, Any]:
    model = press_model_config(config)
    samples = [
        sample
        for sample in model.get("samples", [])
        if float(sample.get("press_ms", 0)) > 0
    ]
    if not samples:
        return model

    max_samples = int(model.get("max_samples", 40))
    if len(samples) > max_samples:
        samples = samples[-max_samples:]
        model["samples"] = samples

    current_y_weight = float(model.get("y_weight", 1.0))
    min_samples_for_weight_fit = int(model.get("min_samples_for_weight_fit", 3))
    durations = [float(sample["press_ms"]) for sample in samples]
    fit_offset = len(samples) >= 4

    best: tuple[float, float, float, float] | None = None
    candidates = (
        calibration_weight_candidates(samples, current_y_weight)
        if len(samples) >= min_samples_for_weight_fit
        else [current_y_weight]
    )
    for y_weight in candidates:
        distances = [
            math.hypot(float(sample["dx_px"]), float(sample["dy_px"]) * y_weight)
            for sample in samples
        ]
        if fit_offset:
            fitted = fit_line_with_offset(distances, durations)
            if fitted is None:
                continue
            slope, offset, rmse = fitted
            if offset < -250 or offset > 350:
                continue
        else:
            slope, rmse = fit_line_through_origin(distances, durations)
            offset = 0.0
            if slope <= 0:
                continue
        if best is None or rmse < best[3]:
            best = (y_weight, slope, offset, rmse)

    if best is None:
        distances = [
            math.hypot(float(sample["dx_px"]), float(sample["dy_px"]) * current_y_weight)
            for sample in samples
        ]
        slope, rmse = fit_line_through_origin(distances, durations)
        best = (current_y_weight, slope, 0.0, rmse)

    y_weight, slope, offset, rmse = best
    model["type"] = "weighted_euclidean"
    model["x_weight"] = 1.0
    model["y_weight"] = y_weight
    model["slope_ms_per_px"] = slope
    model["offset_ms"] = offset
    model["fit_rmse_ms"] = rmse
    model["sample_count"] = len(samples)
    model["curve_points"] = build_press_curve_points(samples, y_weight)
    model["type"] = (
        "weighted_piecewise"
        if len(model["curve_points"]) >= int(model.get("curve_min_samples", 3))
        else "weighted_euclidean"
    )
    config["press_ms_per_px"] = slope
    return model


def short_hop_press_cap_ms(distance_px: float, model: dict[str, Any]) -> float | None:
    if not bool(model.get("short_hop_enabled", True)):
        return None

    samples = [
        sample
        for sample in model.get("samples", [])
        if float(sample.get("press_ms", 0)) > 0
    ]
    if not samples:
        return None

    y_weight = float(model.get("y_weight", 1.0))
    min_anchor_distance = float(model.get("short_hop_min_anchor_distance_px", 80))
    anchors: list[tuple[float, float]] = []
    for sample in samples:
        try:
            sample_distance = sample_effective_distance(sample, y_weight)
            sample_press = float(sample["press_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if sample_distance >= min_anchor_distance and sample_press > 0:
            anchors.append((sample_distance, sample_press))

    if not anchors:
        return None

    anchor_distance, anchor_press = min(anchors, key=lambda item: item[0])
    if distance_px >= anchor_distance:
        return None

    return distance_px * (anchor_press / anchor_distance)


def failure_press_cap_ms(distance_px: float, model: dict[str, Any], config: dict[str, Any]) -> float | None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("failure_learning_enabled", True)):
        return None
    caps = [
        cap
        for cap in model.get("failure_caps", [])
        if float(cap.get("distance_px", 0)) > 0 and float(cap.get("press_cap_ms", 0)) > 0
    ]
    if not caps:
        return None

    window_ratio = float(tuning.get("failure_cap_window_ratio", 0.16))
    min_window_px = float(tuning.get("failure_cap_min_window_px", 42))
    best_cap: float | None = None
    for cap in caps:
        cap_distance = float(cap["distance_px"])
        window_px = max(min_window_px, cap_distance * window_ratio)
        distance_delta = abs(distance_px - cap_distance)
        if distance_delta > window_px:
            continue
        scaled_cap = float(cap["press_cap_ms"]) * distance_px / cap_distance
        penalty = 1.0 + 0.12 * (distance_delta / max(1.0, window_px))
        scaled_cap *= penalty
        best_cap = scaled_cap if best_cap is None else min(best_cap, scaled_cap)
    return best_cap


def calculate_press_ms(distance_or_result: float | DetectionResult, config: dict[str, Any]) -> float:
    if isinstance(distance_or_result, DetectionResult):
        distance_px = distance_or_result.effective_distance_px
    else:
        distance_px = float(distance_or_result)

    model = press_model_config(config)
    ratio = model.get("slope_ms_per_px") or config.get("press_ms_per_px")
    if ratio is None:
        raise JumpAutoError("press_ms_per_px is not configured. Run --calibrate first.")
    if (
        bool(model.get("curve_enabled", True))
        and len(model.get("curve_points", [])) < int(model.get("curve_min_samples", 3))
        and len(model.get("samples", [])) >= int(model.get("curve_min_samples", 3))
    ):
        fit_press_model(config)
        model = press_model_config(config)
    press_ms = base_press_ms_for_distance(distance_px, model, config)
    if piecewise_press_ms(distance_px, model) is None:
        short_cap = short_hop_press_cap_ms(distance_px, model)
        if short_cap is not None:
            press_ms = min(press_ms, short_cap)
    press_ms += segment_correction_ms(distance_px, model)
    failure_cap = failure_press_cap_ms(distance_px, model, config)
    if failure_cap is not None:
        press_ms = min(press_ms, failure_cap)
    return clamp(press_ms, float(config["min_press_ms"]), float(config["max_press_ms"]))


def calibration_sample_from_result(result: DetectionResult, duration_ms: float) -> dict[str, Any]:
    return {
        "timestamp": timestamp(),
        "piece": [result.piece[0], result.piece[1]],
        "target": [result.target[0], result.target[1]],
        "distance_px": result.effective_distance_px,
        "dx_px": result.dx_px,
        "dy_px": result.dy_px,
        "screen_distance_px": result.screen_distance_px,
        "effective_distance_px": result.effective_distance_px,
        "press_ms": duration_ms,
        "landing_error_px": None,
        "confidence": result.confidence,
        "result_type": "manual",
    }


def clear_failure_caps_near_success(config: dict[str, Any], distance_px: float) -> None:
    model = press_model_config(config)
    caps = model.get("failure_caps", [])
    if not caps:
        return
    tuning = auto_tuning_config(config)
    window_ratio = float(tuning.get("failure_cap_window_ratio", 0.16))
    min_window_px = float(tuning.get("failure_cap_min_window_px", 42))
    kept = []
    for cap in caps:
        try:
            cap_distance = float(cap["distance_px"])
        except (KeyError, TypeError, ValueError):
            continue
        window_px = max(min_window_px, cap_distance * window_ratio)
        if abs(distance_px - cap_distance) > window_px:
            kept.append(cap)
    model["failure_caps"] = kept
