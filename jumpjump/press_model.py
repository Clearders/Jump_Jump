from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import auto_tuning_config, press_model_config
from .types import DetectionResult, JumpAutoError
from .utils import clamp, timestamp


@dataclass(frozen=True)
class LandingMeasurement:
    landing_error_px: float
    signed_error_px: float
    projection_ratio: float
    reference_point: tuple[int, int]
    label_confidence: float


def measure_landing(
    previous: DetectionResult,
    current_result: DetectionResult,
    config: dict[str, Any],
) -> LandingMeasurement | None:
    if (
        current_result.landing_platform is None
        or current_result.landing_platform_confidence is None
    ):
        return None

    # The game camera scrolls vertically. Horizontal screen coordinates remain
    # stable, while the current platform supplies the post-scroll Y reference.
    reference = (previous.target[0], current_result.landing_platform[1])
    error_x = float(current_result.piece[0] - reference[0])
    error_y = float(current_result.piece[1] - reference[1])
    landing_error = math.hypot(error_x, error_y)

    model = press_model_config(config)
    x_weight = float(model.get("x_weight", 1.0))
    y_weight = float(model.get("y_weight", 1.0))
    direction_x = previous.dx_px * x_weight
    direction_y = previous.dy_px * y_weight
    direction_distance = math.hypot(direction_x, direction_y)
    if direction_distance <= 1.0:
        return None

    error_effective_x = error_x * x_weight
    error_effective_y = error_y * y_weight
    error_effective_distance = math.hypot(error_effective_x, error_effective_y)
    if error_effective_distance <= 1e-9:
        signed_error = 0.0
        projection_ratio = 1.0
    else:
        signed_error = (
            error_effective_x * direction_x + error_effective_y * direction_y
        ) / direction_distance
        projection_ratio = clamp(abs(signed_error) / error_effective_distance, 0.0, 1.0)
    return LandingMeasurement(
        landing_error_px=landing_error,
        signed_error_px=signed_error,
        projection_ratio=projection_ratio,
        reference_point=reference,
        label_confidence=clamp(float(current_result.landing_platform_confidence), 0.0, 1.0),
    )


def effective_distance_from_delta(dx: float, dy: float, config: dict[str, Any]) -> float:
    model = press_model_config(config)
    x_weight = float(model.get("x_weight", 1.0))
    y_weight = float(model.get("y_weight", 1.0))
    return math.hypot(dx * x_weight, dy * y_weight)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def linear_reference_press_ms(
    distance_px: float,
    width_px: float,
    coefficient: float = 1.390,
) -> float:
    width = _positive_float(width_px)
    if width is None:
        raise JumpAutoError("linear reference width must be positive.")
    return max(0.0, float(distance_px)) * float(coefficient) * 1080.0 / width


def physics_reference_press_ms(
    distance_px: float,
    head_diameter_px: float,
    press_coefficient: float,
) -> float:
    head_diameter = _positive_float(head_diameter_px)
    coefficient = _positive_float(press_coefficient)
    if head_diameter is None:
        raise JumpAutoError("physics head diameter must be positive.")
    if coefficient is None:
        raise JumpAutoError("physics press coefficient must be positive.")
    actual_distance = (
        max(0.0, float(distance_px))
        * (0.945 * 2.0 / head_diameter)
        * (math.sqrt(6.0) / 2.0)
    )
    press_seconds = (
        -945.0 + math.sqrt(945.0**2 + 4.0 * 105.0 * 36.0 * actual_distance)
    ) / (2.0 * 105.0)
    return press_seconds * 1000.0 * coefficient


def physics_head_diameter_px(
    distance_or_result: float | DetectionResult,
    model: dict[str, Any],
) -> float | None:
    configured = _positive_float(model.get("physics_head_diameter_px"))
    if configured is not None:
        return configured

    if isinstance(distance_or_result, DetectionResult):
        try:
            piece_width = float(distance_or_result.piece_bbox[2])
        except (TypeError, ValueError):
            piece_width = 0.0
        multiplier = _positive_float(model.get("physics_piece_width_multiplier", 1.15))
        if piece_width > 0 and multiplier is not None:
            return piece_width * multiplier

    return _positive_float(model.get("physics_default_head_diameter_px", 80.0))


def _distance_px_from_input(distance_or_result: float | DetectionResult) -> float:
    if isinstance(distance_or_result, DetectionResult):
        return float(distance_or_result.effective_distance_px)
    return float(distance_or_result)


def _physics_base_press_ms(
    distance_or_result: float | DetectionResult,
    model: dict[str, Any],
) -> float | None:
    head_diameter = physics_head_diameter_px(distance_or_result, model)
    if head_diameter is None:
        return None
    return physics_reference_press_ms(
        _distance_px_from_input(distance_or_result),
        head_diameter,
        float(model.get("physics_press_coefficient", 1.392)),
    )


def _linear_base_press_ms(distance_px: float, model: dict[str, Any]) -> float | None:
    width = _positive_float(model.get("linear_reference_width_px", 1080))
    if width is None:
        return None
    return linear_reference_press_ms(
        distance_px,
        width,
        float(model.get("linear_reference_coefficient", 1.390)),
    )


def _learned_linear_press_ms(distance_px: float, model: dict[str, Any], config: dict[str, Any]) -> float | None:
    ratio = model.get("slope_ms_per_px") or config.get("press_ms_per_px")
    if ratio is None:
        return None
    return distance_px * float(ratio) + float(model.get("offset_ms", 0.0))


def reference_base_press_ms(
    distance_or_result: float | DetectionResult,
    model: dict[str, Any],
    config: dict[str, Any],
) -> float:
    distance_px = _distance_px_from_input(distance_or_result)
    base_algorithm = str(model.get("base_algorithm", "physics")).lower()
    algorithm_orders = {
        "physics": ("physics", "linear", "learned"),
        "linear": ("linear", "physics", "learned"),
        "learned": ("learned", "physics", "linear"),
    }
    for algorithm in algorithm_orders.get(base_algorithm, algorithm_orders["physics"]):
        if algorithm == "physics":
            press_ms = _physics_base_press_ms(distance_or_result, model)
        elif algorithm == "linear":
            press_ms = _linear_base_press_ms(distance_px, model)
        else:
            press_ms = _learned_linear_press_ms(distance_px, model, config)
        if press_ms is not None:
            return press_ms
    raise JumpAutoError("No press model is configured. Run --calibrate or enable a reference model.")


def learned_curve_adjusted_press_ms(
    base_press_ms: float,
    distance_px: float,
    model: dict[str, Any],
) -> tuple[float, bool]:
    curve_press = piecewise_press_ms(distance_px, model)
    if curve_press is None:
        return base_press_ms, False
    max_ratio = max(0.0, float(model.get("curve_correction_max_ratio", 0.35)))
    max_delta = max(0.0, abs(base_press_ms) * max_ratio)
    correction = clamp(curve_press - base_press_ms, -max_delta, max_delta)
    return base_press_ms + correction, True


def sample_training_press_ms(sample: dict[str, Any]) -> float:
    for field in ("training_press_ms", "center_adjusted_press_ms", "press_ms"):
        try:
            press_ms = float(sample.get(field, 0.0))
        except (TypeError, ValueError):
            continue
        if press_ms > 0:
            return press_ms
    return 0.0


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
            press_ms = sample_training_press_ms(sample)
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
    return reference_base_press_ms(distance_px, model, config)


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
    # Segment identity must not depend on the current sample range.  The old
    # adaptive width changed every segment index as calibration accumulated,
    # so corrections learned for one distance were later applied elsewhere.
    base_size = max(2.0, float(model.get("segment_size_px", 2)))
    segment_index = int(distance_px // base_size)
    distance_min = segment_index * base_size
    distance_max = distance_min + base_size
    segment_center = distance_min + base_size / 2.0
    return segment_index, distance_min, distance_max, segment_center


def _segment_correction_matches_bounds(
    correction: dict[str, Any],
    segment_index: int,
    distance_min: float,
    distance_max: float,
) -> bool:
    try:
        raw_index = correction.get("segment_index")
        if isinstance(raw_index, bool) or float(raw_index).is_integer() is False:
            return False
        return (
            int(raw_index) == segment_index
            and math.isclose(float(correction["distance_min_px"]), distance_min, abs_tol=1e-6)
            and math.isclose(float(correction["distance_max_px"]), distance_max, abs_tol=1e-6)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def segment_correction_ms(distance_px: float, model: dict[str, Any]) -> float:
    segment_index, distance_min, distance_max, _ = segment_bounds_for_distance(distance_px, model)
    for correction in model.get("segment_corrections", []):
        try:
            if _segment_correction_matches_bounds(
                correction,
                segment_index,
                distance_min,
                distance_max,
            ):
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
        if _segment_correction_matches_bounds(
            correction,
            segment_index,
            distance_min,
            distance_max,
        ):
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
    max_corrections = int(model.get("max_segment_corrections", 300))
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
    hits_to_freeze = int(tuning.get("segment_precision_hits_to_freeze", 1))
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
        if _segment_correction_matches_bounds(
            correction,
            segment_index,
            distance_min,
            distance_max,
        ):
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
    max_corrections = int(model.get("max_segment_corrections", 300))
    if len(corrections) > max_corrections:
        del corrections[:-max_corrections]


def decay_segment_center_correction(config: dict[str, Any], distance_px: float) -> None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("segment_correction_enabled", True)):
        return
    model = press_model_config(config)
    segment_index, distance_min, distance_max, _ = segment_bounds_for_distance(distance_px, model)
    decay = clamp(float(tuning.get("segment_correction_success_decay", 0.20)), 0.0, 1.0)
    kept = []
    for correction in model.get("segment_corrections", []):
        if not _segment_correction_matches_bounds(
            correction,
            segment_index,
            distance_min,
            distance_max,
        ):
            kept.append(correction)
            continue
        updated = float(correction.get("correction_ms", 0.0)) * (1.0 - decay)
        if abs(updated) >= 1.0:
            correction["correction_ms"] = updated
            correction["updates"] = int(correction.get("updates", 0)) + 1
            correction["timestamp"] = timestamp()
            kept.append(correction)
    model["segment_corrections"] = kept


def minimum_press_ms_for_distance(
    distance_px: float,
    model: dict[str, Any],
    config: dict[str, Any],
) -> float:
    normal_minimum = float(config["min_press_ms"])
    if not bool(model.get("short_hop_enabled", True)):
        return normal_minimum
    short_hop_limit = max(0.0, float(model.get("short_hop_max_distance_px", 200)))
    if distance_px <= 0 or distance_px >= short_hop_limit:
        return normal_minimum
    short_minimum = float(model.get("short_hop_min_press_ms", normal_minimum))
    return clamp(short_minimum, 1.0, normal_minimum)


def center_adjusted_press_ms(
    previous: DetectionResult,
    current_result: DetectionResult,
    press_ms: float,
    config: dict[str, Any],
    measurement: LandingMeasurement | None = None,
) -> tuple[float, float, float] | None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("center_learning_enabled", True)):
        return None

    measurement = measurement or measure_landing(previous, current_result, config)
    if measurement is None:
        return None
    landing_error = measurement.landing_error_px
    deadzone = float(tuning.get("center_deadzone_px", 14))
    if landing_error <= deadzone:
        return None

    signed_error = measurement.signed_error_px
    projection_ratio = measurement.projection_ratio
    if projection_ratio < float(tuning.get("center_projection_min_ratio", 0.45)):
        return None

    model = press_model_config(config)
    learning_rate = clamp(float(tuning.get("center_learning_rate", 0.65)), 0.05, 1.0)
    max_adjustment = abs(press_ms) * float(tuning.get("center_max_adjustment_ratio", 0.14))
    current_distance = previous.effective_distance_px
    if current_distance > 0:
        b = abs(signed_error) / current_distance
        b_squared = b * b
    else:
        b_squared = 0.0

    correction_ratio = b_squared * learning_rate
    if signed_error > 0:
        factor = 1.0 - correction_ratio
    else:
        factor = 1.0 + correction_ratio

    executable_minimum = minimum_press_ms_for_distance(current_distance, model, config)
    executable_maximum = float(config["max_press_ms"])
    adjustment_minimum = max(executable_minimum, press_ms - max_adjustment)
    adjustment_maximum = min(executable_maximum, press_ms + max_adjustment)
    adjusted_press = clamp(press_ms * factor, adjustment_minimum, adjustment_maximum)
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


def _sample_weight(sample: dict[str, Any], index: int, total: int) -> float:
    confidence = float(sample.get("confidence", 0.75))
    conf_weight = clamp((confidence - 0.40) / 0.55, 0.15, 1.0)
    recency_ratio = (index + 1) / max(1, total)
    recency_weight = 0.55 + 0.45 * recency_ratio
    return conf_weight * recency_weight


def _weighted_fit_line_with_offset(
    distances: list[float],
    durations: list[float],
    weights: list[float],
) -> tuple[float, float, float] | None:
    count = len(distances)
    if count < 2:
        return None
    sw = sum(weights)
    swx = sum(w * x for w, x in zip(weights, distances))
    swy = sum(w * y for w, y in zip(weights, durations))
    swxx = sum(w * x * x for w, x in zip(weights, distances))
    swxy = sum(w * x * y for w, x, y in zip(weights, distances, durations))
    denominator = sw * swxx - swx * swx
    if abs(denominator) < 1e-6:
        return None
    slope = (sw * swxy - swx * swy) / denominator
    offset = (swy - slope * swx) / sw
    if slope <= 0:
        return None
    errors = [
        duration - (slope * distance + offset)
        for distance, duration in zip(distances, durations)
    ]
    weighted_mse = sum(w * e * e for w, e in zip(weights, errors)) / max(1e-6, sw)
    rmse = math.sqrt(weighted_mse)
    return slope, offset, rmse


def _weighted_fit_line_through_origin(
    distances: list[float],
    durations: list[float],
    weights: list[float],
) -> tuple[float, float]:
    swxx = sum(w * d * d for w, d in zip(weights, distances))
    if swxx <= 0:
        raise JumpAutoError("Calibration distances are invalid.")
    swxy = sum(w * d * t for w, d, t in zip(weights, distances, durations))
    slope = swxy / swxx
    sw = sum(weights)
    errors = [duration - slope * distance for distance, duration in zip(distances, durations)]
    weighted_mse = sum(w * e * e for w, e in zip(weights, errors)) / max(1e-6, sw)
    rmse = math.sqrt(weighted_mse)
    return slope, rmse


def _outlier_threshold_from_rmse(rmse: float, sample_count: int) -> float:
    if sample_count <= 3:
        return rmse * 100.0
    return rmse * (2.5 + 0.3 * (min(20.0, float(sample_count)) / 20.0))


def _refit_with_clean_samples(
    samples: list[dict[str, Any]],
    y_weight: float,
    fit_offset: bool,
    slope: float,
    offset: float,
    outlier_threshold: float,
) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for sample in samples:
        distance = math.hypot(float(sample["dx_px"]), float(sample["dy_px"]) * y_weight)
        predicted = slope * distance + offset
        residual = abs(sample_training_press_ms(sample) - predicted)
        if residual <= outlier_threshold:
            clean.append(sample)
    return clean


def fit_press_model(config: dict[str, Any]) -> dict[str, Any]:
    model = press_model_config(config)
    samples = [
        sample
        for sample in model.get("samples", [])
        if sample_training_press_ms(sample) > 0
    ]
    if not samples:
        return model

    max_samples = int(model.get("max_samples", 40))
    if len(samples) > max_samples:
        samples = samples[-max_samples:]
        model["samples"] = samples

    current_y_weight = float(model.get("y_weight", 1.0))
    min_samples_for_weight_fit = int(model.get("min_samples_for_weight_fit", 3))
    fit_offset = len(samples) >= 4
    total = len(samples)
    weights = [_sample_weight(sample, index, total) for index, sample in enumerate(samples)]

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
        durations = [sample_training_press_ms(sample) for sample in samples]
        if fit_offset:
            fitted = _weighted_fit_line_with_offset(distances, durations, weights)
            if fitted is None:
                continue
            slope, offset, rmse = fitted
            if offset < -250 or offset > 350:
                continue
        else:
            slope, rmse = _weighted_fit_line_through_origin(distances, durations, weights)
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
        durations = [sample_training_press_ms(sample) for sample in samples]
        slope, rmse = _weighted_fit_line_through_origin(distances, durations, weights)
        best = (current_y_weight, slope, 0.0, rmse)

    y_weight, slope, offset, rmse = best

    if len(samples) >= 5:
        outlier_threshold = _outlier_threshold_from_rmse(rmse, len(samples))
        clean_samples = _refit_with_clean_samples(samples, y_weight, fit_offset, slope, offset, outlier_threshold)
        min_clean = max(3, int(len(samples) * 0.7))
        if len(clean_samples) >= min_clean:
            clean_total = len(clean_samples)
            clean_weights = [_sample_weight(s, idx, clean_total) for idx, s in enumerate(clean_samples)]
            clean_distances = [
                math.hypot(float(s["dx_px"]), float(s["dy_px"]) * y_weight)
                for s in clean_samples
            ]
            clean_durations = [sample_training_press_ms(s) for s in clean_samples]
            if fit_offset:
                refitted = _weighted_fit_line_with_offset(clean_distances, clean_durations, clean_weights)
                if refitted is not None and refitted[1] >= -250 and refitted[1] <= 350:
                    slope, offset, rmse = refitted
            else:
                slope, rmse = _weighted_fit_line_through_origin(clean_distances, clean_durations, clean_weights)
            model["outlier_ratio"] = 1.0 - len(clean_samples) / len(samples)

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
        if sample_training_press_ms(sample) > 0
    ]
    if not samples:
        return None

    y_weight = float(model.get("y_weight", 1.0))
    min_anchor_distance = float(model.get("short_hop_min_anchor_distance_px", 80))
    short_samples: list[tuple[float, float]] = []
    for sample in samples:
        try:
            sample_distance = sample_effective_distance(sample, y_weight)
            sample_press = sample_training_press_ms(sample)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < sample_distance < min_anchor_distance and sample_press > 0:
            short_samples.append((sample_distance, sample_press))

    if short_samples:
        short_samples.sort(key=lambda item: item[0])
        if short_samples[0][0] <= distance_px <= short_samples[-1][0]:
            first_dist, first_press = short_samples[0]
            if distance_px <= first_dist:
                return distance_px * (first_press / first_dist)
            prev_dist, prev_press = short_samples[0]
            for next_dist, next_press in short_samples[1:]:
                if distance_px <= next_dist:
                    span = max(1.0, next_dist - prev_dist)
                    ratio = (distance_px - prev_dist) / span
                    return prev_press + (next_press - prev_press) * ratio
                prev_dist, prev_press = next_dist, next_press
        if distance_px < short_samples[0][0]:
            return distance_px * (short_samples[0][1] / short_samples[0][0])
        if distance_px > short_samples[-1][0]:
            return distance_px * (short_samples[-1][1] / short_samples[-1][0])

    anchors: list[tuple[float, float]] = []
    for sample in samples:
        try:
            sample_distance = sample_effective_distance(sample, y_weight)
            sample_press = sample_training_press_ms(sample)
        except (KeyError, TypeError, ValueError):
            continue
        if sample_distance >= min_anchor_distance and sample_press > 0:
            anchors.append((sample_distance, sample_press))

    if not anchors:
        return None

    anchor_distance, anchor_press = min(anchors, key=lambda item: item[0])
    if distance_px >= anchor_distance:
        return None

    press = distance_px * (anchor_press / anchor_distance)
    if distance_px < 60:
        press *= 1.0 + 0.04 * (1.0 - distance_px / 60.0)
    return press


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
        cap_confidence = float(cap.get("confidence", 0.70))
        confidence_factor = clamp((cap_confidence - 0.40) / 0.55, 0.4, 1.6)
        penalty = 1.0 + 0.12 * (distance_delta / max(1.0, window_px)) * confidence_factor
        scaled_cap *= penalty
        best_cap = scaled_cap if best_cap is None else min(best_cap, scaled_cap)
    return best_cap


def calculate_press_ms(distance_or_result: float | DetectionResult, config: dict[str, Any]) -> float:
    distance_px = _distance_px_from_input(distance_or_result)

    model = press_model_config(config)
    if (
        bool(model.get("curve_enabled", True))
        and len(model.get("curve_points", [])) < int(model.get("curve_min_samples", 3))
        and len(model.get("samples", [])) >= int(model.get("curve_min_samples", 3))
    ):
        fit_press_model(config)
        model = press_model_config(config)
    press_ms = reference_base_press_ms(distance_or_result, model, config)
    press_ms, _ = learned_curve_adjusted_press_ms(press_ms, distance_px, model)
    short_cap = short_hop_press_cap_ms(distance_px, model)
    if short_cap is not None:
        press_ms = min(press_ms, short_cap)
    press_ms += segment_correction_ms(distance_px, model)
    failure_cap = failure_press_cap_ms(distance_px, model, config)
    if failure_cap is not None:
        press_ms = min(press_ms, failure_cap)
    minimum_press = minimum_press_ms_for_distance(distance_px, model, config)
    return clamp(press_ms, minimum_press, float(config["max_press_ms"]))


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
