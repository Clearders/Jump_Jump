from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .config import (
    CURRENT_BASE_CURVE_FIT_VERSION,
    auto_tuning_config,
    base_curve_sample_is_eligible,
    press_model_config,
)
from .types import DetectionResult, JumpAutoError
from .utils import clamp, timestamp


@dataclass(frozen=True)
class LandingMeasurement:
    landing_error_px: float
    signed_error_px: float
    signed_screen_error_px: float
    projection_ratio: float
    reference_point: tuple[int, int]
    label_confidence: float
    label_source: str = "visible_platform"


def normalized_piece_dimensions(result: DetectionResult) -> tuple[float, float] | None:
    crop_width = result.crop_rect[2] - result.crop_rect[0]
    crop_height = result.crop_rect[3] - result.crop_rect[1]
    piece_width = result.piece_bbox[2]
    piece_height = result.piece_bbox[3]
    if crop_width <= 0 or crop_height <= 0 or piece_width <= 0 or piece_height <= 0:
        return None
    return piece_width / crop_width, piece_height / crop_height


def _temporal_piece_scale_is_consistent(
    previous: DetectionResult,
    current_result: DetectionResult,
    config: dict[str, Any],
) -> bool:
    previous_size = normalized_piece_dimensions(previous)
    current_size = normalized_piece_dimensions(current_result)
    if previous_size is None or current_size is None:
        return False
    width_scale = current_size[0] / previous_size[0]
    height_scale = current_size[1] / previous_size[1]
    tuning = auto_tuning_config(config)
    minimum = float(tuning.get("temporal_piece_scale_min_ratio", 0.90))
    maximum = float(tuning.get("temporal_piece_scale_max_ratio", 1.10))
    if not (minimum <= width_scale <= maximum and minimum <= height_scale <= maximum):
        return False
    aspect_tolerance = float(
        tuning.get("temporal_piece_aspect_tolerance_ratio", 1.10)
    )
    return max(width_scale, height_scale) / max(
        1e-6,
        min(width_scale, height_scale),
    ) <= aspect_tolerance


def _valid_single_jump_score_delta(delta: int, maximum: int) -> bool:
    """A scored jump adds either one point or a positive even-valued bonus."""
    return 1 <= delta <= maximum and (delta == 1 or delta % 2 == 0)


def measure_landing(
    previous: DetectionResult,
    current_result: DetectionResult,
    config: dict[str, Any],
    *,
    allow_temporal_fallback: bool = False,
) -> LandingMeasurement | None:
    crop_left, _, crop_right, _ = current_result.crop_rect
    previous_target_left = previous.crop_rect[0] + previous.target_bbox[0]
    previous_target_width = max(1.0, float(previous.target_bbox[2]))
    previous_target_right = previous_target_left + previous_target_width
    piece_width = max(1.0, float(current_result.piece_bbox[2]))
    crop_width = max(1, crop_right - crop_left)
    visible_fields = (
        current_result.landing_platform,
        current_result.landing_platform_bbox,
        current_result.landing_platform_confidence,
    )
    has_visible_platform = all(value is not None for value in visible_fields)
    if any(value is not None for value in visible_fields) and not has_visible_platform:
        return None

    if has_visible_platform:
        landing_bbox = current_result.landing_platform_bbox
        assert landing_bbox is not None
        landing_x, _, landing_width, _ = landing_bbox
        landing_left = crop_left + landing_x
        landing_right = landing_left + landing_width
        association_margin = max(8.0, piece_width, landing_width * 0.15)
        if not (
            landing_left - association_margin
            <= previous.target[0]
            <= landing_right + association_margin
        ):
            return None
        if landing_width >= crop_width * 0.90:
            return None
        target_cfg = config["target"]
        landing_width_value = max(1.0, float(landing_width))
        width_similarity = min(landing_width_value, previous_target_width) / max(
            landing_width_value,
            previous_target_width,
        )
        if width_similarity < float(target_cfg.get("landing_min_width_similarity", 0.55)):
            return None
        overlap = max(
            0.0,
            min(float(landing_right), previous_target_right)
            - max(float(landing_left), float(previous_target_left)),
        )
        if (
            overlap / min(landing_width_value, previous_target_width)
            < float(target_cfg.get("landing_min_horizontal_overlap", 0.35))
        ):
            return None
        max_center_drift = max(
            piece_width
            * float(target_cfg.get("landing_max_center_drift_piece_ratio", 1.00)),
            previous_target_width
            * float(target_cfg.get("landing_max_center_drift_width_ratio", 0.35)),
        )
        center_drift = abs(
            (float(landing_left) + landing_width_value / 2.0)
            - (float(previous_target_left) + previous_target_width / 2.0)
        )
        if center_drift > max_center_drift:
            return None
        assert current_result.landing_platform is not None
        reference = (previous.target[0], current_result.landing_platform[1])
        assert current_result.landing_platform_confidence is not None
        label_confidence = clamp(
            float(current_result.landing_platform_confidence),
            0.0,
            1.0,
        )
        label_source = "visible_platform"
    else:
        tuning = auto_tuning_config(config)
        if not (
            allow_temporal_fallback
            and bool(tuning.get("temporal_landing_enabled", True))
            and _temporal_piece_scale_is_consistent(previous, current_result, config)
        ):
            return None
        previous_crop_width = previous.crop_rect[2] - previous.crop_rect[0]
        if (
            previous_crop_width <= 0
            or abs(previous_crop_width - crop_width) > max(2.0, crop_width * 0.01)
            or previous_target_width >= crop_width * 0.90
        ):
            return None
        score_confidence_floor = float(
            (config.get("score") or {}).get("stage_min_confidence", 0.65)
        )
        scores_are_trustworthy = (
            previous.game_score is not None
            and current_result.game_score is not None
            and previous.game_score_confidence is not None
            and current_result.game_score_confidence is not None
            and previous.game_score_confidence >= score_confidence_floor
            and current_result.game_score_confidence >= score_confidence_floor
        )
        if scores_are_trustworthy:
            score_delta = current_result.game_score - previous.game_score
            max_score_delta = max(
                1,
                int((config.get("score") or {}).get("max_forward_step", 50)),
            )
            if not _valid_single_jump_score_delta(score_delta, max_score_delta):
                return None
        # The target geometry was already verified before the jump and the
        # camera has no horizontal scroll.  Require the current pawn body to
        # physically overlap the prior target and keep its centre on the top
        # surface (with only a tiny rasterization margin).  Expanding by a
        # whole pawn width would turn near misses into training labels.
        current_piece_left = crop_left + current_result.piece_bbox[0]
        current_piece_right = current_piece_left + piece_width
        piece_overlap = min(current_piece_right, previous_target_right) - max(
            current_piece_left,
            previous_target_left,
        )
        raster_margin = min(4.0, max(1.0, piece_width * 0.08))
        if piece_overlap < 1.0:
            return None
        if not (
            previous_target_left - raster_margin
            <= current_result.piece[0]
            <= previous_target_right + raster_margin
        ):
            return None
        reference = (previous.target[0], current_result.piece[1])
        label_confidence = min(
            clamp(float(previous.confidence), 0.0, 1.0),
            float(tuning.get("temporal_landing_confidence", 0.72)),
        )
        label_source = "temporal_horizontal"

    # Vertical coordinates are shifted by camera motion and also depend on the
    # visible top-surface shape.  Horizontal motion is stable, and jump input
    # controls a single along-track distance, so infer both screen and model
    # distance errors from the horizontal component instead of mixing a stale
    # pre-scroll X with a noisy post-scroll Y projection.
    error_x = float(current_result.piece[0] - reference[0])
    direction_x = float(previous.dx_px)
    screen_distance = max(0.0, float(previous.screen_distance_px))
    effective_distance = max(0.0, float(previous.effective_distance_px))
    minimum_horizontal_ratio = (
        float(
            auto_tuning_config(config).get(
                "temporal_min_horizontal_ratio",
                0.40,
            )
        )
        if label_source == "temporal_horizontal"
        else 0.25
    )
    if (
        abs(direction_x) <= 1.0
        or screen_distance <= 1.0
        or effective_distance <= 1.0
        or abs(direction_x) / screen_distance < minimum_horizontal_ratio
    ):
        return None

    signed_screen_error = error_x * screen_distance / direction_x
    signed_error = error_x * effective_distance / direction_x
    landing_error = abs(signed_screen_error)
    return LandingMeasurement(
        landing_error_px=landing_error,
        signed_error_px=signed_error,
        signed_screen_error_px=signed_screen_error,
        projection_ratio=1.0,
        reference_point=reference,
        label_confidence=label_confidence,
        label_source=label_source,
    )


def effective_distance_from_delta(dx: float, dy: float, config: dict[str, Any]) -> float:
    model = press_model_config(config)
    x_weight = float(model.get("x_weight", 1.0))
    y_weight = float(model.get("y_weight", 1.0))
    return math.hypot(dx * x_weight, dy * y_weight)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
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


def physics_unit_press_ms(
    result: DetectionResult,
    config: dict[str, Any],
) -> float | None:
    """Return physics press time at coefficient 1 for response auditing."""
    model = press_model_config(config)
    head_diameter = physics_head_diameter_px(result, model)
    if head_diameter is None:
        return None
    return physics_reference_press_ms(
        result.effective_distance_px,
        head_diameter,
        1.0,
    )


def _distance_px_from_input(distance_or_result: float | DetectionResult) -> float:
    if isinstance(distance_or_result, DetectionResult):
        return float(distance_or_result.effective_distance_px)
    return float(distance_or_result)


def segment_distance_from_input(distance_or_result: float | DetectionResult) -> float:
    """Return the camera-stable distance used to key local correction bins."""
    if isinstance(distance_or_result, DetectionResult):
        return float(distance_or_result.screen_distance_px)
    return float(distance_or_result)


@dataclass(frozen=True)
class StagePressContext:
    bucket: str
    piece_scale_ratio: float
    press_scale: float
    game_score: int | None
    score_confirmed: bool = True


def _clear_pending_score_observations(model: dict[str, Any]) -> None:
    model["stage_pending_reset"] = False
    model["stage_pending_reset_signature"] = ""
    model["stage_pending_forward_score"] = None
    model["stage_pending_forward_signature"] = ""


def begin_stage_session(config: dict[str, Any]) -> None:
    """Clear incomplete OCR votes while preserving learned score stages."""
    _clear_pending_score_observations(press_model_config(config))


def _raw_score_fields(
    result: DetectionResult,
) -> tuple[int | None, float | None]:
    if result.raw_game_score is not None or result.raw_game_score_confidence is not None:
        return result.raw_game_score, result.raw_game_score_confidence
    if result.stage_bucket is not None and result.stage_bucket.startswith("scale:"):
        # An annotated OCR-missing frame keeps game_score only as transition
        # state. Never reinterpret that backfill as a fresh OCR observation.
        return None, None
    return result.game_score, result.game_score_confidence


def _score_observation(
    result: DetectionResult,
    config: dict[str, Any],
) -> tuple[int | None, float | None, bool]:
    raw_score, raw_confidence = _raw_score_fields(result)
    score = (
        max(0, raw_score)
        if isinstance(raw_score, int) and not isinstance(raw_score, bool)
        else None
    )
    confidence = None
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and not math.isfinite(confidence):
            confidence = None
    confidence_floor = float(
        (config.get("score") or {}).get("stage_min_confidence", 0.65)
    )
    trusted = score is not None and (
        confidence is None or confidence >= confidence_floor
    )
    return score, confidence, trusted


def _stored_score(model: dict[str, Any], key: str) -> int | None:
    raw_score = model.get(key)
    if (
        isinstance(raw_score, (int, float))
        and not isinstance(raw_score, bool)
        and math.isfinite(float(raw_score))
        and float(raw_score) >= 0
    ):
        return int(raw_score)
    return None


def _score_observation_signature(result: DetectionResult) -> str:
    if result.observation_id:
        return result.observation_id
    return ":".join(
        str(value)
        for value in (
            result.piece[0],
            result.piece[1],
            result.target[0],
            result.target[1],
        )
    )


def _confirmed_stage_score(
    result: DetectionResult,
    config: dict[str, Any],
    model: dict[str, Any],
    *,
    create: bool,
) -> tuple[int | None, bool, bool, bool]:
    """Return score, transition state, new-game state, and OCR-observed state."""
    raw_score, confidence, trusted = _score_observation(result, config)
    last_score = _stored_score(model, "stage_last_score")
    if not trusted or raw_score is None:
        # Retain the last confirmed value for logs and future transitions, but
        # tell stage_press_context() to choose a scale:* bucket for this frame.
        return last_score, True, False, False
    if last_score is None:
        if create:
            model["stage_last_score"] = raw_score
            _clear_pending_score_observations(model)
        return raw_score, True, False, True

    signature = _score_observation_signature(result)
    reset_min_confidence = float(
        (config.get("score") or {}).get("reset_min_confidence", 0.80)
    )
    pending_reset = bool(model.get("stage_pending_reset", False))
    pending_reset_signature = str(model.get("stage_pending_reset_signature", ""))
    if pending_reset:
        if raw_score <= 2 and signature == pending_reset_signature:
            # annotate_stage_context() and calculate_press_ms() can inspect the
            # same frame.  One frame must never count as two reset votes.
            return raw_score, False, False, True
        if raw_score <= 2 and (confidence is None or confidence >= reset_min_confidence):
            if create:
                model["stage_last_score"] = raw_score
                _clear_pending_score_observations(model)
            return raw_score, True, True, True
        if create:
            model["stage_pending_reset"] = False
            model["stage_pending_reset_signature"] = ""

    if raw_score <= 1 and last_score > raw_score:
        if confidence is not None and confidence < reset_min_confidence:
            return last_score, True, False, True
        if create:
            model["stage_pending_reset"] = True
            model["stage_pending_reset_signature"] = signature
        # Use the safe base-stage press for the first possible new-game jump,
        # but do not mutate learned mappings until a distinct 0/1 frame agrees.
        return 0, False, False, True

    if raw_score < last_score:
        return last_score, True, False, True
    if raw_score == last_score:
        if create:
            model["stage_pending_forward_score"] = None
            model["stage_pending_forward_signature"] = ""
        return raw_score, True, False, True

    max_immediate_forward_step = max(
        1,
        int((config.get("score") or {}).get("max_immediate_forward_step", 10)),
    )
    max_forward_step = max(
        max_immediate_forward_step,
        int((config.get("score") or {}).get("max_forward_step", 50)),
    )
    score_delta = raw_score - last_score
    if _valid_single_jump_score_delta(score_delta, max_immediate_forward_step):
        if create:
            model["stage_last_score"] = raw_score
            model["stage_pending_forward_score"] = None
            model["stage_pending_forward_signature"] = ""
        return raw_score, True, False, True

    pending_forward = _stored_score(model, "stage_pending_forward_score")
    pending_forward_signature = str(
        model.get("stage_pending_forward_signature", "")
    )
    if (
        pending_forward is not None
        and signature != pending_forward_signature
        and pending_forward <= raw_score <= last_score + max_forward_step
    ):
        # A confirmed score can legitimately be several jumps ahead when OCR
        # was unavailable or rejected on intervening frames.  In that case the
        # aggregate delta need not itself be a valid *single-jump* award (for
        # example, 5 -> 18).  Two distinct captures plus the bounded forward
        # range are the appropriate safeguards here.
        if create:
            model["stage_last_score"] = raw_score
            model["stage_pending_forward_score"] = None
            model["stage_pending_forward_signature"] = ""
        return raw_score, True, False, True
    if create:
        model["stage_pending_forward_score"] = raw_score
        model["stage_pending_forward_signature"] = signature
    return last_score, False, False, True


def stage_feedback_updates_base_curve(context: StagePressContext) -> bool:
    """Only the anchored initial stage may train the cross-stage base curve."""
    return context.score_confirmed and context.bucket in {"base", "score:base"}


def stage_press_context(
    result: DetectionResult,
    config: dict[str, Any],
    *,
    create: bool = True,
) -> StagePressContext:
    model = press_model_config(config)
    if not bool(model.get("stage_adaptation_enabled", True)):
        return StagePressContext("base", 1.0, 1.0, result.game_score, True)

    dimensions = normalized_piece_dimensions(result)
    if dimensions is None:
        return StagePressContext("base", 1.0, 1.0, result.game_score, True)
    width_ratio, height_ratio = dimensions
    reference_width = _positive_float(model.get("stage_reference_width_ratio"))
    reference_height = _positive_float(model.get("stage_reference_height_ratio"))
    if reference_width is None or reference_height is None:
        reference_width = width_ratio
        reference_height = height_ratio
        if create:
            model["stage_reference_width_ratio"] = reference_width
            model["stage_reference_height_ratio"] = reference_height
    width_scale = width_ratio / max(1e-9, reference_width)
    height_scale = height_ratio / max(1e-9, reference_height)
    piece_scale_ratio = math.sqrt(max(1e-9, width_scale * height_scale))

    had_confirmed_score = _stored_score(model, "stage_last_score") is not None
    score, score_confirmed, confirmed_new_game, score_observed = _confirmed_stage_score(
        result,
        config,
        model,
        create=create,
    )
    score_bucket_size = max(1, int(model.get("stage_score_bucket_size", 50)))
    base_score_max = max(0, int(model.get("stage_base_score_max", 5)))
    first_score_observation = score_observed and not had_confirmed_score
    reset_scale_reference = confirmed_new_game or (
        first_score_observation
        and score is not None
        and score <= base_score_max
    )
    if reset_scale_reference:
        reference_width = width_ratio
        reference_height = height_ratio
        piece_scale_ratio = 1.0
        if create:
            # Score-specific multipliers and segments describe game physics
            # and remain reusable.  Only this run's scale reference advances.
            model["stage_reference_width_ratio"] = width_ratio
            model["stage_reference_height_ratio"] = height_ratio
            model["stage_last_multiplier"] = 1.0
            model["stage_scales"] = [
                item
                for item in model.get("stage_scales", [])
                if isinstance(item, dict)
                and str(item.get("stage_bucket", "")).startswith("score:")
            ]
            model["segment_corrections"] = [
                item
                for item in model.get("segment_corrections", [])
                if isinstance(item, dict)
                and not str(item.get("stage_bucket", "")).startswith("scale:")
            ]
            model["coefficient_corrections"] = [
                item
                for item in model.get("coefficient_corrections", [])
                if isinstance(item, dict)
                and not str(item.get("stage_bucket", "")).startswith("scale:")
            ]

    if score_observed and score is not None:
        if score <= base_score_max:
            bucket = "score:base"
            stage_order = 0.0
        else:
            bucket_index = score // score_bucket_size
            bucket = f"score:{bucket_index}"
            stage_order = float(bucket_index + 1)
    else:
        bucket_ratio = max(0.01, float(model.get("stage_piece_bucket_ratio", 0.06)))
        bucket_index = int(round(-math.log(max(1e-6, piece_scale_ratio)) / math.log1p(bucket_ratio)))
        bucket = f"scale:{bucket_index}"
        stage_order = float(bucket_index)

    bucket_family = bucket.partition(":")[0]

    entries = model.setdefault("stage_scales", [])
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict) and str(item.get("stage_bucket")) == bucket
        ),
        None,
    )
    if entry is None and create:
        family_prefix_scales: list[float] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            item_bucket = str(item.get("stage_bucket", ""))
            if item_bucket.partition(":")[0] != bucket_family:
                continue
            try:
                item_order = float(item.get("stage_order", math.inf))
            except (TypeError, ValueError, OverflowError):
                continue
            item_scale = _positive_float(item.get("press_scale"))
            if (
                item_scale is not None
                and math.isfinite(item_order)
                and item_order <= stage_order
            ):
                family_prefix_scales.append(item_scale)
        if bucket == "score:base":
            inherited = 1.0
        elif family_prefix_scales:
            inherited = max(family_prefix_scales)
        elif bucket_family == "score" and first_score_observation:
            # If OCR becomes available mid-game, transfer the currently used
            # scale fallback once so the press cannot drop abruptly.
            inherited = _positive_float(model.get("stage_last_multiplier")) or 1.0
        elif bucket_family == "scale":
            # Score -> scale is a safe one-way handoff when OCR disappears.
            # Scale fallback must never flow back into score buckets.
            inherited = _positive_float(model.get("stage_last_multiplier")) or 1.0
        else:
            inherited = 1.0
        entry = {
            "stage_bucket": bucket,
            "stage_order": stage_order,
            "piece_scale_ratio": piece_scale_ratio,
            "press_scale": inherited,
            "updates": 0,
            "timestamp": timestamp(),
        }
        if score is not None:
            entry["game_score"] = score
        entries.append(entry)
        # A session has only a handful of score buckets.  Bound malformed or
        # exceptionally long runs without disturbing ordering.
        if len(entries) > 32:
            del entries[:-32]
    fallback_scale = (
        (_positive_float(model.get("stage_last_multiplier")) or 1.0)
        if bucket_family == "scale"
        else 1.0
    )
    press_scale = (
        _positive_float(entry.get("press_scale"))
        if isinstance(entry, dict)
        else None
    ) or fallback_scale
    if bucket == "score:base":
        press_scale = 1.0
        if isinstance(entry, dict) and create:
            entry["press_scale"] = 1.0
    else:
        # Enforce the known non-decreasing response at lookup time too, so a
        # persisted mapping cannot return a later stage below an earlier one.
        prefix_scales: list[float] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            item_bucket = str(item.get("stage_bucket", ""))
            if item_bucket.partition(":")[0] != bucket_family:
                continue
            try:
                item_order = float(item.get("stage_order", math.inf))
            except (TypeError, ValueError, OverflowError):
                continue
            item_scale = _positive_float(item.get("press_scale"))
            if (
                item_scale is not None
                and math.isfinite(item_order)
                and item_order <= stage_order
            ):
                prefix_scales.append(item_scale)
        if prefix_scales:
            press_scale = max(press_scale, *prefix_scales)
            if isinstance(entry, dict) and create:
                entry["press_scale"] = press_scale
    if create and score_confirmed:
        model["stage_last_multiplier"] = press_scale
    return StagePressContext(
        bucket,
        piece_scale_ratio,
        press_scale,
        score,
        score_confirmed,
    )


def annotate_stage_context(
    result: DetectionResult,
    config: dict[str, Any],
) -> DetectionResult:
    raw_score, raw_score_confidence = _raw_score_fields(result)
    context = stage_press_context(result, config, create=True)
    confirmed_confidence = (
        raw_score_confidence
        if context.score_confirmed and raw_score == context.game_score
        else None
    )
    return replace(
        result,
        raw_game_score=raw_score,
        raw_game_score_confidence=raw_score_confidence,
        game_score=context.game_score,
        game_score_confidence=confirmed_confidence,
        piece_scale_ratio=context.piece_scale_ratio,
        stage_bucket=context.bucket,
        stage_press_scale=context.press_scale,
        stage_score_confirmed=context.score_confirmed,
    )


def update_stage_press_scale(
    config: dict[str, Any],
    result: DetectionResult,
    executed_press_ms: float,
    desired_press_ms: float,
) -> StagePressContext:
    context = stage_press_context(result, config, create=True)
    model = press_model_config(config)
    if (
        not bool(model.get("stage_adaptation_enabled", True))
        or not context.score_confirmed
        or stage_feedback_updates_base_curve(context)
        or executed_press_ms <= 0
        or desired_press_ms <= 0
        or not math.isfinite(executed_press_ms)
        or not math.isfinite(desired_press_ms)
    ):
        return context
    entries = model.setdefault("stage_scales", [])
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and str(item.get("stage_bucket")) == context.bucket
        ),
        None,
    )
    if entry is None:
        return context
    learning_rate = clamp(
        float(model.get("stage_scale_learning_rate", 0.55)),
        0.0,
        1.0,
    )
    max_step = clamp(
        float(model.get("stage_scale_max_step_ratio", 0.04)),
        0.0,
        0.50,
    )
    raw_ratio = desired_press_ms / executed_press_ms
    bounded_ratio = clamp(raw_ratio, 1.0 - max_step, 1.0 + max_step)
    updated = context.press_scale * math.exp(math.log(bounded_ratio) * learning_rate)
    prior_stage_scales: list[float | None] = []
    current_bucket_family = context.bucket.partition(":")[0]
    try:
        current_stage_order = float(entry.get("stage_order", 0.0))
    except (TypeError, ValueError, OverflowError):
        current_stage_order = 0.0
    for item in entries:
        if not isinstance(item, dict):
            continue
        item_bucket = str(item.get("stage_bucket", ""))
        if item_bucket.partition(":")[0] != current_bucket_family:
            continue
        try:
            item_order = float(item.get("stage_order", math.inf))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(item_order) and item_order < current_stage_order:
            prior_stage_scales.append(_positive_float(item.get("press_scale")))
    monotone_floor = max(
        [float(model.get("stage_scale_min", 0.70))]
        + [value for value in prior_stage_scales if value is not None]
    )
    updated = clamp(
        updated,
        monotone_floor,
        float(model.get("stage_scale_max", 1.40)),
    )
    entry["press_scale"] = updated
    entry["piece_scale_ratio"] = context.piece_scale_ratio
    entry["updates"] = int(entry.get("updates", 0)) + 1
    entry["last_press_ratio"] = raw_ratio
    entry["timestamp"] = timestamp()
    if context.game_score is not None:
        entry["game_score"] = context.game_score
    model["stage_last_multiplier"] = updated
    # Updating an earlier score stage raises the floor for every already
    # learned later stage.  Apply that prefix-max immediately instead of
    # waiting for each later bucket to receive another landing sample.
    for later_entry in entries:
        if not isinstance(later_entry, dict) or later_entry is entry:
            continue
        later_bucket = str(later_entry.get("stage_bucket", ""))
        if later_bucket.partition(":")[0] != current_bucket_family:
            continue
        try:
            later_order = float(later_entry.get("stage_order", math.inf))
        except (TypeError, ValueError, OverflowError):
            continue
        later_scale = _positive_float(later_entry.get("press_scale"))
        if (
            later_scale is not None
            and math.isfinite(later_order)
            and later_order > current_stage_order
            and later_scale < updated
        ):
            later_entry["press_scale"] = updated
            later_entry["timestamp"] = timestamp()
    return StagePressContext(
        context.bucket,
        context.piece_scale_ratio,
        updated,
        context.game_score,
        context.score_confirmed,
    )


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


def sample_stage_press_scale(sample: dict[str, Any]) -> float:
    scale = _positive_float(sample.get("stage_press_scale"))
    return scale if scale is not None else 1.0


def sample_base_training_press_ms(sample: dict[str, Any]) -> float:
    """Remove the score-stage multiplier before fitting the base curve."""
    return sample_training_press_ms(sample) / sample_stage_press_scale(sample)


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
            press_ms = sample_base_training_press_ms(sample)
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


def coefficient_bounds_for_distance(
    distance_px: float,
    model: dict[str, Any],
) -> tuple[int, float, float, float]:
    """Return a stable broad distance band for jump-coefficient learning."""
    band_size = max(10.0, float(model.get("coefficient_band_size_px", 50)))
    band_index = int(max(0.0, distance_px) // band_size)
    distance_min = band_index * band_size
    distance_max = distance_min + band_size
    return band_index, distance_min, distance_max, distance_min + band_size / 2.0


def _coefficient_correction_matches_bounds(
    correction: dict[str, Any],
    band_index: int,
    distance_min: float,
    distance_max: float,
    stage_bucket: str,
) -> bool:
    try:
        raw_index = correction.get("band_index")
        if isinstance(raw_index, bool) or float(raw_index).is_integer() is False:
            return False
        return (
            str(correction.get("stage_bucket", "base")) == stage_bucket
            and int(raw_index) == band_index
            and math.isclose(float(correction["distance_min_px"]), distance_min, abs_tol=1e-6)
            and math.isclose(float(correction["distance_max_px"]), distance_max, abs_tol=1e-6)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def coefficient_correction_entry(
    model: dict[str, Any],
    distance_px: float,
    *,
    create: bool = False,
    stage_bucket: str = "base",
) -> dict[str, Any] | None:
    band_index, distance_min, distance_max, band_center = coefficient_bounds_for_distance(
        distance_px,
        model,
    )
    corrections = model.setdefault("coefficient_corrections", [])
    for correction in corrections:
        if isinstance(correction, dict) and _coefficient_correction_matches_bounds(
            correction,
            band_index,
            distance_min,
            distance_max,
            stage_bucket,
        ):
            return correction
    if not create:
        return None
    correction = {
        "stage_bucket": stage_bucket,
        "band_index": band_index,
        "distance_min_px": distance_min,
        "distance_max_px": distance_max,
        "band_center_px": band_center,
        "coefficient": 1.0,
        "updates": 0,
        "timestamp": timestamp(),
    }
    corrections.append(correction)
    max_corrections = max(1, int(model.get("max_coefficient_corrections", 96)))
    if len(corrections) > max_corrections:
        del corrections[:-max_corrections]
    return correction


def coefficient_correction_ratio(
    distance_px: float,
    model: dict[str, Any],
    stage_bucket: str = "base",
) -> float:
    if not math.isfinite(distance_px) or distance_px < 0:
        return 1.0
    correction = coefficient_correction_entry(
        model,
        distance_px,
        create=False,
        stage_bucket=stage_bucket,
    )
    if correction is None:
        return 1.0
    try:
        ratio = float(correction.get("coefficient", 1.0))
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return ratio if math.isfinite(ratio) and ratio > 0 else 1.0


def record_coefficient_correction(
    config: dict[str, Any],
    distance_px: float,
    current_press_ms: float,
    desired_press_ms: float,
    stage_bucket: str = "base",
) -> float:
    """Learn a conservative multiplicative correction for a broad distance band."""
    tuning = auto_tuning_config(config)
    model = press_model_config(config)
    current = _positive_float(current_press_ms)
    desired = _positive_float(desired_press_ms)
    if not math.isfinite(distance_px) or distance_px < 0:
        return 1.0
    if (
        not bool(tuning.get("coefficient_correction_enabled", True))
        or current is None
        or desired is None
    ):
        return coefficient_correction_ratio(distance_px, model, stage_bucket)

    correction = coefficient_correction_entry(
        model,
        distance_px,
        create=True,
        stage_bucket=stage_bucket,
    )
    if correction is None:
        return 1.0
    previous = coefficient_correction_ratio(distance_px, model, stage_bucket)
    max_step = clamp(float(tuning.get("coefficient_max_step_ratio", 0.04)), 0.0, 0.25)
    observed_step = clamp(desired / current, 1.0 - max_step, 1.0 + max_step)
    learning_rate = clamp(float(tuning.get("coefficient_learning_rate", 0.35)), 0.0, 1.0)
    updated = previous * math.exp(math.log(observed_step) * learning_rate)
    max_adjustment = clamp(
        float(tuning.get("coefficient_max_adjustment_ratio", 0.12)),
        0.0,
        0.40,
    )
    updated = clamp(updated, 1.0 - max_adjustment, 1.0 + max_adjustment)
    correction["coefficient"] = updated
    correction["updates"] = int(correction.get("updates", 0)) + 1
    correction["last_observed_ratio"] = desired / current
    correction["last_current_press_ms"] = current
    correction["last_desired_press_ms"] = desired
    correction["timestamp"] = timestamp()
    return updated


def _segment_correction_matches_bounds(
    correction: dict[str, Any],
    segment_index: int,
    distance_min: float,
    distance_max: float,
    stage_bucket: str = "base",
) -> bool:
    try:
        raw_index = correction.get("segment_index")
        if isinstance(raw_index, bool) or float(raw_index).is_integer() is False:
            return False
        return (
            str(correction.get("stage_bucket", "base")) == stage_bucket
            and int(raw_index) == segment_index
            and math.isclose(float(correction["distance_min_px"]), distance_min, abs_tol=1e-6)
            and math.isclose(float(correction["distance_max_px"]), distance_max, abs_tol=1e-6)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def segment_correction_ms(
    distance_px: float,
    model: dict[str, Any],
    stage_bucket: str = "base",
) -> float:
    segment_index, distance_min, distance_max, _ = segment_bounds_for_distance(distance_px, model)
    for correction in model.get("segment_corrections", []):
        try:
            if _segment_correction_matches_bounds(
                correction,
                segment_index,
                distance_min,
                distance_max,
                stage_bucket,
            ):
                return float(correction.get("correction_ms", 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def segment_correction_entry(
    model: dict[str, Any],
    distance_px: float,
    create: bool = False,
    stage_bucket: str = "base",
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
            stage_bucket,
        ):
            return correction
    if not create:
        return None
    correction = {
        "stage_bucket": stage_bucket,
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


def segment_is_frozen(
    config: dict[str, Any],
    distance_px: float,
    stage_bucket: str = "base",
) -> bool:
    model = press_model_config(config)
    correction = segment_correction_entry(
        model,
        distance_px,
        create=False,
        stage_bucket=stage_bucket,
    )
    return bool(correction and correction.get("frozen", False))


def mark_segment_precision_hit(
    config: dict[str, Any],
    distance_px: float,
    landing_error: float,
    stage_bucket: str = "base",
) -> bool:
    tuning = auto_tuning_config(config)
    model = press_model_config(config)
    correction = segment_correction_entry(
        model,
        distance_px,
        create=True,
        stage_bucket=stage_bucket,
    )
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
    stage_bucket: str = "base",
) -> bool:
    tuning = auto_tuning_config(config)
    unfreeze_error = float(tuning.get("segment_unfreeze_error_px", 18))
    if landing_error < unfreeze_error:
        return False
    model = press_model_config(config)
    correction = segment_correction_entry(
        model,
        distance_px,
        create=False,
        stage_bucket=stage_bucket,
    )
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
    reference_press_ms: float | None = None,
    stage_bucket: str = "base",
) -> None:
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("segment_correction_enabled", True)):
        return

    model = press_model_config(config)
    existing = segment_correction_entry(
        model,
        distance_px,
        create=False,
        stage_bucket=stage_bucket,
    )
    if existing is not None and bool(existing.get("frozen", False)):
        return
    segment_index, distance_min, distance_max, segment_center = segment_bounds_for_distance(
        distance_px,
        model,
    )
    max_ratio = float(tuning.get("segment_max_correction_ratio", 0.18))
    base_press = _positive_float(reference_press_ms)
    if base_press is None:
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
            stage_bucket,
        ):
            previous = float(correction.get("correction_ms", 0.0))
            # correction_delta_ms is an increment relative to the prediction
            # that already contained `previous`; treating it as a new absolute
            # correction makes a same-direction update shrink or even reverse
            # the learned segment value.
            updated = previous + correction_delta_ms * learning_rate
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
            "stage_bucket": stage_bucket,
            "segment_index": segment_index,
            "distance_min_px": distance_min,
            "distance_max_px": distance_max,
            "segment_center_px": segment_center,
            "correction_ms": clamp(
                correction_delta_ms * learning_rate,
                -max_abs_correction,
                max_abs_correction,
            ),
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


def decay_segment_center_correction(
    config: dict[str, Any],
    distance_px: float,
    stage_bucket: str = "base",
) -> None:
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
            stage_bucket,
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
    desired_distance = max(1.0, current_distance - signed_error)
    current_curve_press = piecewise_press_ms(current_distance, model)
    desired_curve_press = piecewise_press_ms(desired_distance, model)
    if current_curve_press is None or desired_curve_press is None:
        # Keep the same detected piece/head geometry used by the executed
        # prediction.  Passing plain float distances here falls back to the
        # generic 80px physics head and badly under-corrects short hops whose
        # actual detected head is much narrower.
        desired_result = replace(
            previous,
            effective_distance_px=desired_distance,
            distance_px=desired_distance,
        )
        current_curve_press = reference_base_press_ms(previous, model, config)
        desired_curve_press = reference_base_press_ms(desired_result, model, config)
    stage_scale = stage_press_context(previous, config, create=True).press_scale
    current_curve_press *= stage_scale
    desired_curve_press *= stage_scale
    adjustment = clamp(
        (desired_curve_press - current_curve_press) * learning_rate,
        -max_adjustment,
        max_adjustment,
    )

    executable_minimum = minimum_press_ms_for_distance(current_distance, model, config)
    executable_maximum = float(config["max_press_ms"])
    adjustment_minimum = max(executable_minimum, press_ms - max_adjustment)
    adjustment_maximum = min(executable_maximum, press_ms + max_adjustment)
    adjusted_press = clamp(press_ms + adjustment, adjustment_minimum, adjustment_maximum)
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
    try:
        confidence = float(sample.get("confidence", 0.75))
    except (TypeError, ValueError, OverflowError):
        confidence = 0.75
    if not math.isfinite(confidence):
        confidence = 0.75
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
        residual = abs(sample_base_training_press_ms(sample) - predicted)
        if residual <= outlier_threshold:
            clean.append(sample)
    return clean


def fit_press_model(config: dict[str, Any]) -> dict[str, Any]:
    model = press_model_config(config)
    samples: list[dict[str, Any]] = []
    for sample in model.get("samples", []):
        if (
            not isinstance(sample, dict)
            or not base_curve_sample_is_eligible(sample, model)
            or sample_training_press_ms(sample) <= 0
        ):
            continue
        try:
            dx = float(sample["dx_px"])
            dy = float(sample["dy_px"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(dx) and math.isfinite(dy):
            samples.append(sample)
    if not samples:
        model["base_curve_fit_version"] = CURRENT_BASE_CURVE_FIT_VERSION
        model["type"] = "weighted_euclidean"
        model["x_weight"] = 1.0
        model["y_weight"] = 1.0
        model["slope_ms_per_px"] = None
        model["offset_ms"] = 0.0
        model["fit_rmse_ms"] = None
        model["sample_count"] = 0
        model["curve_points"] = []
        model.pop("outlier_ratio", None)
        config["press_ms_per_px"] = None
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
        durations = [sample_base_training_press_ms(sample) for sample in samples]
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
        durations = [sample_base_training_press_ms(sample) for sample in samples]
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
            clean_durations = [sample_base_training_press_ms(s) for s in clean_samples]
            if fit_offset:
                refitted = _weighted_fit_line_with_offset(clean_distances, clean_durations, clean_weights)
                if refitted is not None and refitted[1] >= -250 and refitted[1] <= 350:
                    slope, offset, rmse = refitted
            else:
                slope, rmse = _weighted_fit_line_through_origin(clean_distances, clean_durations, clean_weights)
            model["outlier_ratio"] = 1.0 - len(clean_samples) / len(samples)

    model["type"] = "weighted_euclidean"
    model["base_curve_fit_version"] = CURRENT_BASE_CURVE_FIT_VERSION
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
        if isinstance(sample, dict)
        and base_curve_sample_is_eligible(sample, model)
        and sample_training_press_ms(sample) > 0
    ]
    if not samples:
        return None

    y_weight = float(model.get("y_weight", 1.0))
    min_anchor_distance = float(model.get("short_hop_min_anchor_distance_px", 80))
    short_samples: list[tuple[float, float]] = []
    for sample in samples:
        try:
            sample_distance = sample_effective_distance(sample, y_weight)
            sample_press = sample_base_training_press_ms(sample)
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
            sample_press = sample_base_training_press_ms(sample)
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
    segment_distance_px = segment_distance_from_input(distance_or_result)

    model = press_model_config(config)
    eligible_sample_count = sum(
        1
        for sample in model.get("samples", [])
        if isinstance(sample, dict)
        and base_curve_sample_is_eligible(sample, model)
    )
    if (
        bool(model.get("curve_enabled", True))
        and len(model.get("curve_points", [])) < int(model.get("curve_min_samples", 3))
        and eligible_sample_count >= int(model.get("curve_min_samples", 3))
    ):
        fit_press_model(config)
        model = press_model_config(config)
    press_ms = reference_base_press_ms(distance_or_result, model, config)
    press_ms, _ = learned_curve_adjusted_press_ms(press_ms, distance_px, model)
    short_cap = short_hop_press_cap_ms(distance_px, model)
    if short_cap is not None:
        press_ms = min(press_ms, short_cap)
    stage_context = (
        stage_press_context(distance_or_result, config, create=True)
        if isinstance(distance_or_result, DetectionResult)
        else StagePressContext("base", 1.0, 1.0, None)
    )
    press_ms *= stage_context.press_scale
    press_ms *= coefficient_correction_ratio(
        segment_distance_px,
        model,
        stage_context.bucket,
    )
    segment_correction = segment_correction_ms(
        segment_distance_px,
        model,
        stage_context.bucket,
    )
    segment_max_ratio = float(
        auto_tuning_config(config).get("segment_max_correction_ratio", 0.18)
    )
    segment_max_abs = max(8.0, abs(press_ms) * segment_max_ratio)
    press_ms += clamp(segment_correction, -segment_max_abs, segment_max_abs)
    failure_cap = failure_press_cap_ms(distance_px, model, config)
    if failure_cap is not None:
        press_ms = min(press_ms, failure_cap)
    minimum_press = minimum_press_ms_for_distance(distance_px, model, config)
    return clamp(press_ms, minimum_press, float(config["max_press_ms"]))


def calibration_sample_from_result(
    result: DetectionResult,
    duration_ms: float,
    *,
    stage_press_scale: float | None = None,
    stage_bucket: str | None = None,
    piece_scale_ratio: float | None = None,
) -> dict[str, Any]:
    normalized = normalized_piece_dimensions(result)
    return {
        "timestamp": timestamp(),
        "piece": [result.piece[0], result.piece[1]],
        "target": [result.target[0], result.target[1]],
        "distance_px": result.effective_distance_px,
        "dx_px": result.dx_px,
        "dy_px": result.dy_px,
        "screen_distance_px": result.screen_distance_px,
        "effective_distance_px": result.effective_distance_px,
        "piece_width_px": result.piece_bbox[2],
        "piece_height_px": result.piece_bbox[3],
        "piece_width_ratio": normalized[0] if normalized is not None else None,
        "piece_height_ratio": normalized[1] if normalized is not None else None,
        "piece_scale_ratio": (
            piece_scale_ratio
            if piece_scale_ratio is not None
            else result.piece_scale_ratio
        ),
        "stage_bucket": stage_bucket or result.stage_bucket or "base",
        "stage_press_scale": (
            stage_press_scale
            if stage_press_scale is not None
            else result.stage_press_scale or 1.0
        ),
        "game_score": result.game_score,
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
