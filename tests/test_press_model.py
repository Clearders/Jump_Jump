from __future__ import annotations

import copy
from dataclasses import replace
import math
import unittest
from pathlib import Path

from jumpjump.config import DEFAULT_CONFIG, press_model_config
from jumpjump.press_model import (
    annotate_stage_context,
    begin_stage_session,
    calibration_sample_from_result,
    calculate_press_ms,
    center_adjusted_press_ms,
    failure_press_cap_ms,
    fit_press_model,
    linear_reference_press_ms,
    mark_segment_precision_hit,
    measure_landing,
    minimum_press_ms_for_distance,
    maybe_unfreeze_segment_for_error,
    physics_reference_press_ms,
    piecewise_press_ms,
    record_segment_center_correction,
    reference_base_press_ms,
    sample_base_training_press_ms,
    sample_training_press_ms,
    segment_bounds_for_distance,
    segment_correction_entry,
    segment_correction_ms,
    segment_distance_from_input,
    segment_is_frozen,
    stage_press_context,
    stage_feedback_updates_base_curve,
    update_stage_press_scale,
)
from jumpjump.types import DetectionResult


def fresh_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def detection(
    *,
    piece: tuple[int, int],
    target: tuple[int, int],
    distance: float,
    dx: float | None = None,
    dy: float = 0.0,
    landing_platform: tuple[int, int] | None = None,
    landing_platform_confidence: float | None = None,
) -> DetectionResult:
    dx = distance if dx is None else dx
    return DetectionResult(
        piece=piece,
        target=target,
        piece_bbox=(piece[0] - 10, piece[1] - 40, 20, 40),
        target_bbox=(target[0] - 20, target[1], 40, 20),
        crop_rect=(0, 0, 400, 700),
        dx_px=dx,
        dy_px=dy,
        screen_distance_px=distance,
        effective_distance_px=distance,
        distance_px=distance,
        confidence=0.9,
        debug_path=Path("debug.png"),
        landing_platform=landing_platform,
        landing_platform_bbox=(70, 0, 60, 20) if landing_platform else None,
        landing_platform_confidence=landing_platform_confidence,
    )


class PressModelTests(unittest.TestCase):
    def test_short_hop_uses_distance_specific_press_floor(self) -> None:
        config = fresh_config()
        model = press_model_config(config)

        self.assertEqual(minimum_press_ms_for_distance(160.0, model, config), 80.0)
        self.assertEqual(minimum_press_ms_for_distance(200.0, model, config), 180.0)

    def test_short_hop_prediction_is_not_flattened_to_normal_minimum(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["samples"] = [
            {
                "dx_px": 180.0,
                "dy_px": 0.0,
                "press_ms": 180.0,
                "confidence": 0.9,
                "base_curve_eligible": True,
            }
        ]

        self.assertAlmostEqual(calculate_press_ms(160.0, config), 160.0)
        self.assertEqual(calculate_press_ms(40.0, config), 80.0)

    def test_short_hop_feedback_target_respects_executable_floor(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(200, 0),
            target=(260, 0),
            distance=60.0,
            dx=60.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )

        adjusted = center_adjusted_press_ms(previous, current, 85.0, config)

        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted[0], 80.0)

    def test_burningcl_linear_reference_formula(self) -> None:
        self.assertAlmostEqual(linear_reference_press_ms(500.0, 1080.0), 695.0)

    def test_wangshub_physics_reference_formula(self) -> None:
        self.assertAlmostEqual(
            physics_reference_press_ms(400.0, 80.0, 1.392),
            586.3053005392085,
        )

    def test_calculate_press_uses_physics_with_detection_piece_width(self) -> None:
        config = fresh_config()
        result = detection(piece=(0, 0), target=(100, 0), distance=100.0)

        press_ms = calculate_press_ms(result, config)

        self.assertAlmostEqual(
            press_ms,
            physics_reference_press_ms(100.0, 20.0 * 1.15, 1.392),
        )

    def test_calculate_press_uses_default_physics_head_for_plain_distance(self) -> None:
        config = fresh_config()

        press_ms = calculate_press_ms(400.0, config)

        self.assertAlmostEqual(
            press_ms,
            physics_reference_press_ms(400.0, 80.0, 1.392),
        )

    def test_calculate_press_falls_back_to_linear_when_physics_unavailable(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["physics_head_diameter_px"] = 0
        model["physics_default_head_diameter_px"] = 0

        press_ms = calculate_press_ms(500.0, config)

        self.assertAlmostEqual(press_ms, linear_reference_press_ms(500.0, 1080.0))

    def test_curve_correction_is_bounded_over_physics_base(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["curve_min_samples"] = 1
        model["curve_points"] = [{"distance_px": 400.0, "press_ms": 1000.0}]
        base_press = physics_reference_press_ms(400.0, 80.0, 1.392)

        press_ms = calculate_press_ms(400.0, config)

        self.assertAlmostEqual(press_ms, base_press * 1.35)

    def test_piecewise_curve_interpolates_and_extrapolates(self) -> None:
        model = {
            "curve_enabled": True,
            "curve_min_samples": 2,
            "curve_points": [
                {"distance_px": 100.0, "press_ms": 200.0},
                {"distance_px": 200.0, "press_ms": 450.0},
            ],
            "slope_ms_per_px": 2.5,
        }

        self.assertEqual(piecewise_press_ms(50.0, model), 100.0)
        self.assertEqual(piecewise_press_ms(150.0, model), 325.0)
        self.assertEqual(piecewise_press_ms(220.0, model), 500.0)

    def test_training_press_prefers_explicit_then_adjusted_then_raw(self) -> None:
        sample = {"press_ms": 220.0}

        self.assertEqual(sample_training_press_ms(sample), 220.0)

        sample["center_adjusted_press_ms"] = 205.0
        self.assertEqual(sample_training_press_ms(sample), 205.0)

        sample["training_press_ms"] = 198.0
        self.assertEqual(sample_training_press_ms(sample), 198.0)

        sample["training_press_ms"] = 0.0
        self.assertEqual(sample_training_press_ms(sample), 205.0)

    def test_fit_ignores_incomplete_manual_sample(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["samples"] = [{"result_type": "manual", "press_ms": 260.0}]

        fit_press_model(config)

        self.assertIsNone(model["slope_ms_per_px"])

    def test_fit_ignores_manual_sample_marked_as_high_stage(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["samples"] = [
            {
                "result_type": "manual",
                "press_ms": 260.0,
                "dx_px": 100.0,
                "dy_px": 0.0,
                "base_curve_eligible": False,
            }
        ]

        fit_press_model(config)

        self.assertIsNone(model["slope_ms_per_px"])

    def test_fit_ignores_unmarked_samples_with_high_stage_metadata(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["samples"] = [
            {
                "result_type": "manual",
                "press_ms": 260.0,
                "dx_px": 100.0,
                "dy_px": 0.0,
                "game_score": 100,
                "stage_bucket": "base",
            },
            {
                "result_type": "auto_adjusted",
                "feedback_version": 3,
                "landing_error_px": 10.0,
                "press_ms": 260.0,
                "dx_px": 100.0,
                "dy_px": 0.0,
                "stage_bucket": "scale:0",
            },
        ]
        model["slope_ms_per_px"] = 2.6
        model["curve_points"] = [{"distance_px": 100.0, "press_ms": 260.0}]

        fit_press_model(config)

        self.assertIsNone(model["slope_ms_per_px"])
        self.assertEqual(model["curve_points"], [])

    def test_markerless_stale_base_score_with_shrunken_piece_is_ignored(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["samples"] = [
            {
                "result_type": "manual",
                "press_ms": 130.0,
                "dx_px": 50.0,
                "dy_px": 0.0,
                "game_score": 0,
                "stage_bucket": "score:base",
                "piece_scale_ratio": 0.70,
            }
        ]
        model["slope_ms_per_px"] = 2.6
        model["curve_points"] = [{"distance_px": 50.0, "press_ms": 130.0}]

        fit_press_model(config)

        self.assertIsNone(model["slope_ms_per_px"])
        self.assertEqual(model["curve_points"], [])

    def test_markerless_scale_one_samples_cannot_rebuild_base_curve(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["curve_min_samples"] = 3
        model["samples"] = [
            {
                "result_type": "manual",
                "press_ms": distance * 6.0,
                "dx_px": distance,
                "dy_px": 0.0,
                "game_score": 0,
                "stage_bucket": "score:base",
                "piece_scale_ratio": 0.95,
            }
            for distance in (40.0, 50.0, 60.0)
        ]

        fit_press_model(config)

        self.assertIsNone(model["slope_ms_per_px"])
        self.assertEqual(model["curve_points"], [])

    def test_fit_and_curve_use_training_press_ms(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["curve_min_samples"] = 3
        model["samples"] = [
            {
                "dx_px": 100.0,
                "dy_px": 0.0,
                "press_ms": 1000.0,
                "training_press_ms": 100.0,
                "confidence": 0.9,
                "base_curve_eligible": True,
            },
            {
                "dx_px": 200.0,
                "dy_px": 0.0,
                "press_ms": 2000.0,
                "training_press_ms": 200.0,
                "confidence": 0.9,
                "base_curve_eligible": True,
            },
            {
                "dx_px": 300.0,
                "dy_px": 0.0,
                "press_ms": 3000.0,
                "training_press_ms": 300.0,
                "confidence": 0.9,
                "base_curve_eligible": True,
            },
        ]

        fit_press_model(config)

        self.assertAlmostEqual(model["slope_ms_per_px"], 1.0)
        self.assertEqual(
            [point["press_ms"] for point in model["curve_points"]],
            [100.0, 200.0, 300.0],
        )

    def test_segment_bounds_use_two_pixel_bins(self) -> None:
        config = fresh_config()
        model = press_model_config(config)

        segment_index, distance_min, distance_max, center = segment_bounds_for_distance(
            14.2,
            model,
        )

        self.assertEqual(segment_index, 7)
        self.assertEqual(distance_min, 14.0)
        self.assertEqual(distance_max, 16.0)
        self.assertEqual(center, 15.0)

    def test_segment_bounds_do_not_change_as_samples_accumulate(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        before = segment_bounds_for_distance(280.0, model)
        model["samples"] = [
            {"distance_px": 100.0, "press_ms": 200.0},
            {"distance_px": 500.0, "press_ms": 800.0},
        ]

        self.assertEqual(segment_bounds_for_distance(280.0, model), before)

    def test_precision_hits_freeze_segment(self) -> None:
        config = fresh_config()

        for _ in range(3):
            mark_segment_precision_hit(config, 42.0, 4.0)

        self.assertTrue(segment_is_frozen(config, 42.0))

    def test_frozen_segment_small_error_does_not_modify_correction(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["slope_ms_per_px"] = 2.0
        for _ in range(3):
            mark_segment_precision_hit(config, 42.0, 4.0)

        record_segment_center_correction(config, 42.0, 30.0, 12.0, 1.0)

        self.assertEqual(segment_correction_ms(42.0, model), 0.0)
        self.assertTrue(segment_is_frozen(config, 42.0))

    def test_large_error_unfreezes_segment_for_relearning(self) -> None:
        config = fresh_config()
        for _ in range(3):
            mark_segment_precision_hit(config, 42.0, 4.0)

        self.assertTrue(maybe_unfreeze_segment_for_error(config, 42.0, 19.0))
        self.assertFalse(segment_is_frozen(config, 42.0))

    def test_moderate_error_keeps_segment_frozen(self) -> None:
        config = fresh_config()
        for _ in range(3):
            mark_segment_precision_hit(config, 42.0, 3.0)

        self.assertFalse(maybe_unfreeze_segment_for_error(config, 42.0, 9.0))
        self.assertTrue(segment_is_frozen(config, 42.0))

    def test_center_adjustment_uses_local_curve_delta(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["center_learning_rate"] = 0.5
        model = press_model_config(config)
        model["curve_min_samples"] = 2
        model["slope_ms_per_px"] = 2.0
        model["curve_points"] = [
            {"distance_px": 100.0, "press_ms": 200.0},
            {"distance_px": 200.0, "press_ms": 500.0},
        ]
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(112, 0),
            target=(180, 0),
            distance=68.0,
            dx=68.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )

        adjusted = center_adjusted_press_ms(previous, current, 200.0, config)

        self.assertIsNotNone(adjusted)
        adjusted_press, signed_error, projection_ratio = adjusted
        self.assertAlmostEqual(adjusted_press, 188.0)
        self.assertEqual(signed_error, 12.0)
        self.assertEqual(projection_ratio, 1.0)

    def test_center_adjustment_keeps_detected_physics_head(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["center_learning_rate"] = 1.0
        config["auto_tuning"]["center_max_adjustment_ratio"] = 1.0
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(120, 0),
            target=(260, 0),
            distance=140.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )
        model = press_model_config(config)
        executed_press = reference_base_press_ms(previous, model, config)
        desired = replace(
            previous,
            effective_distance_px=80.0,
            distance_px=80.0,
        )
        expected = executed_press + (
            reference_base_press_ms(desired, model, config) - executed_press
        )

        adjusted = center_adjusted_press_ms(
            previous,
            current,
            executed_press,
            config,
        )

        self.assertIsNotNone(adjusted)
        self.assertAlmostEqual(adjusted[0], expected)

    def test_center_learning_rate_controls_adjustment_magnitude(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(120, 0),
            target=(180, 0),
            distance=60.0,
            dx=60.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )
        config["auto_tuning"]["center_max_adjustment_ratio"] = 1.0
        executed_press = reference_base_press_ms(
            previous,
            press_model_config(config),
            config,
        )

        config["auto_tuning"]["center_learning_rate"] = 0.25
        slow = center_adjusted_press_ms(previous, current, executed_press, config)
        config["auto_tuning"]["center_learning_rate"] = 1.0
        fast = center_adjusted_press_ms(previous, current, executed_press, config)

        self.assertIsNotNone(slow)
        self.assertIsNotNone(fast)
        self.assertGreater(slow[0], fast[0])

    def test_landing_measurement_uses_post_scroll_platform_y(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0, dx=100.0)
        current = detection(
            piece=(112, 650),
            target=(200, 400),
            distance=100.0,
            landing_platform=(100, 650),
            landing_platform_confidence=0.88,
        )

        measurement = measure_landing(previous, current, config)

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.reference_point, (100, 650))
        self.assertEqual(measurement.landing_error_px, 12.0)
        self.assertGreater(measurement.signed_error_px, 0.0)

    def test_temporal_horizontal_landing_recovers_occluded_platform(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0)
        current = detection(piece=(112, 650), target=(220, 400), distance=108.0)

        measurement = measure_landing(
            previous,
            current,
            config,
            allow_temporal_fallback=True,
        )

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.label_source, "temporal_horizontal")
        self.assertEqual(measurement.reference_point, (100, 650))
        self.assertEqual(measurement.signed_error_px, 12.0)
        self.assertGreaterEqual(measurement.label_confidence, 0.55)

    def test_temporal_horizontal_landing_rejects_old_piece_position(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0)
        current = detection(piece=(0, 650), target=(220, 400), distance=108.0)

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_rejects_near_miss_without_overlap(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0)
        current = detection(piece=(69, 650), target=(220, 400), distance=151.0)

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_rejects_uniform_abrupt_scale_change(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0)
        current = detection(piece=(112, 650), target=(220, 400), distance=108.0)
        current = replace(current, piece_bbox=(104, 618, 16, 32))

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_rejects_stale_equal_score(self) -> None:
        config = fresh_config()
        previous = replace(
            detection(piece=(0, 500), target=(100, 300), distance=100.0),
            game_score=20,
            game_score_confidence=0.9,
        )
        current = replace(
            detection(piece=(112, 650), target=(220, 400), distance=108.0),
            game_score=20,
            game_score_confidence=0.9,
        )

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_accepts_confirmed_score_increment(self) -> None:
        config = fresh_config()
        previous = replace(
            detection(piece=(0, 500), target=(100, 300), distance=100.0),
            game_score=20,
            game_score_confidence=0.9,
        )
        current = replace(
            detection(piece=(112, 650), target=(220, 400), distance=108.0),
            game_score=21,
            game_score_confidence=0.9,
        )

        self.assertIsNotNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_accepts_large_bonus_score_increment(self) -> None:
        config = fresh_config()
        previous = replace(
            detection(piece=(0, 500), target=(100, 300), distance=100.0),
            game_score=20,
            game_score_confidence=0.9,
        )
        current = replace(
            detection(piece=(112, 650), target=(220, 400), distance=108.0),
            game_score=50,
            game_score_confidence=0.9,
        )

        self.assertIsNotNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_rejects_impossible_odd_score_increment(self) -> None:
        config = fresh_config()
        previous = replace(
            detection(piece=(0, 500), target=(100, 300), distance=100.0),
            game_score=20,
            game_score_confidence=0.9,
        )
        current = replace(
            detection(piece=(112, 650), target=(220, 400), distance=108.0),
            game_score=25,
            game_score_confidence=0.9,
        )

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_temporal_horizontal_landing_rejects_nonuniform_piece_scale(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 500), target=(100, 300), distance=100.0)
        current = detection(piece=(112, 650), target=(220, 400), distance=108.0)
        current = replace(current, piece_bbox=(92, 610, 40, 40))

        self.assertIsNone(
            measure_landing(
                previous,
                current,
                config,
                allow_temporal_fallback=True,
            )
        )

    def test_vertical_camera_noise_does_not_create_press_error(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(100, 20),
            target=(200, 0),
            distance=100.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )

        measurement = measure_landing(previous, current, config)

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.landing_error_px, 0.0)
        self.assertEqual(measurement.signed_error_px, 0.0)
        self.assertEqual(measurement.signed_screen_error_px, 0.0)
        self.assertEqual(measurement.projection_ratio, 1.0)

    def test_landing_measurement_separates_screen_and_effective_units(self) -> None:
        config = fresh_config()
        screen_distance = math.hypot(100.0, -100.0)
        effective_distance = math.hypot(100.0, -150.0)
        previous = detection(
            piece=(0, 100),
            target=(100, 0),
            distance=screen_distance,
            dx=100.0,
            dy=-100.0,
        )
        previous = replace(
            previous,
            effective_distance_px=effective_distance,
            distance_px=effective_distance,
        )
        current = detection(
            piece=(110, 40),
            target=(260, 0),
            distance=150.0,
            landing_platform=(100, 40),
            landing_platform_confidence=0.9,
        )

        measurement = measure_landing(previous, current, config)

        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement.signed_screen_error_px, math.sqrt(200.0))
        self.assertAlmostEqual(measurement.signed_error_px, math.sqrt(325.0))

    def test_landing_measurement_rejects_a_different_platform_bbox(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(112, 0),
            target=(220, 0),
            distance=108.0,
            landing_platform=(260, 0),
            landing_platform_confidence=0.9,
        )
        current = replace(current, landing_platform_bbox=(240, 0, 60, 20))

        self.assertIsNone(measure_landing(previous, current, config))

    def test_landing_measurement_rejects_implausible_platform_width(self) -> None:
        config = fresh_config()
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(
            piece=(110, 0),
            target=(260, 0),
            distance=150.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )
        current = replace(current, landing_platform_bbox=(0, 0, 300, 20))

        self.assertIsNone(measure_landing(previous, current, config))

    def test_mirrored_overshoot_has_the_same_positive_sign(self) -> None:
        config = fresh_config()
        right_previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        right_current = detection(
            piece=(112, 0),
            target=(200, 0),
            distance=88.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )
        left_previous = detection(
            piece=(200, 0),
            target=(100, 0),
            distance=100.0,
            dx=-100.0,
        )
        left_current = detection(
            piece=(88, 0),
            target=(0, 0),
            distance=88.0,
            dx=-88.0,
            landing_platform=(100, 0),
            landing_platform_confidence=0.9,
        )

        right = measure_landing(right_previous, right_current, config)
        left = measure_landing(left_previous, left_current, config)

        self.assertIsNotNone(right)
        self.assertIsNotNone(left)
        self.assertEqual(right.signed_error_px, 12.0)
        self.assertEqual(left.signed_error_px, 12.0)

    def test_segment_feedback_accumulates_incremental_delta(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["segment_correction_learning_rate"] = 0.5
        config["auto_tuning"]["segment_max_correction_ratio"] = 1.0
        model = press_model_config(config)

        record_segment_center_correction(config, 100.0, 40.0, -20.0, 1.0)
        self.assertEqual(segment_correction_ms(100.0, model), 20.0)

        record_segment_center_correction(config, 100.0, 5.0, -5.0, 1.0)
        self.assertEqual(segment_correction_ms(100.0, model), 22.5)

    def test_segment_lookup_uses_screen_distance_when_model_weight_changes(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["segment_correction_learning_rate"] = 1.0
        result = detection(
            piece=(0, 100),
            target=(100, 0),
            distance=math.hypot(100.0, -100.0),
            dx=100.0,
            dy=-100.0,
        )
        result = replace(
            result,
            effective_distance_px=math.hypot(100.0, -150.0),
            distance_px=math.hypot(100.0, -150.0),
        )
        model = press_model_config(config)
        record_segment_center_correction(
            config,
            result.screen_distance_px,
            10.0,
            5.0,
            1.0,
        )

        self.assertEqual(segment_distance_from_input(result), result.screen_distance_px)
        self.assertAlmostEqual(
            segment_correction_ms(segment_distance_from_input(result), model),
            10.0,
        )
        reweighted = replace(
            result,
            effective_distance_px=math.hypot(100.0, -152.5),
            distance_px=math.hypot(100.0, -152.5),
        )
        self.assertAlmostEqual(
            segment_correction_ms(segment_distance_from_input(reweighted), model),
            10.0,
        )

    def test_segment_cap_uses_effective_prediction_not_screen_bin_center(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["segment_correction_learning_rate"] = 1.0
        config["auto_tuning"]["segment_max_correction_ratio"] = 0.12
        model = press_model_config(config)
        model["curve_points"] = [
            {"distance_px": 100.0, "press_ms": 200.0},
            {"distance_px": 200.0, "press_ms": 400.0},
        ]
        effective_press = 2.0 * math.hypot(100.0, -150.0)

        record_segment_center_correction(
            config,
            math.hypot(100.0, -100.0),
            100.0,
            20.0,
            1.0,
            reference_press_ms=effective_press,
        )

        correction = segment_correction_ms(math.hypot(100.0, -100.0), model)
        self.assertAlmostEqual(correction, effective_press * 0.12)

    def test_segment_correction_is_reclamped_against_current_prediction(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["segment_max_correction_ratio"] = 0.10
        model = press_model_config(config)
        model["curve_points"] = [
            {"distance_px": 100.0, "press_ms": 200.0},
            {"distance_px": 200.0, "press_ms": 400.0},
            {"distance_px": 300.0, "press_ms": 600.0},
        ]
        result = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        without_segment = calculate_press_ms(result, config)
        model["segment_corrections"] = [
            {
                "segment_index": 50,
                "stage_bucket": "scale:0",
                "distance_min_px": 100.0,
                "distance_max_px": 102.0,
                "correction_ms": 100.0,
            }
        ]
        with_segment = calculate_press_ms(result, config)

        self.assertAlmostEqual(with_segment - without_segment, without_segment * 0.10)

    def test_stage_multiplier_is_learned_and_inherited_by_next_score_bucket(self) -> None:
        config = fresh_config()
        config["press_model"]["stage_scale_learning_rate"] = 1.0
        begin_stage_session(config)
        base = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=0,
        )
        anchored = update_stage_press_scale(config, base, 200.0, 208.0)
        self.assertEqual(anchored.bucket, "score:base")
        self.assertEqual(anchored.press_scale, 1.0)
        score_10 = annotate_stage_context(replace(base, game_score=10), config)

        updated = update_stage_press_scale(config, score_10, 200.0, 208.0)

        self.assertEqual(updated.bucket, "score:0")
        self.assertAlmostEqual(updated.press_scale, 1.04)
        for score in (20, 30, 40, 50):
            stage_press_context(
                replace(score_10, game_score=score, raw_game_score=score),
                config,
            )
        score_55 = replace(score_10, game_score=55, raw_game_score=55)
        next_stage = stage_press_context(score_55, config)
        self.assertEqual(next_stage.bucket, "score:1")
        self.assertAlmostEqual(next_stage.press_scale, 1.04)

    def test_new_game_score_reset_requires_distinct_frame_and_preserves_learning(self) -> None:
        config = fresh_config()
        config["press_model"]["stage_scale_learning_rate"] = 1.0
        begin_stage_session(config)
        high = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=100,
        )
        update_stage_press_scale(config, high, 200.0, 208.0)

        first_zero = stage_press_context(replace(high, game_score=0), config)
        confirmed = stage_press_context(
            replace(
                high,
                piece=(5, 5),
                target=(110, 0),
                game_score=1,
            ),
            config,
        )

        self.assertEqual(first_zero.bucket, "score:base")
        self.assertFalse(first_zero.score_confirmed)
        self.assertEqual(first_zero.press_scale, 1.0)
        self.assertEqual(confirmed.bucket, "score:base")
        self.assertTrue(confirmed.score_confirmed)
        self.assertEqual(confirmed.press_scale, 1.0)
        self.assertTrue(
            any(
                item["stage_bucket"] == "score:2"
                for item in config["press_model"]["stage_scales"]
            )
        )

    def test_score_reset_below_first_bucket_is_confirmed(self) -> None:
        config = fresh_config()
        score_40 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=40,
        )
        stage_press_context(score_40, config)

        provisional = stage_press_context(replace(score_40, game_score=0), config)
        reset = stage_press_context(
            replace(score_40, piece=(5, 5), target=(110, 0), game_score=1),
            config,
        )

        self.assertFalse(provisional.score_confirmed)
        self.assertTrue(reset.score_confirmed)
        self.assertEqual(reset.game_score, 1)
        self.assertEqual(config["press_model"]["stage_last_score"], 1)

    def test_new_game_can_be_confirmed_when_first_seen_at_score_one(self) -> None:
        config = fresh_config()
        score_40 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=40,
        )
        stage_press_context(score_40, config)
        begin_stage_session(config)

        provisional = stage_press_context(replace(score_40, game_score=1), config)
        confirmed = stage_press_context(
            replace(score_40, piece=(5, 5), target=(110, 0), game_score=2),
            config,
        )

        self.assertFalse(provisional.score_confirmed)
        self.assertTrue(confirmed.score_confirmed)
        self.assertEqual(confirmed.game_score, 2)

    def test_implausible_forward_score_requires_a_distinct_confirmation(self) -> None:
        config = fresh_config()
        score_100 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=100,
        )
        stage_press_context(score_100, config)

        rejected = stage_press_context(replace(score_100, game_score=999), config)
        recovered = stage_press_context(
            replace(score_100, piece=(5, 5), target=(110, 0), game_score=101),
            config,
        )

        self.assertFalse(rejected.score_confirmed)
        self.assertEqual(rejected.game_score, 100)
        self.assertTrue(recovered.score_confirmed)
        self.assertEqual(recovered.game_score, 101)

    def test_large_even_bonus_requires_a_distinct_confirmation(self) -> None:
        config = fresh_config()
        score_20 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=20,
            observation_id="score-20",
        )
        stage_press_context(score_20, config)

        provisional = stage_press_context(
            replace(score_20, game_score=50, observation_id="bonus-first"),
            config,
        )
        confirmed = stage_press_context(
            replace(score_20, game_score=50, observation_id="bonus-second"),
            config,
        )

        self.assertFalse(provisional.score_confirmed)
        self.assertEqual(provisional.game_score, 20)
        self.assertTrue(confirmed.score_confirmed)
        self.assertEqual(confirmed.game_score, 50)

    def test_forward_jump_above_limit_never_confirms(self) -> None:
        config = fresh_config()
        score_100 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=100,
            observation_id="score-100",
        )
        stage_press_context(score_100, config)

        first = stage_press_context(
            replace(score_100, game_score=999, observation_id="bad-first"),
            config,
        )
        second = stage_press_context(
            replace(score_100, game_score=999, observation_id="bad-second"),
            config,
        )

        self.assertFalse(first.score_confirmed)
        self.assertFalse(second.score_confirmed)
        self.assertEqual(second.game_score, 100)

    def test_missing_ocr_uses_piece_scale_instead_of_stale_base_score(self) -> None:
        config = fresh_config()
        base = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=0,
            game_score_confidence=0.9,
        )
        stage_press_context(base, config)
        shrunken = replace(
            base,
            piece_bbox=(-7, -28, 14, 28),
            game_score=None,
            game_score_confidence=None,
        )

        annotated = annotate_stage_context(shrunken, config)
        fallback = stage_press_context(annotated, config)

        self.assertTrue(fallback.score_confirmed)
        self.assertEqual(fallback.game_score, 0)
        self.assertTrue(fallback.bucket.startswith("scale:"))
        self.assertNotEqual(fallback.bucket, "scale:0")
        self.assertFalse(stage_feedback_updates_base_curve(fallback))

    def test_first_high_score_smoothly_takes_over_scale_fallback(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        missing = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        scale_context = stage_press_context(missing, config)
        scale_entry = next(
            item
            for item in model["stage_scales"]
            if item["stage_bucket"] == scale_context.bucket
        )
        scale_entry["press_scale"] = 1.4
        model["stage_last_multiplier"] = 1.4
        high = replace(missing, game_score=100, game_score_confidence=0.9)

        score_context = stage_press_context(high, config)

        self.assertEqual(score_context.bucket, "score:2")
        self.assertAlmostEqual(score_context.press_scale, 1.4)

    def test_first_base_score_reanchors_and_clears_cold_scale_state(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        missing = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        scale_context = stage_press_context(missing, config)
        scale_entry = next(
            item
            for item in model["stage_scales"]
            if item["stage_bucket"] == scale_context.bucket
        )
        scale_entry["press_scale"] = 1.4
        model["stage_last_multiplier"] = 1.4
        base = replace(missing, game_score=0, game_score_confidence=0.9)

        anchored = stage_press_context(base, config)
        missing_again = stage_press_context(missing, config)

        self.assertEqual(anchored.bucket, "score:base")
        self.assertAlmostEqual(anchored.press_scale, 1.0)
        self.assertAlmostEqual(missing_again.press_scale, 1.0)
        self.assertFalse(
            any(
                item["stage_bucket"] == scale_context.bucket
                and item.get("updates", 0) > 0
                for item in model["stage_scales"]
            )
        )

    def test_scale_bucket_does_not_raise_score_bucket(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["stage_scales"] = [
            {
                "stage_bucket": "scale:0",
                "stage_order": 0.0,
                "piece_scale_ratio": 1.0,
                "press_scale": 1.4,
                "updates": 1,
            }
        ]
        model["stage_last_multiplier"] = 1.4
        base = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=0,
        )
        stage_press_context(base, config)

        score_stage = stage_press_context(replace(base, game_score=10), config)

        self.assertEqual(score_stage.bucket, "score:0")
        self.assertAlmostEqual(score_stage.press_scale, 1.0)

    def test_begin_stage_session_preserves_learned_multiplier_and_segments(self) -> None:
        config = fresh_config()
        config["press_model"]["stage_scale_learning_rate"] = 1.0
        high = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=100,
        )
        updated = update_stage_press_scale(config, high, 200.0, 208.0)
        record_segment_center_correction(
            config,
            100.0,
            10.0,
            5.0,
            1.0,
            reference_press_ms=200.0,
            stage_bucket=updated.bucket,
        )

        begin_stage_session(config)
        restored = stage_press_context(high, config)

        self.assertAlmostEqual(restored.press_scale, updated.press_scale)
        self.assertTrue(config["press_model"]["segment_corrections"])

    def test_stage_multiplier_cannot_drop_below_earlier_score_stage(self) -> None:
        config = fresh_config()
        config["press_model"]["stage_scale_learning_rate"] = 1.0
        score_10 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=10,
        )
        earlier = update_stage_press_scale(config, score_10, 200.0, 208.0)
        score_55 = replace(score_10, game_score=55)

        later = update_stage_press_scale(config, score_55, 200.0, 180.0)

        self.assertGreaterEqual(later.press_scale, earlier.press_scale)

    def test_raising_earlier_stage_propagates_to_existing_later_stage(self) -> None:
        config = fresh_config()
        config["press_model"]["stage_scale_learning_rate"] = 1.0
        score_10 = replace(
            detection(piece=(0, 0), target=(100, 0), distance=100.0),
            game_score=10,
        )
        earlier = update_stage_press_scale(config, score_10, 200.0, 208.0)
        for score in (20, 30, 40, 50, 55):
            later_result = replace(score_10, game_score=score)
            stage_press_context(later_result, config)

        stage_press_context(replace(score_10, game_score=0), config)
        stage_press_context(
            replace(score_10, piece=(5, 5), target=(110, 0), game_score=1),
            config,
        )
        for score in (2, 4, 6, 8):
            stage_press_context(replace(score_10, game_score=score), config)
        raised = update_stage_press_scale(config, score_10, 200.0, 208.0)
        later_entry = next(
            item
            for item in config["press_model"]["stage_scales"]
            if item["stage_bucket"] == "score:1"
        )

        self.assertGreater(raised.press_scale, earlier.press_scale)
        self.assertGreaterEqual(later_entry["press_scale"], raised.press_scale)

    def test_missing_score_stage_does_not_train_global_base_curve(self) -> None:
        config = fresh_config()
        result = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        context = stage_press_context(result, config)

        self.assertEqual(context.bucket, "scale:0")
        self.assertFalse(stage_feedback_updates_base_curve(context))

    def test_base_curve_removes_stage_multiplier_from_feedback(self) -> None:
        sample = {
            "training_press_ms": 240.0,
            "stage_press_scale": 1.2,
        }

        self.assertAlmostEqual(sample_base_training_press_ms(sample), 200.0)

    def test_segment_corrections_are_isolated_by_score_stage(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        record_segment_center_correction(
            config,
            100.0,
            20.0,
            10.0,
            1.0,
            reference_press_ms=200.0,
            stage_bucket="score:1",
        )

        self.assertNotEqual(segment_correction_ms(100.0, model, "score:1"), 0.0)
        self.assertEqual(segment_correction_ms(100.0, model, "score:2"), 0.0)

    def test_failure_cap_only_applies_near_distance(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["failure_learning_enabled"] = True
        model = press_model_config(config)
        model["failure_caps"] = [
            {"distance_px": 100.0, "press_cap_ms": 180.0},
        ]

        self.assertIsNotNone(failure_press_cap_ms(112.0, model, config))
        self.assertIsNone(failure_press_cap_ms(250.0, model, config))

    def test_failure_cap_is_disabled_by_default(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["failure_caps"] = [{"distance_px": 420.0, "press_cap_ms": 240.0}]

        self.assertIsNone(failure_press_cap_ms(420.0, model, config))

    def test_segment_correction_is_written_to_matching_bin(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["slope_ms_per_px"] = 2.0

        record_segment_center_correction(config, 20.0, 12.0, 8.0, 1.0)

        entry = segment_correction_entry(model, 20.0)
        neighbor = segment_correction_entry(model, 28.0)
        self.assertIsNotNone(entry)
        self.assertIsNone(neighbor)
        self.assertGreater(segment_correction_ms(20.0, model), 0.0)

    def test_stale_segment_index_is_not_applied_under_new_bin_width(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["segment_corrections"] = [
            {
                "segment_index": 10,
                "distance_min_px": 70.0,
                "distance_max_px": 77.0,
                "correction_ms": 50.0,
            }
        ]

        self.assertEqual(segment_correction_ms(20.0, model), 0.0)
        record_segment_center_correction(config, 20.0, 8.0, 5.0, 1.0)
        self.assertGreater(segment_correction_ms(20.0, model), 0.0)

    def test_calibration_sample_contains_training_export_fields(self) -> None:
        result = detection(piece=(10, 20), target=(80, 40), distance=73.0, dx=70.0, dy=20.0)

        sample = calibration_sample_from_result(result, 180.0)

        self.assertEqual(sample["distance_px"], 73.0)
        self.assertEqual(sample["dx_px"], 70.0)
        self.assertEqual(sample["dy_px"], 20.0)
        self.assertEqual(sample["press_ms"], 180.0)
        self.assertIsNone(sample["landing_error_px"])
        self.assertEqual(sample["target"], [80, 40])
        self.assertEqual(sample["piece"], [10, 20])
        self.assertEqual(sample["confidence"], 0.9)
        self.assertEqual(sample["result_type"], "manual")
        self.assertNotIn("training_press_ms", sample)


if __name__ == "__main__":
    unittest.main()
