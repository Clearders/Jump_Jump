from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jumpjump.config import DEFAULT_CONFIG, press_model_config
from jumpjump.press_model import (
    calibration_sample_from_result,
    calculate_press_ms,
    center_adjusted_press_ms,
    failure_press_cap_ms,
    fit_press_model,
    linear_reference_press_ms,
    mark_segment_precision_hit,
    measure_landing,
    maybe_unfreeze_segment_for_error,
    physics_reference_press_ms,
    piecewise_press_ms,
    record_segment_center_correction,
    sample_training_press_ms,
    segment_bounds_for_distance,
    segment_correction_entry,
    segment_correction_ms,
    segment_is_frozen,
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
        piece_bbox=(0, 0, 20, 40),
        target_bbox=(80, 0, 40, 20),
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
            },
            {
                "dx_px": 200.0,
                "dy_px": 0.0,
                "press_ms": 2000.0,
                "training_press_ms": 200.0,
                "confidence": 0.9,
            },
            {
                "dx_px": 300.0,
                "dy_px": 0.0,
                "press_ms": 3000.0,
                "training_press_ms": 300.0,
                "confidence": 0.9,
            },
        ]

        fit_press_model(config)

        self.assertAlmostEqual(model["slope_ms_per_px"], 1.0)
        self.assertEqual(
            [point["press_ms"] for point in model["curve_points"]],
            [100.0, 200.0, 300.0],
        )

    def test_segment_bounds_use_seven_pixel_bins(self) -> None:
        config = fresh_config()
        model = press_model_config(config)

        segment_index, distance_min, distance_max, center = segment_bounds_for_distance(
            14.2,
            model,
        )

        self.assertEqual(segment_index, 2)
        self.assertEqual(distance_min, 14.0)
        self.assertEqual(distance_max, 21.0)
        self.assertEqual(center, 17.5)

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

    def test_center_adjustment_uses_nonlinear_curve_delta(self) -> None:
        config = fresh_config()
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
        self.assertLess(adjusted_press, 200.0)
        self.assertEqual(signed_error, 12.0)
        self.assertEqual(projection_ratio, 1.0)

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

    def test_low_projection_measurement_is_retained(self) -> None:
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
        self.assertEqual(measurement.signed_error_px, 0.0)
        self.assertEqual(measurement.projection_ratio, 0.0)

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
