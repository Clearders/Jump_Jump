from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jumpjump.config import DEFAULT_CONFIG
from jumpjump.neural_press_model import (
    FEATURE_NAMES,
    NeuralPressPredictor,
    _split_samples,
    apply_safety_limits,
    build_coverage_bins,
    coverage_bin_for_sample,
    coverage_strength,
    eligible_supervised_samples,
    feature_vector,
    landing_comparison,
    model_passes_validation_gate,
    online_guard_decision,
    train_press_model,
)
from jumpjump.types import DetectionResult, JumpAutoError


def detection(distance: float = 100.0) -> DetectionResult:
    return DetectionResult(
        piece=(10, 100),
        target=(110, 50),
        piece_bbox=(0, 0, 20, 40),
        target_bbox=(80, 0, 50, 20),
        crop_rect=(0, 0, 500, 800),
        dx_px=distance,
        dy_px=-50.0,
        screen_distance_px=distance,
        effective_distance_px=distance,
        distance_px=distance,
        confidence=0.9,
        debug_path=Path("debug.png"),
    )


class NeuralPressModelTests(unittest.TestCase):
    def test_feature_vector_is_normalized_and_versioned(self) -> None:
        values = feature_vector(
            {
                "viewport_width_px": 500,
                "viewport_height_px": 1000,
                "dx_px": -250,
                "dy_px": -100,
                "effective_distance_px": 300,
                "piece_width_px": 20,
                "piece_height_px": 40,
                "target_width_px": 50,
                "target_height_px": 25,
                "confidence": 0.8,
                "legacy_press_ms": 600,
            }
        )
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(values[0], -0.5)
        self.assertEqual(values[-1], 0.6)

    def test_safety_limits_reapply_global_and_failure_caps(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["press_model"]["short_hop_enabled"] = False
        config["auto_tuning"]["failure_learning_enabled"] = True
        config["press_model"]["failure_caps"] = [
            {"distance_px": 100.0, "press_cap_ms": 240.0},
        ]
        self.assertEqual(apply_safety_limits(1000.0, detection(), config), 240.0)
        config["press_model"]["failure_caps"] = []
        self.assertEqual(apply_safety_limits(1.0, detection(), config), config["min_press_ms"])

    def test_landing_comparison_requires_30_pairs_and_checks_goal(self) -> None:
        insufficient = landing_comparison([], minimum_pairs=30)
        self.assertEqual(insufficient["status"], "insufficient_data")
        samples = []
        for index in range(30):
            common = {"dx_px": 100, "effective_distance_px": 200 + index / 10}
            samples.append(dict(common, prediction_source="legacy", landing_error_px=40))
            samples.append(dict(common, prediction_source="neural", landing_error_px=30))
        result = landing_comparison(samples)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["accepted"])

    def test_landing_comparison_isolates_model_versions(self) -> None:
        samples = []
        for index in range(30):
            common = {"dx_px": 100, "effective_distance_px": 200 + index / 10}
            samples.append(dict(common, prediction_source="legacy", landing_error_px=40))
            samples.append(
                dict(common, prediction_source="neural", prediction_model_id="old", landing_error_px=200)
            )
            samples.append(
                dict(common, prediction_source="neural", prediction_model_id="new", landing_error_px=30)
            )
        result = landing_comparison(samples, model_id="new")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["neural_median_landing_error_px"], 30)

    def test_validation_gate_requires_overall_gain_and_direction_safety(self) -> None:
        good = {
            "legacy_mae_ms": 100.0,
            "model_mae_ms": 85.0,
            "left_legacy_mae_ms": 100.0,
            "left_model_mae_ms": 90.0,
            "right_legacy_mae_ms": 100.0,
            "right_model_mae_ms": 80.0,
        }
        self.assertTrue(model_passes_validation_gate(good, 0.10, 0.10))
        bad_direction = dict(good, left_model_mae_ms=111.0)
        self.assertFalse(model_passes_validation_gate(bad_direction, 0.10, 0.10))
        no_gain = dict(good, model_mae_ms=95.0)
        self.assertFalse(model_passes_validation_gate(no_gain, 0.10, 0.10))

    def test_training_rejects_insufficient_samples_before_torch_is_needed(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        with self.assertRaises(JumpAutoError):
            train_press_model([], config, Path("unused.pt"), Path("unused.json"))

    def test_coverage_requires_direction_and_distance_support(self) -> None:
        samples = [
            {"dx_px": 100, "effective_distance_px": 250}
            for _ in range(12)
        ] + [
            {"dx_px": -100, "effective_distance_px": 250}
            for _ in range(5)
        ]
        bins = build_coverage_bins(samples, 100, 12)
        self.assertEqual([item["key"] for item in bins], ["right:2"])
        self.assertIsNotNone(coverage_bin_for_sample(samples[0], bins, 100))
        self.assertIsNone(coverage_bin_for_sample(samples[-1], bins, 100))
        self.assertEqual(coverage_strength(11, 12), 0.0)
        self.assertEqual(coverage_strength(12, 12), 0.25)

    def test_predictor_falls_back_before_inference_outside_coverage(self) -> None:
        predictor = NeuralPressPredictor(
            None,
            None,
            {"coverage_bin_size_px": 100, "coverage_bins": [{"key": "right:2", "samples": 12}]},
            None,
        )
        config = copy.deepcopy(DEFAULT_CONFIG)
        result = predictor.predict(detection(650), (500, 800), 700.0, config)
        self.assertEqual(result.source, "legacy")
        self.assertEqual(result.press_ms, 700.0)
        self.assertEqual(result.fallback_reason, "outside_neural_coverage")

    def test_supervised_training_excludes_neural_and_imported_feedback(self) -> None:
        base = {
            "schema_version": 2,
            "landing_label_method": "current_platform",
            "trainable": True,
            "viewport_width_px": 500,
            "viewport_height_px": 800,
            "dx_px": 100,
            "dy_px": -50,
            "effective_distance_px": 120,
            "confidence": 0.9,
            "legacy_press_ms": 300,
            "target_press_ms": 290,
            "prediction_source": "legacy",
        }
        neural = dict(base, prediction_source="neural")
        imported = dict(base, imported_from_config=True)
        self.assertEqual(eligible_supervised_samples([neural, imported, base]), [base])

    def test_session_split_preserves_coverage_in_both_sets(self) -> None:
        samples = []
        for session in range(5):
            for direction in (-1, 1):
                for distance in (320, 420):
                    for index in range(3):
                        samples.append(
                            {
                                "session_id": f"s{session}",
                                "sample_id": f"{session}-{direction}-{distance}-{index}",
                                "dx_px": direction * 100,
                                "effective_distance_px": distance,
                            }
                        )
        training, validation, strategy = _split_samples(samples, 12, 100, 2)
        self.assertTrue(strategy.startswith("session_grouped"))
        self.assertTrue({row["session_id"] for row in training}.isdisjoint(
            {row["session_id"] for row in validation}
        ))

    def test_validation_gate_rejects_bad_bucket_or_harmful_failures(self) -> None:
        overall = {
            "legacy_mae_ms": 100.0,
            "model_mae_ms": 80.0,
            "left_legacy_mae_ms": 100.0,
            "left_model_mae_ms": 80.0,
            "right_legacy_mae_ms": 100.0,
            "right_model_mae_ms": 80.0,
        }
        bad_bucket = {"right:5": {"legacy_mae_ms": 10.0, "model_mae_ms": 20.0}}
        self.assertFalse(model_passes_validation_gate(overall, 0.1, 0.1, bad_bucket))
        self.assertFalse(model_passes_validation_gate(overall, 0.1, 0.1, {}, 0.4, 0.25))

    def test_online_guard_disables_regressed_landing_performance(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        metadata = {"baseline_landing": {"median_error_px": 60.0, "success_rate": 0.70}}
        disabled, details = online_guard_decision([120.0] * 20, metadata, config)
        self.assertTrue(disabled)
        self.assertEqual(details["status"], "disable")
        collecting, details = online_guard_decision([20.0] * 5, metadata, config)
        self.assertFalse(collecting)
        self.assertEqual(details["status"], "collecting")


if __name__ == "__main__":
    unittest.main()
