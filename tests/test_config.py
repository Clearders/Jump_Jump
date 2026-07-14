from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from jumpjump.config import (
    CURRENT_AUTO_FEEDBACK_VERSION,
    CURRENT_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    deep_merge,
    load_config,
    save_config,
    validate_config,
)
from jumpjump.types import ConfigError


def fresh_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


class ConfigReliabilityTests(unittest.TestCase):
    def test_dynamic_piece_shape_bounds_are_validated(self) -> None:
        config = fresh_config()
        config["piece"]["dynamic_shape_min_scale_ratio"] = 1.5
        config["piece"]["dynamic_shape_max_scale_ratio"] = 1.0

        with self.assertRaisesRegex(ConfigError, "minimum scale"):
            validate_config(config)

    def test_deep_merge_does_not_alias_defaults_or_override(self) -> None:
        override = {
            "target": {"diff_threshold": 17},
            "press_model": {"samples": [{"press_ms": 200.0}]},
        }

        merged = deep_merge(DEFAULT_CONFIG, override)
        merged["target"]["diff_threshold"] = 99
        merged["press_model"]["samples"][0]["press_ms"] = 999.0

        self.assertEqual(DEFAULT_CONFIG["target"]["diff_threshold"], 14)
        self.assertEqual(override["press_model"]["samples"][0]["press_ms"], 200.0)

    def test_legacy_migration_preserves_samples_and_unknown_data(self) -> None:
        legacy = {
            "press_model": {
                "samples": [{"distance_px": 100.0, "press_ms": 200.0}],
                "curve_points": [{"distance_px": 100.0, "press_ms": 200.0}],
                "segment_corrections": [{"segment_index": 14, "correction_ms": 3.0}],
                "failure_caps": [{"distance_px": 100.0, "press_cap_ms": 190.0}],
                "future_model_field": {"preserve": True},
            },
            "piece": {"color_samples": [{"h": 120.0, "s": 80.0, "v": 90.0}]},
            "unknown_root": ["keep", 1],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = load_config(path)
            self.assertEqual(migrated["schema_version"], CURRENT_SCHEMA_VERSION)
            self.assertEqual(migrated["press_model"]["samples"], legacy["press_model"]["samples"])
            self.assertEqual(migrated["press_model"]["curve_points"], [])
            self.assertEqual(migrated["press_model"]["x_weight"], 1.0)
            self.assertEqual(migrated["press_model"]["y_weight"], 1.0)
            self.assertIsNone(migrated["press_model"]["slope_ms_per_px"])
            self.assertEqual(migrated["press_model"]["offset_ms"], 0.0)
            self.assertIsNone(migrated["press_model"]["fit_rmse_ms"])
            self.assertIsNone(migrated["press_ms_per_px"])
            self.assertEqual(migrated["press_model"]["segment_corrections"], [])
            self.assertEqual(migrated["press_model"]["failure_caps"], [])
            self.assertFalse(migrated["auto_tuning"]["failure_learning_enabled"])
            self.assertEqual(migrated["piece"]["color_samples"], legacy["piece"]["color_samples"])
            self.assertEqual(migrated["unknown_root"], legacy["unknown_root"])
            self.assertEqual(
                migrated["press_model"]["future_model_field"],
                legacy["press_model"]["future_model_field"],
            )

            save_config(path, migrated)
            reloaded = load_config(path)
            self.assertEqual(reloaded["press_model"]["samples"], legacy["press_model"]["samples"])
            self.assertEqual(reloaded["unknown_root"], legacy["unknown_root"])

    def test_schema_three_migrates_segment_defaults_and_drops_old_bins(self) -> None:
        legacy = {
            "schema_version": 3,
            "press_model": {
                "segment_size_px": 7,
                "max_segment_corrections": 120,
                "segment_corrections": [
                    {
                        "segment_index": 14,
                        "distance_min_px": 98.0,
                        "distance_max_px": 105.0,
                        "segment_center_px": 101.5,
                        "correction_ms": 3.0,
                    }
                ],
            },
            "auto_tuning": {
                "segment_precision_px": 8,
                "segment_precision_hits_to_freeze": 3,
                "segment_unfreeze_error_px": 8,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = load_config(path)

        self.assertEqual(migrated["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(migrated["press_model"]["segment_size_px"], 2)
        self.assertEqual(migrated["press_model"]["max_segment_corrections"], 300)
        self.assertEqual(migrated["press_model"]["segment_corrections"], [])
        self.assertEqual(migrated["auto_tuning"]["segment_precision_px"], 3)
        self.assertEqual(migrated["auto_tuning"]["segment_precision_hits_to_freeze"], 3)
        self.assertEqual(migrated["auto_tuning"]["segment_unfreeze_error_px"], 18)

    def test_schema_three_drops_corrections_with_obsolete_feedback_semantics(self) -> None:
        legacy = {
            "schema_version": 3,
            "press_model": {
                "segment_size_px": 2,
                "segment_corrections": [
                    {
                        "segment_index": 14,
                        "distance_min_px": 98.0,
                        "distance_max_px": 105.0,
                        "correction_ms": 30.0,
                    },
                    {
                        "segment_index": 14,
                        "distance_min_px": 28.0,
                        "distance_max_px": 30.0,
                        "correction_ms": 4.0,
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = load_config(path)

        self.assertEqual(migrated["press_model"]["segment_corrections"], [])

    def test_current_schema_drops_corrections_after_manual_bin_size_change(self) -> None:
        edited = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "press_model": {
                "segment_size_px": 5,
                "segment_corrections": [
                    {
                        "segment_index": 14,
                        "distance_min_px": 28.0,
                        "distance_max_px": 30.0,
                        "correction_ms": 4.0,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(edited), encoding="utf-8")

            loaded = load_config(path)

        self.assertEqual(loaded["press_model"]["segment_size_px"], 5)
        self.assertEqual(loaded["press_model"]["segment_corrections"], [])

    def test_schema_three_preserves_explicit_custom_tuning_values(self) -> None:
        custom = {
            "schema_version": 3,
            "press_model": {
                "segment_size_px": 5,
                "max_segment_corrections": 150,
                "segment_corrections": [
                    {
                        "segment_index": 14,
                        "distance_min_px": 70.0,
                        "distance_max_px": 75.0,
                        "correction_ms": 4.0,
                    }
                ],
            },
            "auto_tuning": {
                "segment_precision_px": 6,
                "segment_precision_hits_to_freeze": 2,
                "segment_unfreeze_error_px": 20,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(custom), encoding="utf-8")

            loaded = load_config(path)

        self.assertEqual(loaded["press_model"]["segment_size_px"], 5)
        self.assertEqual(loaded["press_model"]["max_segment_corrections"], 150)
        self.assertEqual(loaded["press_model"]["segment_corrections"], [])
        self.assertEqual(loaded["auto_tuning"]["segment_precision_px"], 6)
        self.assertEqual(loaded["auto_tuning"]["segment_precision_hits_to_freeze"], 2)
        self.assertEqual(loaded["auto_tuning"]["segment_unfreeze_error_px"], 20)

    def test_schema_four_quarantines_unsafe_feedback_state(self) -> None:
        legacy = {
            "schema_version": 4,
            "press_ms_per_px": 9.0,
            "press_model": {
                "samples": [
                    {
                        "source": "auto_segment_adjusted",
                        "result_type": "auto_adjusted",
                        "landing_error_px": 250.0,
                        "press_ms": 300.0,
                    },
                    {
                        "source": "auto_segment_adjusted",
                        "result_type": "auto_adjusted",
                        "landing_error_px": 40.0,
                        "press_ms": 280.0,
                    },
                    {
                        "source": "auto_segment_adjusted",
                        "result_type": "auto_adjusted",
                        "landing_error_px": "nan",
                        "press_ms": 290.0,
                    },
                    {
                        "source": "auto_segment_adjusted",
                        "result_type": "auto_adjusted",
                        "landing_error_px": None,
                        "press_ms": 295.0,
                    },
                    {
                        "source": "manual",
                        "result_type": "manual",
                        "landing_error_px": None,
                        "press_ms": 275.0,
                    },
                ],
                "curve_points": [{"distance_px": 100.0, "press_ms": 999.0}],
                "sample_count": 5,
                "type": "weighted_piecewise",
                "slope_ms_per_px": 9.0,
                "offset_ms": 300.0,
                "fit_rmse_ms": 1.0,
                "segment_corrections": [
                    {
                        "segment_index": 50,
                        "distance_min_px": 100.0,
                        "distance_max_px": 102.0,
                        "correction_ms": 30.0,
                    }
                ],
            },
            "auto_tuning": {
                "landing_tolerance_px": 500,
                "segment_precision_hits_to_freeze": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            loaded = load_config(path)

        self.assertEqual(loaded["auto_tuning"]["landing_tolerance_px"], 80)
        self.assertEqual(loaded["auto_tuning"]["segment_precision_hits_to_freeze"], 3)
        self.assertEqual(len(loaded["press_model"]["samples"]), 1)
        self.assertEqual(loaded["press_model"]["samples"][0]["result_type"], "manual")
        self.assertEqual(loaded["press_model"]["curve_points"], [])
        self.assertEqual(loaded["press_model"]["segment_corrections"], [])
        self.assertEqual(loaded["press_model"]["type"], "weighted_euclidean")
        self.assertIsNone(loaded["press_model"]["slope_ms_per_px"])
        self.assertEqual(loaded["press_model"]["offset_ms"], 0.0)
        self.assertIsNone(loaded["press_model"]["fit_rmse_ms"])
        self.assertEqual(loaded["press_model"]["x_weight"], 1.0)
        self.assertEqual(loaded["press_model"]["y_weight"], 1.0)
        self.assertIsNone(loaded["press_ms_per_px"])

    def test_current_schema_keeps_only_versioned_automatic_feedback(self) -> None:
        current = fresh_config()
        current["press_model"]["samples"] = [
            {
                "source": "auto_segment_adjusted",
                "result_type": "auto_adjusted",
                "landing_error_px": 20.0,
                "press_ms": 250.0,
            },
            {
                "source": "auto_segment_adjusted",
                "result_type": "auto_adjusted",
                "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
                "landing_error_px": 20.0,
                "press_ms": 245.0,
            },
            {
                "source": "manual",
                "result_type": "manual",
                "press_ms": 240.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(current), encoding="utf-8")

            loaded = load_config(path)

        self.assertEqual(len(loaded["press_model"]["samples"]), 2)
        self.assertEqual(
            loaded["press_model"]["samples"][0]["feedback_version"],
            CURRENT_AUTO_FEEDBACK_VERSION,
        )
        self.assertEqual(loaded["press_model"]["samples"][1]["result_type"], "manual")

    def test_schema_four_resets_stale_fit_even_without_auto_samples(self) -> None:
        legacy = {
            "schema_version": 4,
            "press_ms_per_px": 9.0,
            "press_model": {
                "samples": [],
                "curve_points": [{"distance_px": 100.0, "press_ms": 900.0}],
                "sample_count": 10,
                "type": "weighted_piecewise",
                "x_weight": 1.0,
                "y_weight": 1.5,
                "slope_ms_per_px": 9.0,
                "offset_ms": 300.0,
                "fit_rmse_ms": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            loaded = load_config(path)

        model = loaded["press_model"]
        self.assertEqual(model["curve_points"], [])
        self.assertIsNone(model["slope_ms_per_px"])
        self.assertEqual(model["y_weight"], 1.0)
        self.assertIsNone(loaded["press_ms_per_px"])

    def test_schema_five_archives_stage_blind_state_and_clears_unsafe_caps(self) -> None:
        legacy = fresh_config()
        legacy["schema_version"] = 5
        legacy["press_model"]["physics_press_coefficient"] = 1.45
        legacy["press_model"]["samples"] = [
            {
                "source": "auto_segment_adjusted",
                "result_type": "auto_adjusted",
                "feedback_version": 2,
                "landing_error_px": 12.0,
                "press_ms": 280.0,
            },
            {
                "source": "manual",
                "result_type": "manual",
                "press_ms": 260.0,
                "dx_px": 100.0,
                "dy_px": 0.0,
            },
        ]
        legacy["press_model"]["curve_points"] = [
            {"distance_px": 100.0, "press_ms": 260.0}
        ]
        legacy["press_model"]["segment_corrections"] = [
            {
                "segment_index": 50,
                "distance_min_px": 100.0,
                "distance_max_px": 102.0,
                "correction_ms": 10.0,
            }
        ]
        legacy["press_model"]["failure_caps"] = [
            {"distance_px": 100.0, "press_cap_ms": 180.0}
        ]
        legacy["auto_tuning"]["failure_learning_enabled"] = True
        legacy["neural_press_model"]["training_metrics"] = {"feature_version": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            loaded = load_config(path)

        model = loaded["press_model"]
        archive = model["legacy_feedback_archive"]
        self.assertEqual(loaded["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(model["physics_press_coefficient"], 1.45)
        self.assertEqual(len(model["samples"]), 1)
        self.assertEqual(model["samples"][0]["result_type"], "manual")
        self.assertEqual(len(archive["samples"]), 2)
        self.assertEqual(len(archive["curve_points"]), 1)
        self.assertEqual(len(archive["segment_corrections"]), 1)
        self.assertEqual(len(archive["failure_caps"]), 1)
        self.assertEqual(model["curve_points"], [])
        self.assertEqual(model["segment_corrections"], [])
        self.assertEqual(model["failure_caps"], [])
        self.assertFalse(loaded["auto_tuning"]["failure_learning_enabled"])
        self.assertEqual(loaded["neural_press_model"]["training_metrics"], {})

    def test_current_schema_quarantines_negative_and_non_object_feedback(self) -> None:
        current = fresh_config()
        current["press_model"]["samples"] = [
            123,
            {
                "source": "auto_segment_adjusted",
                "result_type": "auto_adjusted",
                "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
                "landing_error_px": -1.0,
                "press_ms": 245.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(current), encoding="utf-8")
            loaded = load_config(path)

        self.assertEqual(loaded["press_model"]["samples"], [])
        self.assertEqual(
            len(loaded["press_model"]["quarantined_feedback_samples"]),
            2,
        )

    def test_save_config_migrates_schema_three_before_stamping_version(self) -> None:
        old_config = {
            "schema_version": 3,
            "press_model": {
                "segment_size_px": 7,
                "segment_corrections": [
                    {
                        "segment_index": 14,
                        "distance_min_px": 98.0,
                        "distance_max_px": 105.0,
                        "correction_ms": 4.0,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"

            save_config(path, old_config)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(saved["press_model"]["segment_size_px"], 2)
        self.assertEqual(saved["press_model"]["segment_corrections"], [])

    def test_schema_two_migration_repairs_unsafe_press_parameters(self) -> None:
        legacy = {
            "schema_version": 2,
            "press_model": {
                "physics_piece_width_multiplier": 1.6,
                "failure_caps": [{"distance_px": 420.0, "press_cap_ms": 240.0}],
            },
            "auto_tuning": {"failure_learning_enabled": True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = load_config(path)

        self.assertEqual(migrated["schema_version"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(migrated["press_model"]["physics_piece_width_multiplier"], 1.15)
        self.assertEqual(migrated["press_model"]["failure_caps"], [])
        self.assertFalse(migrated["auto_tuning"]["failure_learning_enabled"])

    def test_first_save_creates_main_and_recoverable_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            config = fresh_config()
            config["window_title"] = "saved"

            save_config(path, config)

            backup = path.with_name(f"{path.name}.bak")
            self.assertTrue(path.is_file())
            self.assertTrue(backup.is_file())
            self.assertEqual(load_config(path)["window_title"], "saved")
            self.assertEqual(load_config(backup)["window_title"], "saved")

    def test_invalid_primary_recovers_then_can_be_repaired_without_losing_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            backup = path.with_name(f"{path.name}.bak")
            config = fresh_config()
            config["window_title"] = "backup-value"
            save_config(path, config)
            path.write_text("{not valid json", encoding="utf-8")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = load_config(path)
            self.assertTrue(caught)
            self.assertEqual(recovered["window_title"], "backup-value")

            recovered["window_title"] = "repaired-main"
            save_config(path, recovered)

            self.assertEqual(load_config(path)["window_title"], "repaired-main")
            self.assertEqual(load_config(backup)["window_title"], "backup-value")

    def test_failed_final_replace_leaves_main_and_backup_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            backup = path.with_name(f"{path.name}.bak")
            initial = fresh_config()
            initial["window_title"] = "old"
            save_config(path, initial)
            updated = fresh_config()
            updated["window_title"] = "new"
            real_replace = os.replace

            def fail_main_replace(source, destination):
                if Path(destination) == path:
                    raise OSError("simulated replace failure")
                return real_replace(source, destination)

            with patch("jumpjump.config.os.replace", side_effect=fail_main_replace):
                with self.assertRaises(ConfigError):
                    save_config(path, updated)

            self.assertEqual(load_config(path)["window_title"], "old")
            self.assertEqual(load_config(backup)["window_title"], "old")
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])
            self.assertEqual(list(Path(tmpdir).glob(".*.tmp")), [])

    def test_validation_rejects_unsafe_or_malformed_values(self) -> None:
        cases = []

        nan_confidence = fresh_config()
        nan_confidence["confidence_threshold"] = float("nan")
        cases.append(nan_confidence)

        infinite_ratio = fresh_config()
        infinite_ratio["click_point"]["x_ratio"] = float("inf")
        cases.append(infinite_ratio)

        overflowing_confidence = fresh_config()
        overflowing_confidence["confidence_threshold"] = 10**1000
        cases.append(overflowing_confidence)

        reversed_press_bounds = fresh_config()
        reversed_press_bounds["min_press_ms"] = 900
        reversed_press_bounds["max_press_ms"] = 400
        cases.append(reversed_press_bounds)

        invalid_short_hop_floor = fresh_config()
        invalid_short_hop_floor["press_model"]["short_hop_min_press_ms"] = 181
        cases.append(invalid_short_hop_floor)

        bad_policy = fresh_config()
        bad_policy["debug"]["auto_capture_policy"] = "sometimes"
        cases.append(bad_policy)

        bad_platform_confidence = fresh_config()
        bad_platform_confidence["auto_tuning"]["landing_platform_min_confidence"] = 1.1
        cases.append(bad_platform_confidence)

        unsafe_landing_tolerance = fresh_config()
        unsafe_landing_tolerance["auto_tuning"]["landing_tolerance_px"] = 81
        cases.append(unsafe_landing_tolerance)

        invalid_precision_hits = fresh_config()
        invalid_precision_hits["auto_tuning"]["segment_precision_hits_to_freeze"] = 0
        cases.append(invalid_precision_hits)

        invalid_score_step = fresh_config()
        invalid_score_step["score"]["max_forward_step"] = 0
        cases.append(invalid_score_step)

        invalid_temporal_horizontal_ratio = fresh_config()
        invalid_temporal_horizontal_ratio["auto_tuning"][
            "temporal_min_horizontal_ratio"
        ] = 1.1
        cases.append(invalid_temporal_horizontal_ratio)

        wrong_section_type = fresh_config()
        wrong_section_type["target"] = []
        cases.append(wrong_section_type)

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ConfigError):
                    validate_config(config)

        with self.assertRaises(ConfigError):
            validate_config({})

    def test_load_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jump_config.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
