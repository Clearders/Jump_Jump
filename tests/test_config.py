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

    def test_legacy_migration_preserves_learned_and_unknown_data(self) -> None:
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
            self.assertEqual(migrated["press_model"]["curve_points"], legacy["press_model"]["curve_points"])
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
        self.assertEqual(migrated["auto_tuning"]["segment_precision_hits_to_freeze"], 1)
        self.assertEqual(migrated["auto_tuning"]["segment_unfreeze_error_px"], 18)

    def test_schema_three_keeps_only_corrections_matching_active_bin_size(self) -> None:
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

        corrections = migrated["press_model"]["segment_corrections"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["correction_ms"], 4.0)
        self.assertEqual(corrections[0]["segment_center_px"], 29.0)

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
        self.assertEqual(len(loaded["press_model"]["segment_corrections"]), 1)
        self.assertEqual(loaded["auto_tuning"]["segment_precision_px"], 6)
        self.assertEqual(loaded["auto_tuning"]["segment_precision_hits_to_freeze"], 2)
        self.assertEqual(loaded["auto_tuning"]["segment_unfreeze_error_px"], 20)

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

        bad_policy = fresh_config()
        bad_policy["debug"]["auto_capture_policy"] = "sometimes"
        cases.append(bad_policy)

        bad_platform_confidence = fresh_config()
        bad_platform_confidence["auto_tuning"]["landing_platform_min_confidence"] = 1.1
        cases.append(bad_platform_confidence)

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
