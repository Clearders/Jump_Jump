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
            self.assertEqual(
                migrated["press_model"]["segment_corrections"],
                legacy["press_model"]["segment_corrections"],
            )
            self.assertEqual(migrated["press_model"]["failure_caps"], legacy["press_model"]["failure_caps"])
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
