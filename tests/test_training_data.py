from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jumpjump.training_data import (
    append_sample,
    import_legacy_samples,
    load_samples,
    valid_training_samples,
)


class TrainingDataTests(unittest.TestCase):
    def test_append_deduplicates_and_ignores_corrupt_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            sample = {
                "timestamp": "one",
                "session_id": "session",
                "dx_px": 10,
                "dy_px": -5,
                "executed_press_ms": 200,
                "result_type": "auto_precise",
            }
            self.assertTrue(append_sample(path, sample))
            self.assertFalse(append_sample(path, sample))
            with path.open("a", encoding="utf-8") as file:
                file.write('{"incomplete"')
            loaded = load_samples(path)
            self.assertEqual(len(loaded), 1)
            second = dict(sample, timestamp="two")
            self.assertTrue(append_sample(path, second))
            self.assertEqual(len(load_samples(path)), 2)

    def test_legacy_import_is_one_time_and_trainable(self) -> None:
        old = {
            "timestamp": "old",
            "dx_px": -100,
            "dy_px": -50,
            "press_ms": 300,
            "confidence": 0.9,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            self.assertEqual(import_legacy_samples(path, [old]), 1)
            self.assertEqual(import_legacy_samples(path, [old]), 0)
            self.assertEqual(len(valid_training_samples(load_samples(path))), 1)

    def test_invalid_training_rows_are_filtered(self) -> None:
        valid = {
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
        }
        invalid = dict(valid, target_press_ms=None)
        unlabelled = dict(valid, trainable=False)
        self.assertEqual(valid_training_samples([invalid, unlabelled, valid]), [valid])

    def test_legacy_automatic_labels_are_preserved_but_not_trainable(self) -> None:
        old_auto = {
            "schema_version": 1,
            "result_type": "auto_adjusted",
            "trainable": True,
            "viewport_width_px": 500,
            "viewport_height_px": 800,
            "dx_px": 100,
            "dy_px": -50,
            "effective_distance_px": 120,
            "confidence": 0.9,
            "legacy_press_ms": 300,
            "target_press_ms": 290,
        }
        old_manual = dict(old_auto, result_type="manual")

        self.assertEqual(valid_training_samples([old_auto, old_manual]), [old_manual])


if __name__ == "__main__":
    unittest.main()
