from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jumpjump.training_data as training_data
from jumpjump.config import CURRENT_AUTO_FEEDBACK_VERSION
from jumpjump.training_data import (
    CURRENT_LANDING_LABEL_METHOD,
    DATASET_SCHEMA_VERSION,
    SampleIdIndex,
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

    def test_standalone_append_keeps_uncached_duplicate_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            with patch(
                "jumpjump.training_data._load_sample_ids",
                wraps=training_data._load_sample_ids,
            ) as load_ids:
                self.assertTrue(append_sample(path, {"sample_id": "one"}))
                self.assertTrue(append_sample(path, {"sample_id": "two"}))

            self.assertEqual(load_ids.call_count, 2)

    def test_session_index_scans_once_for_repeated_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            with patch(
                "jumpjump.training_data._load_sample_ids",
                wraps=training_data._load_sample_ids,
            ) as load_ids:
                index = SampleIdIndex(path)
                self.assertTrue(index.append({"sample_id": "one"}))
                self.assertFalse(index.append({"sample_id": "one"}))
                self.assertTrue(index.append({"sample_id": "two"}))

            self.assertEqual(load_ids.call_count, 1)
            self.assertEqual(
                [sample["sample_id"] for sample in load_samples(path)],
                ["one", "two"],
            )

    def test_session_index_refreshes_after_external_append_and_corrupt_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            index = SampleIdIndex(path)
            self.assertTrue(append_sample(path, {"sample_id": "external"}))
            self.assertFalse(index.append({"sample_id": "external"}))

            with path.open("a", encoding="utf-8") as file:
                file.write('{"incomplete"')
            self.assertTrue(index.append({"sample_id": "after-tail"}))
            self.assertEqual(
                [sample["sample_id"] for sample in load_samples(path)],
                ["external", "after-tail"],
            )

    def test_session_index_updates_only_after_fsync_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            index = SampleIdIndex(path)
            with patch(
                "jumpjump.training_data.os.fsync",
                side_effect=OSError("simulated fsync failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated fsync failure"):
                    index.append({"sample_id": "uncertain"})

            self.assertNotIn("uncertain", index.sample_ids)
            # The flushed row is rediscovered from disk rather than trusted from
            # the failed append attempt.
            self.assertFalse(index.append({"sample_id": "uncertain"}))
            self.assertIn("uncertain", index.sample_ids)

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

    def test_legacy_import_scans_existing_ids_once_per_batch(self) -> None:
        old_samples = [
            {
                "timestamp": f"old-{index}",
                "dx_px": -100 - index,
                "dy_px": -50,
                "press_ms": 300 + index,
                "confidence": 0.9,
            }
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            with patch(
                "jumpjump.training_data._load_sample_ids",
                wraps=training_data._load_sample_ids,
            ) as load_ids:
                self.assertEqual(import_legacy_samples(path, []), 0)
                self.assertEqual(load_ids.call_count, 0)
                self.assertEqual(import_legacy_samples(path, old_samples), 20)

            self.assertEqual(load_ids.call_count, 1)
            self.assertEqual(len(load_samples(path)), 20)

    def test_invalid_training_rows_are_filtered(self) -> None:
        valid = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
            "landing_label_method": CURRENT_LANDING_LABEL_METHOD,
            "trainable": True,
            "viewport_width_px": 500,
            "viewport_height_px": 800,
            "dx_px": 100,
            "dy_px": -50,
            "effective_distance_px": 120,
            "confidence": 0.9,
            "legacy_press_ms": 300,
            "target_press_ms": 290,
            "landing_error_px": 20.0,
            "piece_scale_ratio": 1.0,
            "stage_bucket": "scale:0",
            "stage_press_scale": 1.0,
            "stage_score_confirmed": True,
            "physics_unit_press_ms": 200.0,
            "effective_press_coefficient": 1.5,
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

    def test_automatic_rows_above_safe_landing_tolerance_are_quarantined(self) -> None:
        base = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
            "result_type": "auto_adjusted",
            "landing_label_method": CURRENT_LANDING_LABEL_METHOD,
            "trainable": True,
            "viewport_width_px": 500,
            "viewport_height_px": 800,
            "dx_px": 100,
            "dy_px": -50,
            "effective_distance_px": 120,
            "confidence": 0.9,
            "legacy_press_ms": 300,
            "target_press_ms": 290,
            "piece_scale_ratio": 1.0,
            "stage_bucket": "score:1",
            "stage_press_scale": 1.1,
            "stage_score_confirmed": True,
            "physics_unit_press_ms": 200.0,
            "effective_press_coefficient": 1.5,
        }
        trusted = dict(base, landing_error_px=40.0)
        poisoned = dict(base, landing_error_px=250.0)
        missing = dict(base, landing_error_px=None)
        legacy_semantics = dict(
            trusted,
            schema_version=2,
            landing_label_method="current_platform",
        )

        self.assertEqual(
            valid_training_samples([poisoned, missing, legacy_semantics, trusted]),
            [trusted],
        )

    def test_negative_landing_error_is_quarantined(self) -> None:
        sample = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "feedback_version": CURRENT_AUTO_FEEDBACK_VERSION,
            "result_type": "auto_adjusted",
            "landing_label_method": CURRENT_LANDING_LABEL_METHOD,
            "landing_error_px": -1.0,
            "trainable": True,
            "viewport_width_px": 500,
            "viewport_height_px": 800,
            "dx_px": 100,
            "dy_px": -50,
            "effective_distance_px": 120,
            "confidence": 0.9,
            "legacy_press_ms": 300,
            "target_press_ms": 290,
            "piece_scale_ratio": 1.0,
            "stage_bucket": "score:1",
            "stage_press_scale": 1.1,
            "stage_score_confirmed": True,
            "physics_unit_press_ms": 200.0,
            "effective_press_coefficient": 1.5,
        }

        self.assertEqual(valid_training_samples([sample]), [])

    def test_legacy_import_ignores_non_object_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            self.assertEqual(import_legacy_samples(path, [123, None, "bad"]), 0)
            self.assertEqual(load_samples(path), [])


if __name__ == "__main__":
    unittest.main()
