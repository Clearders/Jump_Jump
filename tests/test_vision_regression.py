from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from jumpjump.config import DEFAULT_CONFIG
from jumpjump.types import RecognitionError
from jumpjump.vision import (
    collect_target_candidates,
    detect_jump,
    find_landing_platform,
    screen_overlay_present,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def opencv_modules():
    return cv2, np


def synthetic_scene(width: int = 400, height: int = 300):
    cv2, np = opencv_modules()
    if cv2 is None or np is None:
        return None, None, None
    crop = np.full((height, width, 3), (214, 224, 232), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    return cv2, crop, mask


class VisionRegressionTests(unittest.TestCase):
    def test_landing_platform_is_found_under_partially_occluding_piece(self) -> None:
        cv2, crop, _ = synthetic_scene(width=420, height=320)
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (210, 220)
        piece_bbox = (190, 145, 40, 82)
        crop[205:275, 105:315] = (105, 110, 115)
        crop[145:227, 190:230] = (58, 42, 78)

        result = find_landing_platform(crop, piece, piece_bbox, config)

        self.assertIsNotNone(result)
        point, bbox, confidence = result
        self.assertLessEqual(abs(point[1] - piece[1]), 20)
        self.assertLessEqual(bbox[0], piece[0])
        self.assertGreaterEqual(bbox[0] + bbox[2], piece[0])
        self.assertGreaterEqual(confidence, 0.55)

    def test_annotated_platform_fixture_detects_expected_points(self) -> None:
        sample_path = FIXTURE_DIR / "dry_run_20260710_003422_692036.png"
        self.assertTrue(sample_path.is_file(), f"missing tracked fixture: {sample_path}")
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (1614, 869, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_jump(
                frame,
                copy.deepcopy(DEFAULT_CONFIG),
                Path(tmpdir),
                "vision_regression_platform_right",
            )

        self.assertLessEqual(math.dist(result.piece, (271, 968)), 8.0)
        self.assertLessEqual(math.dist(result.target, (621, 749)), 12.0)
        self.assertGreaterEqual(result.confidence, 0.45)
        self.assertGreater(result.target[0], result.piece[0])
        self.assertIsNotNone(result.landing_platform)
        self.assertIsNotNone(result.landing_platform_bbox)
        self.assertGreaterEqual(result.landing_platform_confidence, 0.55)
        self.assertLessEqual(abs(result.landing_platform[1] - result.piece[1]), 12)

    def test_game_over_fixture_is_rejected_as_overlay(self) -> None:
        sample_path = FIXTURE_DIR / "auto_0033_failed_20260710_015402_000166.png"
        self.assertTrue(sample_path.is_file(), f"missing tracked fixture: {sample_path}")
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (1614, 869, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RecognitionError, "game-over or modal overlay"):
                detect_jump(
                    frame,
                    copy.deepcopy(DEFAULT_CONFIG),
                    Path(tmpdir),
                    "vision_regression_overlay",
                    save_debug=False,
                )
            self.assertEqual(len(list(Path(tmpdir).glob("*_failed_*.png"))), 1)

    def test_sparse_dark_platform_is_not_treated_as_overlay(self) -> None:
        frame = np.full((1000, 800, 3), (245, 245, 220), dtype=np.uint8)
        cv2.rectangle(frame, (100, 400), (280, 580), (55, 55, 55), -1)
        cv2.rectangle(frame, (520, 650), (700, 830), (55, 55, 55), -1)
        cv2.line(frame, (250, 550), (550, 680), (55, 55, 55), 10)
        config = copy.deepcopy(DEFAULT_CONFIG)

        self.assertFalse(screen_overlay_present(frame, config))

    def test_success_debug_can_be_disabled(self) -> None:
        sample_path = FIXTURE_DIR / "dry_run_20260710_003422_692036.png"
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_jump(
                frame,
                copy.deepcopy(DEFAULT_CONFIG),
                Path(tmpdir),
                "vision_regression_no_debug",
                save_debug=False,
            )
            self.assertIsNone(result.debug_path)
            self.assertEqual(list(Path(tmpdir).glob("*.png")), [])

    def test_tight_retention_keeps_returned_debug_path_when_masks_are_saved(self) -> None:
        sample_path = FIXTURE_DIR / "dry_run_20260710_003422_692036.png"
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["debug"]["max_files"] = 1
        config["debug"]["max_size_mb"] = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_jump(
                frame,
                config,
                Path(tmpdir),
                "vision_regression_tight_retention",
                save_debug=True,
                save_mask=True,
            )
            self.assertIsNotNone(result.debug_path)
            self.assertTrue(result.debug_path.is_file())
            self.assertEqual(list(Path(tmpdir).glob("*.png")), [result.debug_path])

    def test_current_platform_side_candidate_is_low_confidence(self) -> None:
        cv2, crop, mask = synthetic_scene()
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (250, 220)
        piece_bbox = (230, 165, 40, 70)
        crop[190:246, 165:228] = (92, 96, 98)
        cv2.rectangle(mask, (165, 198), (225, 238), 255, -1)

        candidates = collect_target_candidates(
            crop,
            mask,
            piece,
            piece_bbox,
            config,
            confidence_scale=1.0,
            source="test",
        )

        self.assertTrue(candidates)
        best = max(candidates, key=lambda candidate: candidate.confidence)
        self.assertLess(best.confidence, config["auto_tuning"]["run_confidence_floor"])
        self.assertTrue(set(best.risks) & {"current_platform", "current_platform_band"})

    def test_far_edge_large_surface_focuses_target_point(self) -> None:
        cv2, crop, mask = synthetic_scene(width=500, height=400)
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (170, 300)
        piece_bbox = (145, 235, 50, 80)
        crop[90:220, 250:500] = (245, 245, 245)
        cv2.rectangle(mask, (200, 90), (499, 220), 255, -1)

        candidates = collect_target_candidates(
            crop,
            mask,
            piece,
            piece_bbox,
            config,
            confidence_scale=1.0,
            source="test",
        )

        self.assertTrue(candidates)
        best = max(candidates, key=lambda candidate: candidate.score)
        self.assertGreater(best.point[0], 330)
        self.assertGreaterEqual(best.bbox[0], 250)

    def test_lower_target_is_not_hard_rejected(self) -> None:
        cv2, crop, mask = synthetic_scene()
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (140, 130)
        piece_bbox = (120, 80, 40, 70)
        crop[215:285, 240:330] = (120, 124, 126)
        cv2.rectangle(mask, (240, 215), (330, 285), 255, -1)

        candidates = collect_target_candidates(
            crop,
            mask,
            piece,
            piece_bbox,
            config,
            confidence_scale=1.0,
            source="test",
        )

        self.assertTrue(candidates)
        best = max(candidates, key=lambda candidate: candidate.score)
        self.assertGreater(best.point[1], piece[1])
        self.assertGreaterEqual(best.confidence, config["auto_tuning"]["run_confidence_floor"])


if __name__ == "__main__":
    unittest.main()
