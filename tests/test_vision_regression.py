from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from jumpjump.config import DEFAULT_CONFIG
from jumpjump.types import DetectionResult, RecognitionError
from jumpjump.vision import (
    _landing_platform_candidates_from_mask,
    collect_target_candidates,
    detect_jump,
    dynamic_piece_shape_reference,
    find_landing_platform,
    piece_candidates_from_mask,
    screen_overlay_present,
    update_piece_color_model,
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
    def test_dynamic_piece_shape_accepts_gradual_shrink_and_rejects_abrupt_square(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["piece"]["shape_samples"] = [
            {"width_ratio": 0.100, "height_ratio": 0.267},
            {"width_ratio": 0.098, "height_ratio": 0.262},
            {"width_ratio": 0.096, "height_ratio": 0.257},
        ]
        piece_bgr = cv2.cvtColor(
            np.array([[[122, 180, 100]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        crop = np.full((300, 400, 3), piece_bgr, dtype=np.uint8)

        gradual_mask = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(gradual_mask, (180, 150), (216, 222), 255, -1)
        gradual_candidates = piece_candidates_from_mask(gradual_mask, crop, config)

        square_mask = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(square_mask, (180, 170), (219, 211), 255, -1)
        square_candidates = piece_candidates_from_mask(square_mask, crop, config)

        self.assertTrue(gradual_candidates)
        self.assertEqual(square_candidates, [])

    def test_dynamic_piece_shape_compares_aspect_in_normalized_crop_space(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        crop_height, crop_width = 1344, 869
        piece_width, piece_height = 61, 112
        config["piece"]["shape_samples"] = [
            {
                "width_ratio": piece_width / crop_width,
                "height_ratio": piece_height / crop_height,
            }
            for _ in range(3)
        ]
        piece_bgr = cv2.cvtColor(
            np.array([[[124, 94, 86]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        crop = np.full((crop_height, crop_width, 3), piece_bgr, dtype=np.uint8)
        mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        cv2.rectangle(
            mask,
            (550, 670),
            (550 + piece_width - 1, 670 + piece_height - 1),
            255,
            -1,
        )

        candidates = piece_candidates_from_mask(mask, crop, config)

        self.assertTrue(candidates)

    def test_piece_shape_history_is_normalized_and_bounded(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["piece"]["dynamic_shape_max_samples"] = 3
        result = DetectionResult(
            piece=(120, 220),
            target=(300, 120),
            piece_bbox=(100, 120, 40, 80),
            target_bbox=(270, 100, 80, 40),
            crop_rect=(10, 20, 410, 320),
            dx_px=180.0,
            dy_px=-100.0,
            screen_distance_px=205.9,
            effective_distance_px=205.9,
            distance_px=205.9,
            confidence=0.8,
            debug_path=None,
            piece_median_hsv=None,
        )

        for _ in range(5):
            self.assertTrue(update_piece_color_model(config, result, "confirmed_success"))

        samples = config["piece"]["shape_samples"]
        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[-1]["width_ratio"], 0.1)
        self.assertAlmostEqual(samples[-1]["height_ratio"], 80 / 300, places=6)
        self.assertIsNotNone(dynamic_piece_shape_reference(config))

    def test_landing_surface_fragment_is_rejected_after_refinement(self) -> None:
        cv2_module, crop, mask = synthetic_scene(width=400, height=300)
        config = copy.deepcopy(DEFAULT_CONFIG)
        cv2_module.rectangle(mask, (100, 190), (300, 250), 255, -1)

        with patch(
            "jumpjump.vision.estimate_top_surface",
            return_value=((201, 201), (200, 197, 2, 8), 0.9, 16.0),
        ):
            candidates = _landing_platform_candidates_from_mask(
                crop,
                mask,
                (200, 210),
                (180, 130, 40, 80),
                config,
                confidence_scale=1.0,
            )

        self.assertEqual(candidates, [])

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
