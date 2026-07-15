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
    TargetCandidate,
    build_background_diff_mask,
    build_edge_mask,
    collect_target_candidates,
    crop_game_area,
    detect_game_score,
    detect_jump,
    dynamic_piece_shape_reference,
    estimate_top_surface,
    find_center_marker,
    find_landing_platform,
    find_target,
    keep_seeded_component,
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
    def marker_scene(self):
        crop = np.full((300, 400, 3), (150, 150, 150), dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        surface_bbox = (120, 100, 160, 80)
        surface_point = (200, 140)
        return crop, config, surface_point, surface_bbox

    def test_center_marker_refines_to_valid_near_white_component(self) -> None:
        crop, config, surface_point, surface_bbox = self.marker_scene()
        cv2.ellipse(crop, (215, 145), (8, 5), 0, 0, 360, (245, 245, 245), -1)

        marker = find_center_marker(crop, surface_point, surface_bbox, config)

        self.assertIsNotNone(marker)
        self.assertLessEqual(math.dist(marker.point, (215, 145)), 1.5)
        self.assertEqual(marker.source, "center_marker")
        self.assertGreaterEqual(marker.confidence, 0.75)

    def test_center_marker_rejects_trajectory_line_and_wrong_size(self) -> None:
        crop, config, surface_point, surface_bbox = self.marker_scene()
        cv2.line(crop, (140, 135), (260, 145), (245, 245, 245), 2)
        cv2.circle(crop, (200, 145), 3, (245, 245, 245), -1)

        self.assertIsNone(find_center_marker(crop, surface_point, surface_bbox, config))

    def test_center_marker_ignores_bright_ui_outside_surface(self) -> None:
        crop, config, surface_point, surface_bbox = self.marker_scene()
        # This sits inside the padded search ROI but outside the accepted
        # platform surface, like a nearby bright UI decoration.
        cv2.ellipse(crop, (116, 96), (8, 5), 0, 0, 360, (245, 245, 245), -1)

        self.assertIsNone(find_center_marker(crop, surface_point, surface_bbox, config))

    def test_center_marker_prefers_component_nearest_surface_center(self) -> None:
        crop, config, surface_point, surface_bbox = self.marker_scene()
        cv2.ellipse(crop, (150, 125), (8, 5), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(crop, (205, 142), (8, 5), 0, 0, 360, (245, 245, 245), -1)

        marker = find_center_marker(crop, surface_point, surface_bbox, config)

        self.assertIsNotNone(marker)
        self.assertLessEqual(math.dist(marker.point, (205, 142)), 1.5)

    def test_center_marker_can_be_disabled(self) -> None:
        crop, config, surface_point, surface_bbox = self.marker_scene()
        config["target"]["center_marker"]["enabled"] = False
        cv2.ellipse(crop, (200, 140), (8, 5), 0, 0, 360, (245, 245, 245), -1)

        self.assertIsNone(find_center_marker(crop, surface_point, surface_bbox, config))

    def test_detection_uses_marker_without_boosting_platform_confidence(self) -> None:
        frame, config, surface_point, surface_bbox = self.marker_scene()
        config["crop"].update(
            {"left_ratio": 0.0, "right_ratio": 1.0, "top_ratio": 0.0, "bottom_ratio": 1.0}
        )
        cv2.ellipse(frame, (215, 145), (8, 5), 0, 0, 360, (245, 245, 245), -1)
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        with (
            patch("jumpjump.vision.recognition_strategy_configs", return_value=[("default", config)]),
            patch(
                "jumpjump.vision.find_piece",
                return_value=((80, 230), (70, 180, 20, 50), empty_mask),
            ),
            patch("jumpjump.vision.build_background_diff_mask", return_value=empty_mask),
            patch("jumpjump.vision.build_edge_mask", return_value=empty_mask),
            patch(
                "jumpjump.vision.find_target",
                return_value=(surface_point, surface_bbox, 0.61, empty_mask),
            ),
            patch("jumpjump.vision.find_landing_platform", return_value=None),
            patch("jumpjump.vision.detect_game_score", return_value=None),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            result = detect_jump(
                frame,
                config,
                Path(tmpdir),
                "marker_pipeline",
                save_debug=False,
            )

        self.assertEqual(result.target_source, "center_marker")
        self.assertLessEqual(math.dist(result.target, (215, 145)), 1.5)
        self.assertEqual(result.confidence, 0.61)
        self.assertGreaterEqual(result.target_marker_confidence, 0.75)

    def test_background_diff_mask_matches_reference_reduction(self) -> None:
        rng = np.random.default_rng(20260714)
        crop = rng.integers(0, 256, size=(140, 180, 3), dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        target_cfg = config["target"]
        height, width = crop.shape[:2]
        margin = max(4, int(width * 0.04))
        sample = np.concatenate(
            [crop[:, :margin, :], crop[:, width - margin :, :]],
            axis=1,
        )
        sample_float = sample.astype(np.float32)
        sample_median = np.median(sample_float, axis=1, keepdims=True)
        sample_std = np.maximum(np.std(sample_float, axis=1, keepdims=True), 1.0)
        deviation = np.abs(sample_float - sample_median) / sample_std
        masked_sample = np.where(
            np.all(deviation < 2.0, axis=2, keepdims=True),
            sample_float,
            sample_median,
        )
        background = np.median(masked_sample, axis=1).reshape(height, 1, 3)
        diff = crop.astype(np.float32) - background.astype(np.float32)
        distance = np.sqrt(np.sum(diff * diff, axis=2))
        expected = (
            distance > float(target_cfg["diff_threshold"])
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN, kernel)
        expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, kernel, iterations=2)

        actual = build_background_diff_mask(crop, config)

        np.testing.assert_array_equal(actual, expected)

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

    def test_piece_without_platform_is_not_a_landing_platform(self) -> None:
        crop = np.full((320, 420, 3), (220, 205, 180), dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (210, 220)
        piece_bbox = (190, 145, 40, 82)
        crop[145:227, 190:230] = (58, 42, 78)

        self.assertIsNone(find_landing_platform(crop, piece, piece_bbox, config))

    def test_landing_surface_must_meet_configured_minimum_height(self) -> None:
        cv2_module, crop, mask = synthetic_scene(width=400, height=300)
        config = copy.deepcopy(DEFAULT_CONFIG)
        cv2_module.rectangle(mask, (100, 190), (300, 250), 255, -1)

        with patch(
            "jumpjump.vision.estimate_top_surface",
            return_value=((200, 201), (175, 197, 50, 8), 0.9, 400.0),
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

    def test_landing_refines_lower_surface_when_platform_components_merge(self) -> None:
        crop = np.full((320, 420, 3), (214, 224, 232), dtype=np.uint8)
        mask = np.zeros((320, 420), dtype=np.uint8)
        cv2.rectangle(crop, (40, 50), (220, 190), (104, 104, 104), -1)
        cv2.rectangle(mask, (40, 50), (220, 190), 255, -1)
        cv2.rectangle(crop, (180, 150), (380, 275), (246, 246, 246), -1)
        cv2.rectangle(mask, (180, 150), (380, 275), 255, -1)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["target"]["max_area_ratio"] = 0.60

        candidates = _landing_platform_candidates_from_mask(
            crop,
            mask,
            (290, 215),
            (270, 135, 40, 88),
            config,
            confidence_scale=1.0,
        )

        self.assertTrue(candidates)
        _, point, bbox = candidates[0]
        self.assertEqual(bbox, (180, 150, 201, 126))
        self.assertLessEqual(abs(point[0] - 290), 12)
        self.assertLessEqual(abs(point[1] - 215), 12)

    def test_tall_isometric_platform_can_support_piece_below_top_center(self) -> None:
        cv2_module, crop, mask = synthetic_scene(width=420, height=320)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["target"]["max_area_ratio"] = 0.60
        cv2_module.rectangle(mask, (70, 45), (350, 275), 255, -1)

        with patch(
            "jumpjump.vision.estimate_top_surface",
            return_value=((210, 80), (110, 45, 200, 90), 0.95, 18000.0),
        ):
            candidates = _landing_platform_candidates_from_mask(
                crop,
                mask,
                (210, 230),
                (190, 150, 40, 88),
                config,
                confidence_scale=1.0,
            )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0][2], (110, 45, 200, 90))
        self.assertLess(candidates[0][1][1], 100)

    def test_hollow_component_does_not_claim_to_support_piece_foot(self) -> None:
        cv2_module, crop, mask = synthetic_scene(width=420, height=320)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["target"]["max_area_ratio"] = 0.60
        cv2_module.rectangle(mask, (70, 45), (350, 275), 255, -1)
        cv2_module.rectangle(mask, (100, 80), (350, 275), 0, -1)

        with patch(
            "jumpjump.vision.estimate_top_surface",
            return_value=((210, 80), (110, 45, 200, 90), 0.95, 18000.0),
        ):
            candidates = _landing_platform_candidates_from_mask(
                crop,
                mask,
                (210, 230),
                (190, 150, 40, 88),
                config,
                confidence_scale=1.0,
            )

        self.assertEqual(candidates, [])

    def test_landing_hint_selects_the_previous_target_platform(self) -> None:
        crop = np.zeros((160, 240, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        wrong = (0.92, (55, 100), (20, 90, 70, 30))
        expected = (0.72, (175, 100), (140, 90, 70, 30))

        with patch(
            "jumpjump.vision._landing_platform_candidates_from_mask",
            return_value=[wrong, expected],
        ):
            result = find_landing_platform(
                crop,
                (170, 100),
                (160, 60, 20, 42),
                config,
                background_diff_mask=mask,
                edge_mask=mask,
                expected_x=175.0,
                expected_width=70.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[0], expected[1])
        self.assertEqual(result[1], expected[2])

    def test_landing_hint_rejects_an_implausibly_wide_platform(self) -> None:
        crop = np.zeros((160, 240, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        wrong = (0.90, (120, 100), (20, 90, 200, 30))
        expected = (0.72, (175, 100), (140, 90, 70, 30))

        with patch(
            "jumpjump.vision._landing_platform_candidates_from_mask",
            return_value=[wrong, expected],
        ):
            result = find_landing_platform(
                crop,
                (170, 100),
                (160, 60, 20, 42),
                config,
                background_diff_mask=mask,
                edge_mask=mask,
                expected_x=175.0,
                expected_left=140.0,
                expected_width=70.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[1], expected[2])

    def test_target_skips_edge_scoring_only_above_its_score_bound(self) -> None:
        crop = np.zeros((120, 160, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        piece = (40, 90)
        piece_bbox = (32, 58, 16, 32)
        diff_candidate = TargetCandidate(
            point=(120, 50),
            bbox=(110, 44, 20, 12),
            score=0.73,
            confidence=0.70,
            source="diff",
        )

        with patch(
            "jumpjump.vision.collect_target_candidates",
            return_value=[diff_candidate],
        ) as collect:
            result = find_target(
                crop,
                piece,
                piece_bbox,
                config,
                background_diff_mask=mask,
                edge_mask=mask,
            )

        self.assertEqual(result[:3], (diff_candidate.point, diff_candidate.bbox, 0.70))
        self.assertEqual(collect.call_count, 1)

        boundary_diff = TargetCandidate(
            point=(118, 52),
            bbox=(108, 46, 20, 12),
            score=0.72,
            confidence=0.60,
            source="diff",
        )
        boundary_edge = TargetCandidate(
            point=(122, 48),
            bbox=(112, 42, 20, 12),
            score=0.72,
            confidence=0.65,
            source="edge",
        )
        with patch(
            "jumpjump.vision.collect_target_candidates",
            side_effect=[[boundary_diff], [boundary_edge]],
        ) as collect:
            result = find_target(
                crop,
                piece,
                piece_bbox,
                config,
                background_diff_mask=mask,
                edge_mask=mask,
            )

        self.assertEqual(result[:3], (boundary_edge.point, boundary_edge.bbox, 0.65))
        self.assertEqual(collect.call_count, 2)

    def test_landing_prefilter_rejects_impossible_component_bbox(self) -> None:
        crop = np.zeros((300, 420, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (310, 20), (400, 100), 255, -1)
        config = copy.deepcopy(DEFAULT_CONFIG)

        with patch("jumpjump.vision.estimate_top_surface") as estimate:
            candidates = _landing_platform_candidates_from_mask(
                crop,
                mask,
                (60, 240),
                (45, 190, 30, 55),
                config,
                confidence_scale=1.0,
            )

        self.assertEqual(candidates, [])
        estimate.assert_not_called()

    def test_landing_prefilter_keeps_exact_gap_boundary(self) -> None:
        crop = np.zeros((180, 220, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (124, 80), (160, 120), 255, -1)
        config = copy.deepcopy(DEFAULT_CONFIG)

        with patch("jumpjump.vision.estimate_top_surface", return_value=None) as estimate:
            candidates = _landing_platform_candidates_from_mask(
                crop,
                mask,
                (100, 100),
                (85, 55, 30, 50),
                config,
                confidence_scale=1.0,
            )

        self.assertEqual(candidates, [])
        estimate.assert_called_once()

    def test_top_surface_skips_dominated_seeded_refinements(self) -> None:
        crop = np.full((180, 220, 3), (80, 130, 190), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (159, 119), 255, -1)
        contour = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0][0]
        bbox = cv2.boundingRect(contour)

        with patch(
            "jumpjump.vision.keep_seeded_component",
            wraps=keep_seeded_component,
        ) as keep_component:
            result = estimate_top_surface(
                crop,
                contour,
                bbox,
                copy.deepcopy(DEFAULT_CONFIG),
            )

        self.assertIsNotNone(result)
        self.assertEqual(keep_component.call_count, 1)

    def test_top_surface_lab_distance_matches_reference_result(self) -> None:
        crop = np.full((180, 220, 3), (214, 224, 232), dtype=np.uint8)
        for x in range(40, 160):
            crop[40:72, x] = (
                90 + (x - 40) // 12,
                130 + (x - 40) // 15,
                180 + (x - 40) // 20,
            )
        crop[72:120, 40:160] = (55, 85, 125)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (159, 119), 255, -1)
        contour = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0][0]
        bbox = cv2.boundingRect(contour)

        result = estimate_top_surface(
            crop,
            contour,
            bbox,
            copy.deepcopy(DEFAULT_CONFIG),
        )

        self.assertEqual(result, ((100, 54), (40, 40, 120, 32), 1.0, 3840))

    def test_top_surface_deduplicates_identical_unseeded_refinements(self) -> None:
        crop = np.full((180, 220, 3), (30, 80, 150), dtype=np.uint8)
        crop[40:48, 40:160] = (0, 0, 0)
        crop[48:56, 40:160] = (255, 255, 255)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (159, 119), 255, -1)
        contour = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0][0]
        bbox = cv2.boundingRect(contour)
        config = copy.deepcopy(DEFAULT_CONFIG)
        for key in (
            "top_surface_color_tolerance",
            "top_surface_hue_tolerance",
            "top_surface_saturation_tolerance",
            "top_surface_value_tolerance",
        ):
            config["target"][key] = 0
        geometry_fallback = ((100, 60), bbox, 0.55, 100)

        with (
            patch(
                "jumpjump.vision.keep_seeded_component",
                wraps=keep_seeded_component,
            ) as keep_component,
            patch(
                "jumpjump.vision.estimate_surface_by_geometry",
                return_value=geometry_fallback,
            ),
        ):
            result = estimate_top_surface(crop, contour, bbox, config)

        self.assertEqual(result, geometry_fallback)
        self.assertEqual(keep_component.call_count, 1)

    def test_landing_skips_edge_pass_only_above_confidence_bound(self) -> None:
        crop = np.zeros((160, 200, 3), dtype=np.uint8)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        diff_candidate = (0.77, (100, 100), (70, 90, 60, 20))

        with patch(
            "jumpjump.vision._landing_platform_candidates_from_mask",
            return_value=[diff_candidate],
        ) as collect:
            result = find_landing_platform(
                crop,
                (100, 100),
                (90, 60, 20, 42),
                config,
                background_diff_mask=mask,
                edge_mask=mask,
            )

        self.assertEqual(result, (diff_candidate[1], diff_candidate[2], 0.77))
        self.assertEqual(collect.call_count, 1)

        boundary_diff = (0.76, (98, 100), (68, 90, 60, 20))
        boundary_edge = (0.76, (102, 100), (72, 90, 60, 20))
        with patch(
            "jumpjump.vision._landing_platform_candidates_from_mask",
            side_effect=[[boundary_diff], [boundary_edge]],
        ) as collect:
            result = find_landing_platform(
                crop,
                (100, 100),
                (90, 60, 20, 42),
                config,
                background_diff_mask=mask,
                edge_mask=mask,
            )

        self.assertEqual(result, (boundary_diff[1], boundary_diff[2], 0.76))
        self.assertEqual(collect.call_count, 2)

        wide_diff = (0.77, (100, 100), (35, 90, 130, 20))
        matched_edge = (0.76, (100, 100), (65, 90, 70, 20))
        with patch(
            "jumpjump.vision._landing_platform_candidates_from_mask",
            side_effect=[[wide_diff], [matched_edge]],
        ) as collect:
            result = find_landing_platform(
                crop,
                (100, 100),
                (90, 60, 20, 42),
                config,
                background_diff_mask=mask,
                edge_mask=mask,
                expected_x=100.0,
                expected_left=65.0,
                expected_width=70.0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[1], matched_edge[2])
        self.assertEqual(collect.call_count, 2)

    def test_landing_hint_compares_all_recognition_strategies(self) -> None:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["crop"] = {
            "left_ratio": 0.0,
            "right_ratio": 1.0,
            "top_ratio": 0.0,
            "bottom_ratio": 1.0,
        }
        wide_config = copy.deepcopy(config)
        wide_config["target"]["diff_threshold"] = 12
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        piece_mask = mask.copy()
        hint = DetectionResult(
            piece=(60, 130),
            target=(170, 90),
            piece_bbox=(50, 90, 20, 42),
            target_bbox=(135, 80, 70, 25),
            crop_rect=(0, 0, 240, 180),
            dx_px=110.0,
            dy_px=-40.0,
            screen_distance_px=117.0,
            effective_distance_px=117.0,
            distance_px=117.0,
            confidence=0.9,
            debug_path=None,
        )
        wrong = ((110, 130), (20, 115, 190, 18), 0.74)
        expected = ((170, 130), (135, 115, 75, 18), 0.89)

        with (
            patch(
                "jumpjump.vision.recognition_strategy_configs",
                return_value=[("default", config), ("target_wide", wide_config)],
            ),
            patch("jumpjump.vision.screen_overlay_present", return_value=False),
            patch(
                "jumpjump.vision.find_piece",
                return_value=((60, 130), (50, 90, 20, 42), piece_mask),
            ),
            patch(
                "jumpjump.vision.find_target",
                return_value=((210, 70), (190, 60, 40, 20), 0.9, mask),
            ),
            patch("jumpjump.vision.build_background_diff_mask", return_value=mask),
            patch("jumpjump.vision.build_edge_mask", return_value=mask),
            patch(
                "jumpjump.vision.find_landing_platform",
                side_effect=[wrong, expected],
            ) as find_landing,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            result = detect_jump(
                frame,
                config,
                Path(tmpdir),
                "compare_landing_strategies",
                save_debug=False,
                landing_hint=hint,
            )

        self.assertEqual(find_landing.call_count, 2)
        self.assertEqual(result.landing_platform_bbox, expected[1])

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
        self.assertEqual(result.game_score, 0)
        self.assertGreaterEqual(result.game_score_confidence, 0.80)

    def test_score_reader_matches_live_block_font(self) -> None:
        cases = {
            "dry_run_20260714_135018_913721.png": 0,
            "auto_0011_failed_20260713_133702_562219.png": 30,
            "auto_0059_low_confidence_20260712_150557_580290.png": 95,
            "auto_0099_low_confidence_20260714_134418_189929.png": 417,
        }
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        available = 0
        for filename, expected in cases.items():
            path = debug_dir / filename
            if not path.is_file():
                continue
            frame = cv2.imread(str(path))
            self.assertIsNotNone(frame)
            available += 1
            crop, _ = crop_game_area(frame, copy.deepcopy(DEFAULT_CONFIG))
            result = detect_game_score(crop, copy.deepcopy(DEFAULT_CONFIG))
            self.assertIsNotNone(result)
            self.assertEqual(result[0], expected)
        if available == 0:
            self.skipTest("no local live score frames are available")

    def test_landing_refines_foot_surface_inside_merged_component(self) -> None:
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        sample_path = debug_dir / "auto_0011_failed_20260713_133702_562219.png"
        if not sample_path.is_file():
            self.skipTest("no local merged-platform frame is available")
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_jump(
                frame,
                copy.deepcopy(DEFAULT_CONFIG),
                Path(tmpdir),
                "vision_regression_merged_landing",
                save_debug=False,
            )

        self.assertIsNotNone(result.landing_platform)
        self.assertIsNotNone(result.landing_platform_bbox)
        self.assertGreaterEqual(result.landing_platform_confidence, 0.55)
        self.assertLessEqual(abs(result.landing_platform[0] - result.piece[0]), 8)
        self.assertLessEqual(abs(result.landing_platform[1] - result.piece[1]), 12)

    def test_detection_reuses_raw_masks_for_target_and_landing(self) -> None:
        sample_path = FIXTURE_DIR / "dry_run_20260710_003422_692036.png"
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)

        with (
            patch(
                "jumpjump.vision.build_background_diff_mask",
                wraps=build_background_diff_mask,
            ) as build_diff,
            patch("jumpjump.vision.build_edge_mask", wraps=build_edge_mask) as build_edge,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            result = detect_jump(
                frame,
                copy.deepcopy(DEFAULT_CONFIG),
                Path(tmpdir),
                "vision_regression_mask_reuse",
                save_debug=False,
            )

        self.assertGreaterEqual(result.confidence, 0.45)
        self.assertEqual(build_diff.call_count, 1)
        self.assertEqual(build_edge.call_count, 1)

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
