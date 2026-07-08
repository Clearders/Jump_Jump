from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jumpjump.config import DEFAULT_CONFIG, load_config
from jumpjump.vision import detect_jump


class VisionRegressionTests(unittest.TestCase):
    def test_green_platform_sample_detects_platform_center(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError:
            self.skipTest("opencv-python is not installed")
        sample_path = Path("debug/auto_0005_20260708_230856_977308.png")
        if not sample_path.exists():
            self.skipTest(f"missing local regression image: {sample_path}")

        config_path = Path("jump_config.json")
        config = load_config(config_path) if config_path.exists() else copy.deepcopy(DEFAULT_CONFIG)
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)

        result = detect_jump(frame, config, Path("debug"), "vision_regression_green_platform")

        self.assertGreaterEqual(result.confidence, 0.45)
        self.assertGreater(result.target[0], result.piece[0])
        self.assertGreater(result.target_bbox[2], 100)


if __name__ == "__main__":
    unittest.main()
