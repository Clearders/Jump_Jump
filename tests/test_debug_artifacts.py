from __future__ import annotations

import copy
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from jumpjump.config import DEFAULT_CONFIG
from jumpjump.debug_artifacts import (
    auto_capture_policy,
    enforce_debug_retention,
    is_generated_debug_image,
    write_debug_image,
)
from jumpjump.types import JumpAutoError


def fresh_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def write_sized_file(path: Path, size: int, mtime_ns: int) -> None:
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime_ns, mtime_ns))


class DebugArtifactTests(unittest.TestCase):
    def test_auto_capture_policy_uses_configured_mode(self) -> None:
        config = fresh_config()
        self.assertEqual(auto_capture_policy(config), "failures_and_rechecks")
        config["debug"]["auto_capture_policy"] = "all"
        self.assertEqual(auto_capture_policy(config), "all")

    def test_recognizes_only_project_generated_png_names(self) -> None:
        self.assertTrue(
            is_generated_debug_image(Path("auto_0001_20260710_010000_000001.png"))
        )
        self.assertTrue(
            is_generated_debug_image(
                Path("auto_0001_recheck_second_failed_20260710_010000_000001.png")
            )
        )
        self.assertTrue(
            is_generated_debug_image(
                Path("dry_run_20260710_010000_000001_target_mask.png")
            )
        )
        self.assertTrue(
            is_generated_debug_image(
                Path("calibrate_01_preview_20260710_010000_000001.png")
            )
        )
        self.assertFalse(is_generated_debug_image(Path("family.png")))
        self.assertFalse(is_generated_debug_image(Path("auto_vacation.png")))
        self.assertFalse(is_generated_debug_image(Path("dry_run_notes.png")))
        self.assertFalse(is_generated_debug_image(Path("auto_0001_notes.txt")))

    def test_retention_enforces_file_count_and_preserves_unrelated_files(self) -> None:
        config = fresh_config()
        config["debug"]["max_files"] = 2
        config["debug"]["max_size_mb"] = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            oldest = debug_dir / "auto_0000_20260710_010000_000001.png"
            middle = debug_dir / "dry_run_20260710_010001_000001.png"
            newest = debug_dir / "single_step_20260710_010002_000001.png"
            unrelated_png = debug_dir / "family.png"
            unrelated_text = debug_dir / "auto_0000_notes.txt"
            write_sized_file(oldest, 10, 1_000_000_000)
            write_sized_file(middle, 10, 2_000_000_000)
            write_sized_file(newest, 10, 3_000_000_000)
            write_sized_file(unrelated_png, 10, 500_000_000)
            write_sized_file(unrelated_text, 10, 500_000_000)

            enforce_debug_retention(debug_dir, config)

            self.assertFalse(oldest.exists())
            self.assertTrue(middle.exists())
            self.assertTrue(newest.exists())
            self.assertTrue(unrelated_png.exists())
            self.assertTrue(unrelated_text.exists())

    def test_retention_enforces_total_size_oldest_first(self) -> None:
        config = fresh_config()
        config["debug"]["max_files"] = 10
        config["debug"]["max_size_mb"] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            paths = [
                debug_dir / "auto_0000_20260710_010000_000001.png",
                debug_dir / "auto_0001_20260710_010001_000001.png",
                debug_dir / "auto_0002_20260710_010002_000001.png",
            ]
            for index, path in enumerate(paths, start=1):
                write_sized_file(path, 600_000, index * 1_000_000_000)

            enforce_debug_retention(debug_dir, config)

            self.assertFalse(paths[0].exists())
            self.assertFalse(paths[1].exists())
            self.assertTrue(paths[2].exists())

    def test_undeletable_old_file_does_not_corrupt_count_accounting(self) -> None:
        config = fresh_config()
        config["debug"]["max_files"] = 2
        config["debug"]["max_size_mb"] = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            oldest = debug_dir / "auto_0000_20260710_010000_000001.png"
            middle = debug_dir / "auto_0001_20260710_010001_000001.png"
            newest = debug_dir / "auto_0002_20260710_010002_000001.png"
            for index, path in enumerate((oldest, middle, newest), start=1):
                write_sized_file(path, 10, index * 1_000_000_000)

            def guarded_unlink(path, *args, **kwargs):
                if path == oldest:
                    raise PermissionError("locked")
                os.unlink(path)

            with (
                patch.object(Path, "unlink", autospec=True, side_effect=guarded_unlink),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                enforce_debug_retention(debug_dir, config)

            self.assertTrue(caught)
            self.assertTrue(oldest.exists())
            self.assertFalse(middle.exists())
            self.assertTrue(newest.exists())

    def test_failed_image_write_raises_instead_of_returning_missing_path(self) -> None:
        fake_cv2 = MagicMock()
        fake_cv2.imwrite.return_value = False
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dry_run_failed.png"
            with patch("jumpjump.debug_artifacts.import_cv", return_value=(fake_cv2, object())):
                with self.assertRaises(JumpAutoError):
                    write_debug_image(output, object(), fresh_config())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
