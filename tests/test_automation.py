from __future__ import annotations

import copy
from dataclasses import replace
import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from jumpjump.automation import (
    click_point_for_rect,
    detection_recheck_consistency,
    focus_window,
    confidence_run_decision,
    press_in_window,
    recheck_low_confidence_detection,
    recognition_failure_pause_status,
    record_auto_success_if_landed,
    result_is_good_learning_candidate,
    run_auto,
    run_single_step,
)
from jumpjump.config import DEFAULT_CONFIG, press_model_config
from jumpjump.types import DependencyError, DetectionResult, JumpAutoError, WindowInfo


def fresh_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def detection(
    *,
    piece: tuple[int, int],
    target: tuple[int, int],
    distance: float,
    dx: float | None = None,
    dy: float = 0.0,
    confidence: float = 0.9,
    debug_path: Path | None = Path("debug.png"),
) -> DetectionResult:
    dx = distance if dx is None else dx
    return DetectionResult(
        piece=piece,
        target=target,
        piece_bbox=(0, 0, 20, 40),
        target_bbox=(80, 0, 40, 20),
        crop_rect=(0, 0, 400, 700),
        dx_px=dx,
        dy_px=dy,
        screen_distance_px=distance,
        effective_distance_px=distance,
        distance_px=distance,
        confidence=confidence,
        debug_path=debug_path,
    )


def test_window() -> WindowInfo:
    return WindowInfo(
        hwnd=10,
        title="Jump",
        window_rect=(0, 0, 100, 200),
        client_rect=(0, 0, 100, 200),
        iconic=False,
    )


class FakeWin32Gui:
    def __init__(self) -> None:
        self.exists = True
        self.visible = True
        self.iconic = False
        self.foreground = 10
        self.point_window = 10
        self.set_foreground_calls: list[int] = []

    def IsWindow(self, _: int) -> bool:
        return self.exists

    def IsWindowVisible(self, _: int) -> bool:
        return self.visible

    def IsIconic(self, _: int) -> bool:
        return self.iconic

    def GetForegroundWindow(self) -> int:
        return self.foreground

    def GetAncestor(self, hwnd: int, _: int) -> int:
        return 100 if hwnd == 10 else 200

    def WindowFromPoint(self, _: tuple[int, int]) -> int:
        return self.point_window

    def SetForegroundWindow(self, hwnd: int) -> None:
        self.set_foreground_calls.append(hwnd)


class AutomationBehaviorTests(unittest.TestCase):
    def test_low_confidence_above_run_floor_requires_recheck_and_is_not_learning_candidate(self) -> None:
        config = fresh_config()
        config["confidence_threshold"] = 0.45
        config["auto_tuning"]["run_confidence_floor"] = 0.35
        result = detection(piece=(0, 0), target=(100, 0), distance=100.0, confidence=0.40)

        should_pause, is_low_confidence, run_floor, threshold = confidence_run_decision(
            config,
            result.confidence,
        )

        self.assertFalse(should_pause)
        self.assertTrue(is_low_confidence)
        self.assertEqual(run_floor, 0.35)
        self.assertEqual(threshold, 0.45)
        self.assertFalse(result_is_good_learning_candidate(config, result))

    def test_confidence_boundaries_and_nonfinite_values_are_safe(self) -> None:
        config = fresh_config()
        config["confidence_threshold"] = 0.45
        config["auto_tuning"]["run_confidence_floor"] = 0.35

        self.assertEqual(
            confidence_run_decision(config, 0.45)[:2],
            (False, False),
        )
        self.assertEqual(
            confidence_run_decision(config, 0.35)[:2],
            (False, True),
        )
        self.assertEqual(
            confidence_run_decision(config, float("nan"))[:2],
            (True, True),
        )

    def test_recognition_failures_pause_only_at_limit(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["max_recognition_failures_before_pause"] = 3

        self.assertEqual(recognition_failure_pause_status(config, 1), (False, 3))
        self.assertEqual(recognition_failure_pause_status(config, 2), (False, 3))
        self.assertEqual(recognition_failure_pause_status(config, 3), (True, 3))

    def test_auto_success_writes_adjusted_training_press(self) -> None:
        config = fresh_config()
        model = press_model_config(config)
        model["slope_ms_per_px"] = 2.0
        config["press_ms_per_px"] = 2.0
        previous = detection(piece=(0, 0), target=(100, 0), distance=100.0)
        current = detection(piece=(112, 0), target=(180, 0), distance=68.0, dx=68.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                recorded = record_auto_success_if_landed(
                    config,
                    Path(tmpdir) / "jump_config.json",
                    {"result": previous, "press_ms": 200.0},
                    current,
                )

        self.assertTrue(recorded)
        sample = model["samples"][-1]
        self.assertEqual(sample["press_ms"], 200.0)
        self.assertEqual(sample["training_press_ms"], sample["center_adjusted_press_ms"])
        self.assertLess(sample["training_press_ms"], sample["press_ms"])

    def test_recheck_recaptures_once_and_returns_recovered_result(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["low_confidence_recheck_delay_s"] = 0.0
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )
        second = detection(
            piece=(104, 302),
            target=(308, 205),
            distance=226.0,
            dx=204.0,
            dy=-97.0,
            confidence=0.55,
            debug_path=None,
        )
        first_frame = object()
        second_frame = object()
        rect = (0, 0, 400, 700)

        def mark_saved(frame, result, config, debug_dir, label):
            return replace(result, debug_path=Path(f"{label}.png"))

        with (
            patch("jumpjump.automation.capture_window", return_value=(second_frame, rect)) as capture,
            patch("jumpjump.automation.detect_jump", return_value=second) as detect,
            patch("jumpjump.automation.save_detection_result_debug", side_effect=mark_saved),
        ):
            verified, verified_rect, reason = recheck_low_confidence_detection(
                test_window(),
                first_frame,
                rect,
                first,
                config,
                Path("debug"),
                "auto_0000",
                threading.Event(),
                threading.Event(),
            )

        self.assertIs(verified, second)
        self.assertEqual(verified_rect, rect)
        self.assertEqual(reason, "detections agree")
        capture.assert_called_once()
        detect.assert_called_once()

    def test_recheck_rejects_low_or_inconsistent_second_detection(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["low_confidence_recheck_delay_s"] = 0.0
        rect = (0, 0, 400, 700)
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )

        def mark_saved(frame, result, config, debug_dir, label):
            return replace(result, debug_path=Path(f"{label}.png"))

        cases = [
            detection(
                piece=(102, 301),
                target=(305, 204),
                distance=225.0,
                dx=203.0,
                dy=-97.0,
                confidence=0.44,
                debug_path=None,
            ),
            detection(
                piece=(102, 301),
                target=(100, 205),
                distance=96.0,
                dx=-2.0,
                dy=-96.0,
                confidence=0.60,
                debug_path=None,
            ),
        ]
        for second in cases:
            with self.subTest(second=second):
                with (
                    patch("jumpjump.automation.capture_window", return_value=(object(), rect)) as capture,
                    patch("jumpjump.automation.detect_jump", return_value=second),
                    patch("jumpjump.automation.save_detection_result_debug", side_effect=mark_saved),
                ):
                    verified, _, _ = recheck_low_confidence_detection(
                        test_window(),
                        object(),
                        rect,
                        first,
                        config,
                        Path("debug"),
                        "auto_0000",
                        threading.Event(),
                        threading.Event(),
                    )
                self.assertIsNone(verified)
                capture.assert_called_once()

    def test_recheck_capture_failure_rejects_without_second_detection(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["low_confidence_recheck_delay_s"] = 0.0
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )

        def mark_saved(frame, result, config, debug_dir, label):
            return replace(result, debug_path=Path(f"{label}.png"))

        with (
            patch(
                "jumpjump.automation.capture_window",
                side_effect=RuntimeError("capture failed"),
            ) as capture,
            patch("jumpjump.automation.detect_jump") as detect,
            patch("jumpjump.automation.save_detection_result_debug", side_effect=mark_saved),
        ):
            verified, _, reason = recheck_low_confidence_detection(
                test_window(),
                object(),
                (0, 0, 400, 700),
                first,
                config,
                Path("debug"),
                "auto_0000",
                threading.Event(),
                threading.Event(),
            )

        self.assertIsNone(verified)
        self.assertIn("capture failed", reason)
        capture.assert_called_once()
        detect.assert_not_called()

    def test_recheck_unexpected_detection_error_is_rejected(self) -> None:
        config = fresh_config()
        config["auto_tuning"]["low_confidence_recheck_delay_s"] = 0.0
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )
        second_frame = SimpleNamespace(shape=(700, 400, 3))

        def mark_saved(frame, result, config, debug_dir, label):
            return replace(result, debug_path=Path(f"{label}.png"))

        with (
            patch(
                "jumpjump.automation.capture_window",
                return_value=(second_frame, (0, 0, 400, 700)),
            ),
            patch("jumpjump.automation.detect_jump", side_effect=RuntimeError("opencv failed")),
            patch("jumpjump.automation.save_detection_result_debug", side_effect=mark_saved),
            patch(
                "jumpjump.automation.save_recognition_failure_debug",
                return_value=Path("second_failed.png"),
            ),
        ):
            verified, _, reason = recheck_low_confidence_detection(
                test_window(),
                object(),
                (0, 0, 400, 700),
                first,
                config,
                Path("debug"),
                "auto_0000",
                threading.Event(),
                threading.Event(),
            )

        self.assertIsNone(verified)
        self.assertIn("opencv failed", reason)

    def test_detection_recheck_respects_position_tolerances(self) -> None:
        config = fresh_config()
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
        )
        within = detection(
            piece=(106, 300),
            target=(310, 200),
            distance=225.0,
            dx=204.0,
            dy=-100.0,
        )
        outside = detection(
            piece=(107, 300),
            target=(310, 200),
            distance=225.0,
            dx=203.0,
            dy=-100.0,
        )
        rect = (0, 0, 400, 700)

        self.assertTrue(detection_recheck_consistency(first, within, rect, rect, config)[0])
        self.assertFalse(detection_recheck_consistency(first, outside, rect, rect, config)[0])


class PressSafetyTests(unittest.TestCase):
    def test_click_point_at_one_ratio_is_clamped_inside_client_rect(self) -> None:
        config = fresh_config()
        config["click_point"] = {"x_ratio": 1.0, "y_ratio": 1.0}
        self.assertEqual(click_point_for_rect((10, 20, 110, 220), config), (109, 219))

    def _press_with_window_state(
        self,
        gui: FakeWin32Gui,
        *,
        current_rect: tuple[int, int, int, int] = (0, 0, 100, 200),
        obscured: bool = False,
    ) -> MagicMock:
        pyautogui = MagicMock()
        with (
            patch("jumpjump.automation.focus_window"),
            patch("jumpjump.automation.require_windows"),
            patch(
                "jumpjump.automation.import_win32",
                return_value=(gui, SimpleNamespace(GA_ROOT=2), None),
            ),
            patch("jumpjump.automation.client_rect_on_screen", return_value=current_rect),
            patch("jumpjump.automation.client_area_looks_obscured", return_value=obscured),
            patch("jumpjump.automation.import_pyautogui", return_value=pyautogui),
        ):
            with self.assertRaises(JumpAutoError):
                press_in_window(
                    test_window(),
                    (0, 0, 100, 200),
                    fresh_config(),
                    200.0,
                )
        return pyautogui

    def test_each_unsafe_window_state_prevents_mouse_down(self) -> None:
        cases = []

        missing = FakeWin32Gui()
        missing.exists = False
        cases.append(("missing", missing, (0, 0, 100, 200), False))

        hidden = FakeWin32Gui()
        hidden.visible = False
        cases.append(("hidden", hidden, (0, 0, 100, 200), False))

        minimized = FakeWin32Gui()
        minimized.iconic = True
        cases.append(("minimized", minimized, (0, 0, 100, 200), False))

        moved = FakeWin32Gui()
        cases.append(("moved", moved, (1, 0, 101, 200), False))

        background = FakeWin32Gui()
        background.foreground = 99
        cases.append(("background", background, (0, 0, 100, 200), False))

        covered = FakeWin32Gui()
        cases.append(("covered", covered, (0, 0, 100, 200), True))

        wrong_owner = FakeWin32Gui()
        wrong_owner.point_window = 99
        cases.append(("wrong_owner", wrong_owner, (0, 0, 100, 200), False))

        for name, gui, current_rect, obscured in cases:
            with self.subTest(name=name):
                pyautogui = self._press_with_window_state(
                    gui,
                    current_rect=current_rect,
                    obscured=obscured,
                )
                pyautogui.mouseDown.assert_not_called()

    def test_focus_failure_prevents_mouse_input(self) -> None:
        pyautogui = MagicMock()
        with (
            patch(
                "jumpjump.automation.focus_window",
                side_effect=JumpAutoError("focus failed"),
            ),
            patch("jumpjump.automation.import_pyautogui", return_value=pyautogui),
        ):
            with self.assertRaises(JumpAutoError):
                press_in_window(test_window(), (0, 0, 100, 200), fresh_config(), 200.0)
        pyautogui.mouseDown.assert_not_called()

    def test_pause_after_cursor_move_prevents_mouse_down(self) -> None:
        pyautogui = MagicMock()
        pause_event = threading.Event()
        verification_count = 0

        def verify(*args, **kwargs):
            nonlocal verification_count
            verification_count += 1
            if verification_count == 2:
                pause_event.set()

        with (
            patch("jumpjump.automation.focus_window"),
            patch("jumpjump.automation.verify_press_target", side_effect=verify),
            patch("jumpjump.automation.import_pyautogui", return_value=pyautogui),
        ):
            with self.assertRaises(JumpAutoError):
                press_in_window(
                    test_window(),
                    (0, 0, 100, 200),
                    fresh_config(),
                    200.0,
                    pause_event=pause_event,
                )
        pyautogui.moveTo.assert_called_once()
        pyautogui.mouseDown.assert_not_called()

    def test_mouse_up_runs_after_mouse_down_error(self) -> None:
        pyautogui = MagicMock()
        pyautogui.mouseDown.side_effect = RuntimeError("partial mouse down")
        with (
            patch("jumpjump.automation.focus_window"),
            patch("jumpjump.automation.verify_press_target"),
            patch("jumpjump.automation.import_pyautogui", return_value=pyautogui),
        ):
            with self.assertRaises(RuntimeError):
                press_in_window(test_window(), (0, 0, 100, 200), fresh_config(), 200.0)
        pyautogui.mouseUp.assert_called_once()

    def test_action_gate_prevents_pause_race_before_mouse_down(self) -> None:
        pyautogui = MagicMock()
        moved = threading.Event()
        pause_event = threading.Event()
        action_lock = threading.Lock()
        action_lock.acquire()
        pyautogui.moveTo.side_effect = lambda *args, **kwargs: moved.set()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                press_in_window(
                    test_window(),
                    (0, 0, 100, 200),
                    fresh_config(),
                    200.0,
                    pause_event=pause_event,
                    action_lock=action_lock,
                )
            except BaseException as exc:
                errors.append(exc)

        with (
            patch("jumpjump.automation.focus_window"),
            patch("jumpjump.automation.verify_press_target"),
            patch("jumpjump.automation.import_pyautogui", return_value=pyautogui),
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(moved.wait(timeout=1.0))
            pause_event.set()
            action_lock.release()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertIsInstance(errors[0], JumpAutoError)
        pyautogui.mouseDown.assert_not_called()


class FocusWindowTests(unittest.TestCase):
    def test_focus_requires_target_root_to_become_foreground(self) -> None:
        gui = FakeWin32Gui()
        with (
            patch("jumpjump.automation.require_windows"),
            patch(
                "jumpjump.automation.import_win32",
                return_value=(gui, SimpleNamespace(GA_ROOT=2), None),
            ),
        ):
            focus_window(10, timeout_s=0.0)
        self.assertEqual(gui.set_foreground_calls, [10])

        gui.foreground = 99
        with (
            patch("jumpjump.automation.require_windows"),
            patch(
                "jumpjump.automation.import_win32",
                return_value=(gui, SimpleNamespace(GA_ROOT=2), None),
            ),
        ):
            with self.assertRaises(JumpAutoError):
                focus_window(10, timeout_s=0.0)


class AutomationLoopTests(unittest.TestCase):
    def test_accepted_recheck_uses_second_result_for_learning_and_press(self) -> None:
        config = fresh_config()
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )
        second = detection(
            piece=(102, 301),
            target=(304, 202),
            distance=225.0,
            dx=202.0,
            dy=-99.0,
            confidence=0.70,
            debug_path=Path("recheck.png"),
        )
        window = test_window()
        rect = window.client_rect
        listener = MagicMock()

        def stop_after_press(*args, **kwargs):
            kwargs["stop_event"].set()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                no_auto_tune=False,
                window_title=None,
                debug_dir=Path(tmpdir) / "debug",
                config=Path(tmpdir) / "jump_config.json",
                interval=0.0,
            )
            with (
                patch("jumpjump.automation.start_hotkey_listener", return_value=listener),
                patch("jumpjump.automation.locate_window", return_value=window),
                patch("jumpjump.automation.capture_window", return_value=(object(), rect)),
                patch("jumpjump.automation.detect_jump", return_value=first),
                patch(
                    "jumpjump.automation.recheck_low_confidence_detection",
                    return_value=(second, rect, "detections agree"),
                ) as recheck,
                patch("jumpjump.automation.record_auto_success_if_landed") as learn,
                patch("jumpjump.automation.calculate_press_ms", return_value=240.0) as calculate,
                patch("jumpjump.automation.press_in_window", side_effect=stop_after_press) as press,
                patch("jumpjump.automation.time.sleep"),
                redirect_stdout(io.StringIO()),
            ):
                run_auto(args, config)

        recheck.assert_called_once()
        self.assertIs(learn.call_args.args[3], second)
        calculate.assert_called_once_with(second, config)
        self.assertIs(press.call_args.args[0], window)
        self.assertEqual(press.call_args.args[1], rect)

    def test_high_confidence_result_does_not_recheck(self) -> None:
        config = fresh_config()
        result = detection(piece=(100, 300), target=(300, 200), distance=224.0, confidence=0.90)
        window = test_window()
        listener = MagicMock()

        def stop_after_press(*args, **kwargs):
            kwargs["stop_event"].set()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                no_auto_tune=True,
                window_title=None,
                debug_dir=Path(tmpdir) / "debug",
                config=Path(tmpdir) / "jump_config.json",
                interval=0.0,
            )
            with (
                patch("jumpjump.automation.start_hotkey_listener", return_value=listener),
                patch("jumpjump.automation.locate_window", return_value=window),
                patch("jumpjump.automation.capture_window", return_value=(object(), window.client_rect)),
                patch("jumpjump.automation.detect_jump", return_value=result),
                patch("jumpjump.automation.recheck_low_confidence_detection") as recheck,
                patch("jumpjump.automation.calculate_press_ms", return_value=240.0),
                patch("jumpjump.automation.press_in_window", side_effect=stop_after_press),
                patch("jumpjump.automation.time.sleep"),
                redirect_stdout(io.StringIO()),
            ):
                run_auto(args, config)

        recheck.assert_not_called()

    def test_rejected_recheck_never_presses_or_learns(self) -> None:
        class StopPausedLoop(RuntimeError):
            pass

        config = fresh_config()
        first = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            dx=200.0,
            dy=-100.0,
            confidence=0.40,
            debug_path=None,
        )
        window = test_window()
        listener = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                no_auto_tune=False,
                window_title=None,
                debug_dir=Path(tmpdir) / "debug",
                config=Path(tmpdir) / "jump_config.json",
                interval=0.0,
            )
            with (
                patch("jumpjump.automation.start_hotkey_listener", return_value=listener),
                patch("jumpjump.automation.locate_window", return_value=window),
                patch("jumpjump.automation.capture_window", return_value=(object(), window.client_rect)),
                patch("jumpjump.automation.detect_jump", return_value=first),
                patch(
                    "jumpjump.automation.recheck_low_confidence_detection",
                    return_value=(None, window.client_rect, "confidence stayed low"),
                ),
                patch("jumpjump.automation.record_auto_success_if_landed") as learn,
                patch("jumpjump.automation.press_in_window") as press,
                patch("jumpjump.automation.time.sleep", side_effect=StopPausedLoop),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(StopPausedLoop):
                    run_auto(args, config)

        learn.assert_not_called()
        press.assert_not_called()

    def test_press_dependency_error_is_not_converted_to_resumable_pause(self) -> None:
        config = fresh_config()
        result = detection(piece=(100, 300), target=(300, 200), distance=224.0, confidence=0.90)
        window = test_window()
        listener = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                no_auto_tune=True,
                window_title=None,
                debug_dir=Path(tmpdir) / "debug",
                config=Path(tmpdir) / "jump_config.json",
                interval=0.0,
            )
            with (
                patch("jumpjump.automation.start_hotkey_listener", return_value=listener),
                patch("jumpjump.automation.locate_window", return_value=window),
                patch("jumpjump.automation.capture_window", return_value=(object(), window.client_rect)),
                patch("jumpjump.automation.detect_jump", return_value=result),
                patch("jumpjump.automation.calculate_press_ms", return_value=240.0),
                patch(
                    "jumpjump.automation.press_in_window",
                    side_effect=DependencyError("pyautogui missing"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(DependencyError):
                    run_auto(args, config)

    def test_single_step_rejects_nonfinite_confidence(self) -> None:
        config = fresh_config()
        result = detection(
            piece=(100, 300),
            target=(300, 200),
            distance=224.0,
            confidence=float("nan"),
        )
        window = test_window()
        args = SimpleNamespace(
            window_title=None,
            debug_dir=Path("debug"),
            save_masks=False,
        )
        with (
            patch("jumpjump.automation.locate_window", return_value=window),
            patch("jumpjump.automation.capture_window", return_value=(object(), window.client_rect)),
            patch("jumpjump.automation.detect_jump", return_value=result),
            patch("jumpjump.automation.calculate_press_ms", return_value=240.0),
            patch("jumpjump.automation.press_in_window") as press,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(JumpAutoError):
                run_single_step(args, config)
        press.assert_not_called()


if __name__ == "__main__":
    unittest.main()
