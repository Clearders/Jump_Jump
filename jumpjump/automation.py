from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_CONFIG, auto_tuning_config, press_model_config, save_config
from .dependencies import (
    import_cv,
    import_mss,
    import_pyautogui,
    import_pynput_keyboard,
    import_pynput_mouse,
    import_win32,
    require_windows,
)
from .press_model import (
    calculate_press_ms,
    calibration_sample_from_result,
    center_adjusted_press_ms,
    clear_failure_caps_near_success,
    decay_segment_center_correction,
    fit_press_model,
    mark_segment_precision_hit,
    maybe_unfreeze_segment_for_error,
    record_segment_center_correction,
    segment_correction_ms,
    segment_is_frozen,
)
from .types import DetectionResult, JumpAutoError, RecognitionError, WindowInfo
from .utils import clamp, timestamp
from .vision import detect_jump, update_piece_color_model


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    win32gui, _, _ = import_win32()
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return screen_left, screen_top, screen_right, screen_bottom


def client_area_looks_obscured(hwnd: int, rect: tuple[int, int, int, int]) -> bool:
    win32gui, win32con, _ = import_win32()
    ga_root = getattr(win32con, "GA_ROOT", 2)
    root_hwnd = win32gui.GetAncestor(hwnd, ga_root)
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    points = [
        (left + width // 2, top + height // 2),
        (left + width // 3, top + height // 3),
        (left + width * 2 // 3, top + height // 3),
        (left + width // 3, top + height * 2 // 3),
        (left + width * 2 // 3, top + height * 2 // 3),
    ]
    mismatches = 0
    for point in points:
        point_hwnd = win32gui.WindowFromPoint(point)
        point_root = win32gui.GetAncestor(point_hwnd, ga_root)
        if point_root != root_hwnd:
            mismatches += 1
    return mismatches >= 2


def enumerate_windows() -> list[WindowInfo]:
    require_windows()
    win32gui, _, _ = import_win32()
    windows: list[WindowInfo] = []

    def callback(hwnd: int, _: Any) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return
        try:
            window_rect = tuple(int(v) for v in win32gui.GetWindowRect(hwnd))
            client_rect = client_rect_on_screen(hwnd)
            iconic = bool(win32gui.IsIconic(hwnd))
        except Exception:
            return
        width = client_rect[2] - client_rect[0]
        height = client_rect[3] - client_rect[1]
        if width < 120 or height < 120:
            return
        windows.append(WindowInfo(hwnd, title, window_rect, client_rect, iconic))

    win32gui.EnumWindows(callback, None)
    return windows


def window_score(window: WindowInfo, keywords: Iterable[str]) -> int:
    title_lower = window.title.lower()
    score = 0
    for keyword in keywords:
        if keyword and keyword.lower() in title_lower:
            score += 100 if keyword == "跳一跳" else 50
    aspect = window.client_width / max(1, window.client_height)
    if 0.45 <= aspect <= 0.90:
        score += 15
    if window.iconic:
        score -= 1000
    return score


def locate_window(title_hint: str | None, config: dict[str, Any]) -> WindowInfo:
    windows = enumerate_windows()
    min_width = int(config["min_client_width"])
    min_height = int(config["min_client_height"])
    eligible = [
        window
        for window in windows
        if not window.iconic
        and window.client_width >= min_width
        and window.client_height >= min_height
    ]

    if title_hint:
        matches = [window for window in eligible if title_hint.lower() in window.title.lower()]
        if matches:
            return max(matches, key=lambda item: item.client_width * item.client_height)
        raise JumpAutoError(
            f"No visible window matched --window-title '{title_hint}'.\n"
            + format_window_candidates(windows)
        )

    configured_title = str(config.get("window_title") or "").strip()
    if configured_title:
        matches = [
            window for window in eligible if configured_title.lower() in window.title.lower()
        ]
        if matches:
            return max(matches, key=lambda item: item.client_width * item.client_height)

    keywords = config.get("window_keywords") or DEFAULT_CONFIG["window_keywords"]
    scored = [(window_score(window, keywords), window) for window in eligible]
    scored = [item for item in scored if item[0] > 0]
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    raise JumpAutoError(
        "No visible WeChat/Jump window was found.\n" + format_window_candidates(windows)
    )


def format_window_candidates(windows: list[WindowInfo], limit: int = 20) -> str:
    if not windows:
        return "No visible windows were found."
    rows = ["Visible window candidates:"]
    for window in windows[:limit]:
        state = "minimized" if window.iconic else "visible"
        rows.append(
            f"  hwnd={window.hwnd} size={window.client_width}x{window.client_height} "
            f"state={state} title={window.title!r}"
        )
    if len(windows) > limit:
        rows.append(f"  ... {len(windows) - limit} more")
    return "\n".join(rows)


def print_window_list() -> None:
    print(format_window_candidates(enumerate_windows(), limit=100))


def capture_window(window: WindowInfo, config: dict[str, Any]):
    require_windows()
    _, _, _ = import_win32()
    mss_module = import_mss()
    _, np = import_cv()

    current_rect = client_rect_on_screen(window.hwnd)
    width = current_rect[2] - current_rect[0]
    height = current_rect[3] - current_rect[1]
    if width < int(config["min_client_width"]) or height < int(config["min_client_height"]):
        raise JumpAutoError(f"Window client area is too small: {width}x{height}.")
    if client_area_looks_obscured(window.hwnd, current_rect):
        raise JumpAutoError("The target window appears to be covered by another window.")

    monitor = {
        "left": current_rect[0],
        "top": current_rect[1],
        "width": width,
        "height": height,
    }
    mss_factory = getattr(mss_module, "MSS", None) or mss_module.mss
    with mss_factory() as sct:
        frame = np.array(sct.grab(monitor))
    return frame[:, :, :3].copy(), current_rect


def focus_window(hwnd: int) -> None:
    require_windows()
    win32gui, win32con, _ = import_win32()
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.15)
    except Exception:
        pass


def press_in_window(window: WindowInfo, client_rect: tuple[int, int, int, int], config: dict[str, Any], press_ms: float) -> None:
    pyautogui = import_pyautogui()
    pyautogui.PAUSE = 0.02
    pyautogui.FAILSAFE = True
    left, top, right, bottom = client_rect
    click_cfg = config["click_point"]
    x = int(left + (right - left) * float(click_cfg["x_ratio"]))
    y = int(top + (bottom - top) * float(click_cfg["y_ratio"]))
    focus_window(window.hwnd)
    pyautogui.moveTo(x, y, duration=0)
    pressed = False
    try:
        pyautogui.mouseDown()
        pressed = True
        time.sleep(max(0.0, press_ms / 1000.0))
    finally:
        if pressed:
            pyautogui.mouseUp()


def run_dry_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    window = locate_window(args.window_title, config)
    frame, _ = capture_window(window, config)
    result = detect_jump(frame, config, args.debug_dir, "dry_run", save_mask=args.save_masks)
    print_detection(window, result)


def run_single_step(args: argparse.Namespace, config: dict[str, Any]) -> None:
    window = locate_window(args.window_title, config)
    frame, client_rect = capture_window(window, config)
    first_result = detect_jump(frame, config, args.debug_dir, "single_step_preview")
    press_ms = calculate_press_ms(first_result, config)
    result = detect_jump(frame, config, args.debug_dir, "single_step", press_ms=press_ms)
    print_detection(window, result, press_ms=press_ms)
    if result.confidence < float(config["confidence_threshold"]):
        raise JumpAutoError(
            f"Recognition confidence {result.confidence:.2f} is below threshold; not pressing."
        )
    press_in_window(window, client_rect, config, press_ms)
    print("Single step press completed.")


def print_detection(window: WindowInfo, result: DetectionResult, press_ms: float | None = None) -> None:
    print(f"Window: hwnd={window.hwnd} title={window.title!r}")
    print(f"Piece: {result.piece}  Target: {result.target}")
    print(f"Delta: dx={result.dx_px:.1f}px  dy={result.dy_px:.1f}px")
    print(
        f"Distance: effective={result.effective_distance_px:.1f}px  "
        f"screen={result.screen_distance_px:.1f}px  Confidence: {result.confidence:.2f}"
    )
    if press_ms is not None:
        print(f"Press: {press_ms:.0f}ms")
    print(f"Debug image: {result.debug_path}")


def record_one_manual_press() -> float:
    mouse = import_pynput_mouse()
    start_time: float | None = None
    duration: float | None = None

    def on_click(_: int, __: int, button: Any, pressed: bool) -> bool | None:
        nonlocal start_time, duration
        if button != mouse.Button.left:
            return None
        if pressed:
            start_time = time.perf_counter()
            return None
        if start_time is not None:
            duration = (time.perf_counter() - start_time) * 1000.0
            return False
        return None

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    if duration is None:
        raise JumpAutoError("No complete left-button press was recorded.")
    return duration


def print_press_model(config: dict[str, Any]) -> None:
    model = press_model_config(config)
    slope = model.get("slope_ms_per_px") or config.get("press_ms_per_px")
    if slope is None:
        print("Press model: not calibrated.")
        return
    print(
        "Press model: "
        f"slope={float(slope):.4f} ms/px  "
        f"y_weight={float(model.get('y_weight', 1.0)):.3f}  "
        f"offset={float(model.get('offset_ms', 0.0)):.1f}ms  "
        f"curve_points={len(model.get('curve_points', []))}  "
        f"segments={len(model.get('segment_corrections', []))}  "
        f"samples={int(model.get('sample_count', len(model.get('samples', []))))}"
    )
    if model.get("fit_rmse_ms") is not None:
        print(f"Fit RMSE: {float(model['fit_rmse_ms']):.1f}ms")


def record_auto_success_if_landed(
    config: dict[str, Any],
    config_path: Path,
    pending: dict[str, Any] | None,
    current_result: DetectionResult,
) -> bool:
    if pending is None:
        return False
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("enabled", True)):
        return False

    previous: DetectionResult = pending["result"]
    press_ms = float(pending["press_ms"])
    landing_error = math.dist(current_result.piece, previous.target)
    tolerance = float(tuning.get("landing_tolerance_px", 80))
    min_confidence = float(tuning.get("min_confidence", 0.60))
    if (
        landing_error > tolerance
        or previous.confidence < min_confidence
        or current_result.confidence < min_confidence
    ):
        return False

    model = press_model_config(config)
    precision_px = float(tuning.get("segment_precision_px", 8))
    if landing_error <= precision_px:
        frozen = mark_segment_precision_hit(config, previous.effective_distance_px, landing_error)
        clear_failure_caps_near_success(config, previous.effective_distance_px)
        update_piece_color_model(config, previous, "auto_previous")
        update_piece_color_model(config, current_result, "auto_current")
        if bool(tuning.get("save_every_success", True)):
            save_config(config_path, config)
        print(
            f"Auto-tune skipped: segment precise landing_error={landing_error:.1f}px "
            f"frozen={frozen}"
        )
        return True

    if segment_is_frozen(config, previous.effective_distance_px):
        unfrozen = maybe_unfreeze_segment_for_error(
            config,
            previous.effective_distance_px,
            landing_error,
        )
        if not unfrozen:
            update_piece_color_model(config, previous, "auto_previous")
            update_piece_color_model(config, current_result, "auto_current")
            if bool(tuning.get("save_every_success", True)):
                save_config(config_path, config)
            print(
                f"Auto-tune skipped: segment frozen landing_error={landing_error:.1f}px"
            )
            return True

    adjusted = center_adjusted_press_ms(previous, current_result, press_ms, config)
    signed_error = 0.0
    projection_ratio = 0.0
    sample_source = "auto"
    if adjusted is not None:
        adjusted_press_ms, signed_error, projection_ratio = adjusted
        sample_source = "auto_segment_adjusted"

    sample = calibration_sample_from_result(previous, press_ms)
    sample["source"] = sample_source
    sample["landing_error_px"] = landing_error
    sample["result_type"] = "auto_adjusted" if adjusted is not None else "auto_success"
    if adjusted is not None:
        sample["center_adjusted_press_ms"] = adjusted_press_ms
        sample["signed_landing_error_px"] = signed_error
        sample["projection_ratio"] = projection_ratio
    model.setdefault("samples", []).append(sample)
    fit_press_model(config)
    if adjusted is not None:
        record_segment_center_correction(
            config,
            previous.effective_distance_px,
            adjusted_press_ms - press_ms,
            signed_error,
            projection_ratio,
        )
    else:
        decay_segment_center_correction(config, previous.effective_distance_px)
    clear_failure_caps_near_success(config, previous.effective_distance_px)
    update_piece_color_model(config, previous, "auto_previous")
    update_piece_color_model(config, current_result, "auto_current")
    if bool(tuning.get("save_every_success", True)):
        save_config(config_path, config)
    print(
        f"Auto-tuned from previous jump: landing_error={landing_error:.1f}px "
        f"samples={len(model.get('samples', []))}"
    )
    if adjusted is not None:
        corrected_press = press_ms + segment_correction_ms(previous.effective_distance_px, model)
        print(
            f"Center correction: signed_error={signed_error:.1f}px "
            f"segment press {press_ms:.0f}ms -> {corrected_press:.0f}ms"
        )
    return True


def recognition_error_is_overlay(exc: RecognitionError) -> bool:
    return "game-over or modal overlay" in str(exc).lower()


def record_auto_failure_if_overlay(
    config: dict[str, Any],
    config_path: Path,
    pending: dict[str, Any] | None,
    exc: RecognitionError,
) -> bool:
    if pending is None or not recognition_error_is_overlay(exc):
        return False
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("enabled", True)) or not bool(tuning.get("failure_learning_enabled", True)):
        return False

    previous: DetectionResult = pending["result"]
    press_ms = float(pending["press_ms"])
    if previous.confidence < float(tuning.get("min_confidence", 0.60)):
        return False

    shrink_ratio = clamp(float(tuning.get("failure_shrink_ratio", 0.92)), 0.70, 0.98)
    press_cap = press_ms * shrink_ratio
    model = press_model_config(config)
    caps = model.setdefault("failure_caps", [])
    caps.append(
        {
            "timestamp": timestamp(),
            "distance_px": previous.effective_distance_px,
            "dx_px": previous.dx_px,
            "dy_px": previous.dy_px,
            "screen_distance_px": previous.screen_distance_px,
            "piece": [previous.piece[0], previous.piece[1]],
            "target": [previous.target[0], previous.target[1]],
            "failed_press_ms": press_ms,
            "press_cap_ms": press_cap,
            "landing_error_px": None,
            "confidence": previous.confidence,
            "result_type": "auto_failure",
            "reason": "overlay_after_jump",
        }
    )
    max_caps = int(model.get("max_failure_caps", 24))
    if len(caps) > max_caps:
        del caps[:-max_caps]
    if bool(tuning.get("save_every_success", True)):
        save_config(config_path, config)
    print(
        f"Auto-learned from failed jump: distance={previous.effective_distance_px:.1f}px "
        f"press {press_ms:.0f}ms -> cap {press_cap:.0f}ms"
    )
    return True


def run_calibration(args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> None:
    model = press_model_config(config)
    if args.reset_calibration:
        model["samples"] = []
        model["slope_ms_per_px"] = None
        model["offset_ms"] = 0.0
        model["fit_rmse_ms"] = None
        model["sample_count"] = 0
        config["press_ms_per_px"] = None

    sample_count = max(1, int(args.calibration_samples))
    accepted = 0
    for index in range(sample_count):
        window = locate_window(args.window_title, config)
        frame, _ = capture_window(window, config)
        label = "calibrate_preview" if sample_count == 1 else f"calibrate_{index + 1:02d}_preview"
        result = detect_jump(frame, config, args.debug_dir, label, save_mask=args.save_masks)
        print()
        print(f"Calibration sample {index + 1}/{sample_count}")
        print_detection(window, result)
        print()
        print("Open the debug image and confirm the markers are correct.")
        answer = input("Type 'y' to record one manual successful jump, or anything else to stop: ")
        if answer.strip().lower() != "y":
            if accepted == 0:
                raise JumpAutoError("Calibration cancelled.")
            break

        print("Now perform exactly one manual left-button long press in the WeChat game window.")
        print("The script will record the next complete left-button press/release.")
        duration_ms = record_one_manual_press()
        coefficient = duration_ms / max(1.0, result.effective_distance_px)
        print(f"Recorded manual press: {duration_ms:.0f}ms")
        print(f"Single-sample coefficient: {coefficient:.4f} ms/effective-px")
        success = input("Did that jump land correctly? Type 'y' to keep this sample: ")
        if success.strip().lower() != "y":
            print("Sample discarded.")
            continue

        sample = calibration_sample_from_result(result, duration_ms)
        model.setdefault("samples", []).append(sample)
        fit_press_model(config)
        update_piece_color_model(config, result, "manual_calibration")
        accepted += 1
        if args.window_title:
            config["window_title"] = args.window_title
        save_config(config_path, config)
        print_press_model(config)
        print(f"Saved config: {config_path}")
        time.sleep(0.35)

    if accepted == 0:
        raise JumpAutoError("Calibration was not saved.")


def start_hotkey_listener(stop_event: threading.Event, pause_event: threading.Event) -> Any:
    keyboard = import_pynput_keyboard()

    def on_press(key: Any) -> bool | None:
        if key == keyboard.Key.esc:
            stop_event.set()
            print("Esc received; stopping after current action.")
            return False
        if key == keyboard.Key.f8:
            if pause_event.is_set():
                pause_event.clear()
                print("Resumed.")
            else:
                pause_event.set()
                print("Paused. Press F8 to resume or Esc to exit.")
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def run_auto(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model = press_model_config(config)
    if model.get("slope_ms_per_px") is None and config.get("press_ms_per_px") is None:
        raise JumpAutoError("press_ms_per_px is not configured. Run --calibrate first.")

    stop_event = threading.Event()
    pause_event = threading.Event()
    listener = start_hotkey_listener(stop_event, pause_event)
    print("Auto mode started. Press F8 to pause/resume, Esc to exit.")
    jump_count = 0
    pending_jump: dict[str, Any] | None = None
    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.15)
                continue

            window = locate_window(args.window_title, config)
            frame, client_rect = capture_window(window, config)
            try:
                preview = detect_jump(frame, config, args.debug_dir, f"auto_{jump_count:04d}")
            except RecognitionError as exc:
                if not args.no_auto_tune:
                    record_auto_failure_if_overlay(config, args.config, pending_jump, exc)
                pending_jump = None
                print(f"Recognition failed; pausing. {exc}")
                pause_event.set()
                continue
            if args.no_auto_tune:
                pending_jump = None
            else:
                record_auto_success_if_landed(config, args.config, pending_jump, preview)
                pending_jump = None
            press_ms = calculate_press_ms(preview, config)

            if preview.confidence < float(config["confidence_threshold"]):
                print(
                    f"Low confidence {preview.confidence:.2f}; pausing. "
                    f"Debug image: {preview.debug_path}"
                )
                pause_event.set()
                continue

            print_detection(window, preview, press_ms=press_ms)
            press_in_window(window, client_rect, config, press_ms)
            pending_jump = {"result": preview, "press_ms": press_ms}
            jump_count += 1
            time.sleep(float(args.interval))
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        print(f"Auto mode ended. Completed jumps: {jump_count}")
