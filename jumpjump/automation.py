from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import (
    CURRENT_AUTO_FEEDBACK_VERSION,
    DEFAULT_CONFIG,
    auto_tuning_config,
    neural_press_model_config,
    press_model_config,
    save_config,
)
from .debug_artifacts import auto_capture_policy
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
    LandingMeasurement,
    annotate_stage_context,
    begin_stage_session,
    calculate_press_ms,
    calibration_sample_from_result,
    center_adjusted_press_ms,
    clear_failure_caps_near_success,
    decay_segment_center_correction,
    effective_distance_from_delta,
    fit_press_model,
    mark_segment_precision_hit,
    measure_landing,
    maybe_unfreeze_segment_for_error,
    physics_unit_press_ms,
    record_segment_center_correction,
    segment_is_frozen,
    stage_feedback_updates_base_curve,
    stage_press_context,
    update_stage_press_scale,
)
from .types import DependencyError, DetectionResult, JumpAutoError, RecognitionError, WindowInfo
from .utils import clamp, timestamp
from .neural_press_model import (
    NeuralPressPredictor,
    Prediction,
    coverage_key,
    legacy_prediction,
    online_guard_decision,
)
from .training_data import (
    CURRENT_LANDING_LABEL_METHOD,
    SampleIdIndex,
    append_sample,
    import_legacy_samples,
    jump_record,
    resolve_runtime_path,
)
from .vision import (
    detect_jump,
    save_detection_debug,
    save_recognition_failure_debug,
    update_piece_color_model,
)


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


def _root_window(hwnd: int, win32gui: Any, win32con: Any) -> int:
    ga_root = getattr(win32con, "GA_ROOT", 2)
    return int(win32gui.GetAncestor(hwnd, ga_root))


def focus_window(hwnd: int, timeout_s: float = 0.50) -> None:
    require_windows()
    win32gui, win32con, _ = import_win32()
    try:
        if not win32gui.IsWindow(hwnd):
            raise JumpAutoError("The target window no longer exists.")
        if not win32gui.IsWindowVisible(hwnd):
            raise JumpAutoError("The target window is not visible.")
        if win32gui.IsIconic(hwnd):
            raise JumpAutoError("The target window is minimized.")
        target_root = _root_window(hwnd, win32gui, win32con)
        win32gui.SetForegroundWindow(hwnd)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            foreground = win32gui.GetForegroundWindow()
            if foreground and _root_window(foreground, win32gui, win32con) == target_root:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.025)
    except JumpAutoError:
        raise
    except Exception as exc:
        raise JumpAutoError(f"Could not focus the target window: {exc}") from exc
    raise JumpAutoError("The target window did not become the foreground window.")


def click_point_for_rect(
    client_rect: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[int, int]:
    left, top, right, bottom = client_rect
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise JumpAutoError("The target window has an invalid client rectangle.")
    click_cfg = config["click_point"]
    x = int(left + width * float(click_cfg["x_ratio"]))
    y = int(top + height * float(click_cfg["y_ratio"]))
    return max(left, min(right - 1, x)), max(top, min(bottom - 1, y))


def verify_press_target(
    window: WindowInfo,
    captured_rect: tuple[int, int, int, int],
    click_point: tuple[int, int],
) -> None:
    require_windows()
    win32gui, win32con, _ = import_win32()
    try:
        if not win32gui.IsWindow(window.hwnd):
            raise JumpAutoError("The target window no longer exists.")
        if not win32gui.IsWindowVisible(window.hwnd):
            raise JumpAutoError("The target window is not visible.")
        if win32gui.IsIconic(window.hwnd):
            raise JumpAutoError("The target window is minimized.")
        current_rect = client_rect_on_screen(window.hwnd)
        if current_rect != captured_rect:
            raise JumpAutoError(
                "The target window moved or resized after capture; a new frame is required."
            )
        target_root = _root_window(window.hwnd, win32gui, win32con)
        foreground = win32gui.GetForegroundWindow()
        if not foreground or _root_window(foreground, win32gui, win32con) != target_root:
            raise JumpAutoError("The target window is no longer in the foreground.")
        if client_area_looks_obscured(window.hwnd, current_rect):
            raise JumpAutoError("The target window appears to be covered by another window.")
        point_window = win32gui.WindowFromPoint(click_point)
        if not point_window or _root_window(point_window, win32gui, win32con) != target_root:
            raise JumpAutoError("The configured click point is not owned by the target window.")
    except JumpAutoError:
        raise
    except Exception as exc:
        raise JumpAutoError(f"Could not verify the target window safely: {exc}") from exc


def _automation_abort_requested(
    stop_event: threading.Event | None,
    pause_event: threading.Event | None,
) -> bool:
    return bool(
        (stop_event is not None and stop_event.is_set())
        or (pause_event is not None and pause_event.is_set())
    )


def press_in_window(
    window: WindowInfo,
    client_rect: tuple[int, int, int, int],
    config: dict[str, Any],
    press_ms: float,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    action_lock: Any | None = None,
) -> None:
    focus_window(window.hwnd)
    click_point = click_point_for_rect(client_rect, config)
    verify_press_target(window, client_rect, click_point)
    if _automation_abort_requested(stop_event, pause_event):
        raise JumpAutoError("Automatic mode was paused or stopped before pressing.")

    pyautogui = import_pyautogui()
    pyautogui.PAUSE = 0.02
    pyautogui.FAILSAFE = True
    x, y = click_point
    pyautogui.moveTo(x, y, duration=0)
    verify_press_target(window, client_rect, click_point)
    if _automation_abort_requested(stop_event, pause_event):
        raise JumpAutoError("Automatic mode was paused or stopped before pressing.")

    mouse_down_attempted = False
    try:
        with (action_lock if action_lock is not None else nullcontext()):
            if _automation_abort_requested(stop_event, pause_event):
                raise JumpAutoError("Automatic mode was paused or stopped before pressing.")
            mouse_down_attempted = True
            pyautogui.mouseDown()
        time.sleep(max(0.0, press_ms / 1000.0))
    finally:
        if mouse_down_attempted:
            pyautogui.mouseUp()


def run_dry_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    window = locate_window(args.window_title, config)
    frame, _ = capture_window(window, config)
    result = detect_jump(frame, config, args.debug_dir, "dry_run", save_mask=args.save_masks)
    result = annotate_stage_context(result, config)
    print_detection(window, result)


def run_single_step(args: argparse.Namespace, config: dict[str, Any]) -> None:
    window = locate_window(args.window_title, config)
    frame, client_rect = capture_window(window, config)
    first_result = annotate_stage_context(
        detect_jump(frame, config, args.debug_dir, "single_step_preview"),
        config,
    )
    if first_result.stage_score_confirmed is False:
        time.sleep(
            float(
                auto_tuning_config(config).get(
                    "low_confidence_recheck_delay_s",
                    0.15,
                )
            )
        )
    second_frame, client_rect = capture_window(window, config)
    result = annotate_stage_context(
        detect_jump(second_frame, config, args.debug_dir, "single_step"),
        config,
    )
    if result.stage_score_confirmed is False:
        raise JumpAutoError(
            "Score transition is not confirmed by two captures; not pressing."
        )
    press_ms = calculate_press_ms(result, config)
    print_detection(window, result, press_ms=press_ms)
    if (
        not math.isfinite(result.confidence)
        or result.confidence < float(config["confidence_threshold"])
    ):
        raise JumpAutoError(
            f"Recognition confidence {result.confidence:.2f} is below threshold; not pressing."
        )
    press_in_window(window, client_rect, config, press_ms)
    print("Single step press completed.")


def print_detection(
    window: WindowInfo,
    result: DetectionResult,
    press_ms: float | None = None,
    prediction_source: str | None = None,
) -> None:
    print(f"Window: hwnd={window.hwnd} title={window.title!r}")
    print(f"Piece: {result.piece}  Target: {result.target}")
    print(f"Delta: dx={result.dx_px:.1f}px  dy={result.dy_px:.1f}px")
    print(
        f"Distance: effective={result.effective_distance_px:.1f}px  "
        f"screen={result.screen_distance_px:.1f}px  Confidence: {result.confidence:.2f}"
    )
    if result.game_score is not None or result.stage_bucket is not None:
        score_text = "?" if result.game_score is None else str(result.game_score)
        raw_score_text = (
            "?" if result.raw_game_score is None else str(result.raw_game_score)
        )
        stage_text = result.stage_bucket or "base"
        stage_scale = result.stage_press_scale or 1.0
        piece_scale = result.piece_scale_ratio or 1.0
        confirmation_text = (
            "confirmed"
            if result.stage_score_confirmed is not False
            else "provisional"
        )
        print(
            f"Stage: score={score_text} raw_score={raw_score_text} "
            f"state={confirmation_text} bucket={stage_text} "
            f"piece_scale={piece_scale:.3f} press_scale={stage_scale:.3f}"
        )
    if press_ms is not None:
        source = f" source={prediction_source}" if prediction_source else ""
        print(f"Press: {press_ms:.0f}ms{source}")
    if result.debug_path is not None:
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
    slope_text = "n/a" if slope is None else f"{float(slope):.4f} ms/px"
    fixed_head = model.get("physics_head_diameter_px")
    if fixed_head is None:
        head_text = (
            f"piece_width*{float(model.get('physics_piece_width_multiplier', 1.15)):.2f} "
            f"fallback={float(model.get('physics_default_head_diameter_px', 80.0)):.1f}px"
        )
    else:
        head_text = f"fixed={float(fixed_head):.1f}px"
    print(
        "Press model: "
        f"base={model.get('base_algorithm', 'physics')}  "
        f"physics_coeff={float(model.get('physics_press_coefficient', 1.392)):.3f}  "
        f"head={head_text}  "
        f"slope={slope_text}  "
        f"y_weight={float(model.get('y_weight', 1.0)):.3f}  "
        f"offset={float(model.get('offset_ms', 0.0)):.1f}ms  "
        f"curve_points={len(model.get('curve_points', []))}  "
        f"segments={len(model.get('segment_corrections', []))}  "
        f"samples={int(model.get('sample_count', len(model.get('samples', []))))}"
    )
    if model.get("fit_rmse_ms") is not None:
        print(f"Fit RMSE: {float(model['fit_rmse_ms']):.1f}ms")


def confidence_run_decision(config: dict[str, Any], confidence: float) -> tuple[bool, bool, float, float]:
    threshold = max(0.0, float(config["confidence_threshold"]))
    tuning = auto_tuning_config(config)
    run_floor = clamp(float(tuning.get("run_confidence_floor", 0.35)), 0.0, threshold)
    if not math.isfinite(confidence):
        return True, True, run_floor, threshold
    should_pause = confidence < run_floor
    is_low_confidence = confidence < threshold
    return should_pause, is_low_confidence, run_floor, threshold


def save_detection_result_debug(
    frame: Any,
    result: DetectionResult,
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
) -> DetectionResult:
    if result.debug_path is not None and result.debug_path.exists():
        return result
    debug_path = save_detection_debug(frame, result, config, debug_dir, label)
    return replace(result, debug_path=debug_path)


@dataclass(frozen=True)
class DetectionRecheckOutcome:
    result: DetectionResult | None
    client_rect: tuple[int, int, int, int]
    reason: str
    landing_result: DetectionResult | None = None

    def __iter__(self):
        # Preserve the existing three-value unpacking API while exposing the
        # independently verified landing observation to the auto loop.
        yield self.result
        yield self.client_rect
        yield self.reason


def landing_recheck_consistency(
    first: DetectionResult,
    second: DetectionResult,
    first_client_rect: tuple[int, int, int, int],
    second_client_rect: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[bool, str]:
    if first_client_rect != second_client_rect:
        return False, "the window moved or resized between captures"
    if first.crop_rect != second.crop_rect:
        return False, "the game crop changed between captures"
    tuning = auto_tuning_config(config)
    crop_width = max(1, first.crop_rect[2] - first.crop_rect[0])
    piece_tolerance = max(
        6.0,
        crop_width * float(tuning.get("recheck_piece_tolerance_ratio", 0.015)),
    )
    piece_shift = math.dist(first.piece, second.piece)
    if piece_shift > piece_tolerance:
        return False, f"piece moved {piece_shift:.1f}px (limit {piece_tolerance:.1f}px)"
    return True, "landing detections agree"


def detection_recheck_consistency(
    first: DetectionResult,
    second: DetectionResult,
    first_client_rect: tuple[int, int, int, int],
    second_client_rect: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[bool, str]:
    landing_consistent, landing_reason = landing_recheck_consistency(
        first,
        second,
        first_client_rect,
        second_client_rect,
        config,
    )
    if not landing_consistent:
        return False, landing_reason
    if not math.isfinite(first.dx_px) or not math.isfinite(second.dx_px):
        return False, "the detected jump direction was not finite"
    if first.dx_px == 0.0 or second.dx_px == 0.0 or first.dx_px * second.dx_px <= 0.0:
        return False, "the detected jump direction changed"

    tuning = auto_tuning_config(config)
    crop_width = max(1, first.crop_rect[2] - first.crop_rect[0])
    target_tolerance = max(
        10.0,
        crop_width * float(tuning.get("recheck_target_tolerance_ratio", 0.025)),
    )
    target_shift = math.dist(first.target, second.target)
    if target_shift > target_tolerance:
        return False, f"target moved {target_shift:.1f}px (limit {target_tolerance:.1f}px)"
    return True, "detections agree"


def recheck_low_confidence_detection(
    window: WindowInfo,
    first_frame: Any,
    first_client_rect: tuple[int, int, int, int],
    first_result: DetectionResult,
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
    stop_event: threading.Event,
    pause_event: threading.Event,
    landing_hint: DetectionResult | None = None,
) -> DetectionRecheckOutcome:
    policy = auto_capture_policy(config)
    first_saved = first_result
    if policy in {"failures_and_rechecks", "all"}:
        first_saved = save_detection_result_debug(
            first_frame,
            first_result,
            config,
            debug_dir,
            f"{label}_recheck_first",
        )

    delay_s = float(
        auto_tuning_config(config).get("low_confidence_recheck_delay_s", 0.15)
    )
    if stop_event.wait(delay_s) or pause_event.is_set():
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            "automatic mode was paused or stopped during recheck",
        )

    try:
        second_frame, second_client_rect = capture_window(window, config)
    except DependencyError:
        raise
    except Exception as exc:
        save_detection_result_debug(
            first_frame,
            first_saved,
            config,
            debug_dir,
            f"{label}_recheck_first_failed",
        )
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            f"recheck capture failed: {exc}",
        )

    if second_client_rect != first_client_rect:
        first_saved = save_detection_result_debug(
            first_frame,
            first_saved,
            config,
            debug_dir,
            f"{label}_recheck_first_failed",
        )
        height, width = second_frame.shape[:2]
        second_debug = save_recognition_failure_debug(
            second_frame,
            (0, 0, width, height),
            config,
            debug_dir,
            f"{label}_recheck_second",
            "The target window moved or resized during confidence recheck.",
        )
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            f"window changed during recheck; debug images: {first_saved.debug_path}, {second_debug}",
        )

    try:
        second_result = detect_jump(
            second_frame,
            config,
            debug_dir,
            f"{label}_recheck_second",
            save_debug=policy in {"failures_and_rechecks", "all"},
            landing_hint=landing_hint,
        )
    except RecognitionError as exc:
        first_saved = save_detection_result_debug(
            first_frame,
            first_saved,
            config,
            debug_dir,
            f"{label}_recheck_first_failed",
        )
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            f"recheck recognition failed: {exc}; first: {first_saved.debug_path}",
        )
    except DependencyError:
        raise
    except Exception as exc:
        first_saved = save_detection_result_debug(
            first_frame,
            first_saved,
            config,
            debug_dir,
            f"{label}_recheck_first_failed",
        )
        second_debug: Path | None = None
        try:
            height, width = second_frame.shape[:2]
            second_debug = save_recognition_failure_debug(
                second_frame,
                (0, 0, width, height),
                config,
                debug_dir,
                f"{label}_recheck_second",
                f"Unexpected recognition error during confidence recheck: {exc}",
            )
        except Exception:
            pass
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            f"recheck recognition failed unexpectedly: {exc}; "
            f"debug images: {first_saved.debug_path}, {second_debug}",
        )

    threshold = float(config["confidence_threshold"])
    landing_consistent, _ = landing_recheck_consistency(
        first_result,
        second_result,
        first_client_rect,
        second_client_rect,
        config,
    )
    landing_result = second_result if landing_consistent else None
    consistent, consistency_reason = detection_recheck_consistency(
        first_result,
        second_result,
        first_client_rect,
        second_client_rect,
        config,
    )
    recovered = math.isfinite(second_result.confidence) and second_result.confidence >= threshold
    if not recovered or not consistent:
        first_saved = save_detection_result_debug(
            first_frame,
            first_saved,
            config,
            debug_dir,
            f"{label}_recheck_first_rejected",
        )
        second_saved = save_detection_result_debug(
            second_frame,
            second_result,
            config,
            debug_dir,
            f"{label}_recheck_second_rejected",
        )
        confidence_reason = (
            f"confidence {second_result.confidence:.2f} did not recover to {threshold:.2f}"
            if not recovered
            else consistency_reason
        )
        return DetectionRecheckOutcome(
            None,
            first_client_rect,
            f"{confidence_reason}; debug images: {first_saved.debug_path}, {second_saved.debug_path}",
            landing_result,
        )
    return DetectionRecheckOutcome(
        second_result,
        second_client_rect,
        consistency_reason,
        landing_result,
    )


def recognition_failure_pause_status(config: dict[str, Any], failure_count: int) -> tuple[bool, int]:
    tuning = auto_tuning_config(config)
    max_failures = max(1, int(tuning.get("max_recognition_failures_before_pause", 3)))
    return failure_count >= max_failures, max_failures


def result_is_good_learning_candidate(config: dict[str, Any], result: DetectionResult) -> bool:
    tuning = auto_tuning_config(config)
    return result.confidence >= float(tuning.get("min_confidence", 0.60))


def _dataset_path(config: dict[str, Any], config_path: Path) -> Path:
    settings = neural_press_model_config(config)
    return resolve_runtime_path(config_path, str(settings["dataset_path"]))


def _append_neural_sample(
    path: Path,
    record: dict[str, Any],
    sample_index: SampleIdIndex | None = None,
) -> bool:
    try:
        if sample_index is not None:
            return sample_index.append(record)
        return append_sample(path, record)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Could not append neural training sample; continuing safely. {exc}")
        return False


def record_neural_success_sample(
    config: dict[str, Any],
    config_path: Path,
    pending: dict[str, Any] | None,
    current_result: DetectionResult,
    measurement: LandingMeasurement | None = None,
    *,
    sample_index: SampleIdIndex | None = None,
) -> bool:
    if pending is None:
        return False
    previous: DetectionResult = pending["result"]
    if (
        previous.stage_bucket is None
        or previous.stage_press_scale is None
        or previous.piece_scale_ratio is None
        or previous.stage_score_confirmed is None
    ):
        previous = annotate_stage_context(previous, config)
    executed = float(pending["press_ms"])
    physics_unit = physics_unit_press_ms(previous, config)
    effective_coefficient = (
        executed / physics_unit
        if physics_unit is not None and physics_unit > 0
        else None
    )
    tuning = auto_tuning_config(config)
    measurement = measurement or measure_landing(previous, current_result, config)
    min_confidence = float(tuning.get("min_confidence", 0.60))
    platform_min_confidence = float(tuning.get("landing_platform_min_confidence", 0.55))
    tolerance = float(tuning.get("landing_tolerance_px", 80))
    precision = float(tuning.get("segment_precision_px", 3))
    deadzone = float(tuning.get("center_deadzone_px", precision))
    confidence_ok = (
        previous.confidence >= min_confidence
        and previous.stage_score_confirmed is not False
        and measurement is not None
        and measurement.label_confidence >= platform_min_confidence
    )
    target_press: float | None = None
    trainable = False
    result_type = "auto_unlabelled_platform" if measurement is None else "auto_low_confidence"
    if confidence_ok and measurement.landing_error_px <= max(precision, deadzone):
        target_press = executed
        trainable = True
        result_type = "auto_precise" if measurement.landing_error_px <= precision else "auto_deadzone"
    elif confidence_ok and measurement.landing_error_px > tolerance:
        result_type = "auto_out_of_tolerance"
    elif confidence_ok and measurement.projection_ratio < float(
        tuning.get("center_projection_min_ratio", 0.45)
    ):
        result_type = "auto_low_projection"
    elif confidence_ok:
        adjusted = center_adjusted_press_ms(
            previous,
            current_result,
            executed,
            config,
            measurement,
        )
        if adjusted is not None:
            target_press = adjusted[0]
            trainable = True
            result_type = "auto_adjusted"
    record = jump_record(
        previous,
        session_id=str(pending.get("session_id", "unknown")),
        viewport_size=tuple(pending.get("viewport_size", (1080, 1920))),
        legacy_press_ms=float(pending.get("legacy_press_ms", executed)),
        executed_press_ms=executed,
        prediction_source=str(pending.get("prediction_source", "legacy")),
        prediction_model_id=pending.get("prediction_model_id"),
        result_type=result_type,
        landing_error_px=measurement.landing_error_px if measurement else None,
        target_press_ms=target_press,
        signed_landing_error_px=measurement.signed_error_px if measurement else None,
        projection_ratio=measurement.projection_ratio if measurement else None,
        landing_label_method=CURRENT_LANDING_LABEL_METHOD if measurement else None,
        landing_label_confidence=measurement.label_confidence if measurement else None,
        landing_label_source=measurement.label_source if measurement else None,
        landing_reference=measurement.reference_point if measurement else None,
        landing_platform_bbox=current_result.landing_platform_bbox,
        trainable=trainable,
        jump_index=pending.get("jump_index"),
        physics_unit_press_ms=physics_unit,
        effective_press_coefficient=effective_coefficient,
    )
    return _append_neural_sample(
        _dataset_path(config, config_path),
        record,
        sample_index,
    )


def record_neural_failure_sample(
    config: dict[str, Any],
    config_path: Path,
    pending: dict[str, Any] | None,
    reason: str,
    *,
    sample_index: SampleIdIndex | None = None,
) -> bool:
    if pending is None:
        return False
    previous: DetectionResult = pending["result"]
    if (
        previous.stage_bucket is None
        or previous.stage_press_scale is None
        or previous.piece_scale_ratio is None
        or previous.stage_score_confirmed is None
    ):
        previous = annotate_stage_context(previous, config)
    executed = float(pending["press_ms"])
    physics_unit = physics_unit_press_ms(previous, config)
    effective_coefficient = (
        executed / physics_unit
        if physics_unit is not None and physics_unit > 0
        else None
    )
    record = jump_record(
        previous,
        session_id=str(pending.get("session_id", "unknown")),
        viewport_size=tuple(pending.get("viewport_size", (1080, 1920))),
        legacy_press_ms=float(pending.get("legacy_press_ms", executed)),
        executed_press_ms=executed,
        prediction_source=str(pending.get("prediction_source", "legacy")),
        prediction_model_id=pending.get("prediction_model_id"),
        result_type="auto_failure",
        trainable=False,
        reason=reason,
        jump_index=pending.get("jump_index"),
        physics_unit_press_ms=physics_unit,
        effective_press_coefficient=effective_coefficient,
    )
    return _append_neural_sample(
        _dataset_path(config, config_path),
        record,
        sample_index,
    )


def load_neural_predictor(
    config: dict[str, Any],
    config_path: Path,
    disabled: bool = False,
) -> NeuralPressPredictor | None:
    settings = neural_press_model_config(config)
    if disabled or not bool(settings.get("enabled", False)):
        return None
    model_path = resolve_runtime_path(config_path, str(settings["model_path"]))
    metadata_path = resolve_runtime_path(config_path, str(settings["metadata_path"]))
    try:
        predictor = NeuralPressPredictor.load(model_path, metadata_path)
    except (DependencyError, JumpAutoError) as exc:
        print(f"Neural press model unavailable; using legacy model. {exc}")
        return None
    print(f"Neural press model loaded on {predictor.device}.")
    return predictor


def predict_press(
    result: DetectionResult,
    config: dict[str, Any],
    viewport_size: tuple[int, int],
    predictor: NeuralPressPredictor | None,
) -> Prediction:
    legacy_ms = calculate_press_ms(result, config)
    if predictor is None or result.stage_score_confirmed is False:
        return legacy_prediction(legacy_ms)
    try:
        return predictor.predict(result, viewport_size, legacy_ms, config)
    except Exception as exc:
        print(f"Neural prediction failed; using legacy model for this jump. {exc}")
        return legacy_prediction(legacy_ms)


def record_auto_success_if_landed(
    config: dict[str, Any],
    config_path: Path,
    pending: dict[str, Any] | None,
    current_result: DetectionResult,
    measurement: LandingMeasurement | None = None,
) -> bool:
    if pending is None:
        return False
    tuning = auto_tuning_config(config)
    if not bool(tuning.get("enabled", True)):
        return False
    prediction_source = str(pending.get("prediction_source", "legacy"))

    previous: DetectionResult = pending["result"]
    press_ms = float(pending["press_ms"])
    measurement = measurement or measure_landing(
        previous,
        current_result,
        config,
        allow_temporal_fallback=True,
    )
    if measurement is None:
        print(
            "Auto-tune skipped: no trustworthy visible or temporal-horizontal landing"
        )
        return False
    landing_error = measurement.landing_error_px
    tolerance = float(tuning.get("landing_tolerance_px", 80))
    min_confidence = float(tuning.get("min_confidence", 0.60))
    platform_min_confidence = float(tuning.get("landing_platform_min_confidence", 0.55))
    if (
        landing_error > tolerance
        or previous.confidence < min_confidence
        or measurement.label_confidence < platform_min_confidence
    ):
        print(
            f"Auto-tune skipped: landing_error={landing_error:.1f}px (tolerance={tolerance:.0f}) "
            f"prior_conf={previous.confidence:.2f} "
            f"platform_conf={measurement.label_confidence:.2f}"
        )
        return False

    if prediction_source != "legacy":
        update_piece_color_model(config, previous, "auto_previous")
        update_piece_color_model(config, current_result, "auto_current")
        if bool(tuning.get("save_every_success", True)):
            save_config(config_path, config)
        print(
            f"Legacy auto-tune unchanged: prediction source is {prediction_source}; "
            "legacy feedback requires a legacy-executed jump."
        )
        return True

    model = press_model_config(config)
    stage_context = stage_press_context(previous, config, create=True)
    if not stage_context.score_confirmed:
        print(
            "Auto-tune skipped: score transition is awaiting a second frame; "
            f"raw_score={previous.raw_game_score!r} stage={stage_context.bucket}"
        )
        return False
    segment_distance = previous.screen_distance_px
    precision_px = float(tuning.get("segment_precision_px", 3))
    deadzone = float(tuning.get("center_deadzone_px", precision_px))
    if landing_error <= max(precision_px, deadzone):
        frozen = False
        if landing_error <= precision_px:
            frozen = mark_segment_precision_hit(
                config,
                segment_distance,
                landing_error,
                stage_context.bucket,
            )
        clear_failure_caps_near_success(config, previous.effective_distance_px)
        update_piece_color_model(config, previous, "auto_previous")
        update_piece_color_model(config, current_result, "auto_current")
        if bool(tuning.get("save_every_success", True)):
            save_config(config_path, config)
        print(
            f"Auto-tune stable: segment precise landing_error={landing_error:.1f}px "
            f"frozen={frozen} prior_conf={previous.confidence:.2f} "
            f"next_target_conf={current_result.confidence:.2f} "
            f"landing_source={measurement.label_source} stage={stage_context.bucket}"
        )
        return True

    if measurement.projection_ratio < float(tuning.get("center_projection_min_ratio", 0.45)):
        proj_min = float(tuning.get("center_projection_min_ratio", 0.45))
        print(
            f"Auto-tune skipped: sideways drift, projection_ratio={measurement.projection_ratio:.2f} "
            f"< {proj_min:.2f}"
        )
        return False

    if segment_is_frozen(config, segment_distance, stage_context.bucket):
        unfrozen = maybe_unfreeze_segment_for_error(
            config,
            segment_distance,
            landing_error,
            stage_context.bucket,
        )
        if not unfrozen:
            update_piece_color_model(config, previous, "auto_previous")
            update_piece_color_model(config, current_result, "auto_current")
            if bool(tuning.get("save_every_success", True)):
                save_config(config_path, config)
            print(
                f"Auto-tune stable: segment frozen landing_error={landing_error:.1f}px"
            )
            return True

    adjusted = center_adjusted_press_ms(previous, current_result, press_ms, config, measurement)
    signed_error = 0.0
    projection_ratio = 0.0
    sample_source = "auto"
    if adjusted is not None:
        adjusted_press_ms, signed_error, projection_ratio = adjusted
        sample_source = "auto_segment_adjusted"
    else:
        deadzone_cfg = float(tuning.get("center_deadzone_px", 8))
        print(
            f"Auto-tune skipped: adjustment not computed, "
            f"error={landing_error:.1f}px deadzone={deadzone_cfg:.1f} "
            f"proj={measurement.projection_ratio:.2f}"
        )
        return False

    updated_stage = update_stage_press_scale(
        config,
        previous,
        press_ms,
        adjusted_press_ms,
    )
    sample = calibration_sample_from_result(
        previous,
        press_ms,
        stage_press_scale=updated_stage.press_scale,
        stage_bucket=updated_stage.bucket,
        piece_scale_ratio=updated_stage.piece_scale_ratio,
    )
    sample["source"] = sample_source
    sample["feedback_version"] = CURRENT_AUTO_FEEDBACK_VERSION
    sample["landing_error_px"] = landing_error
    sample["landing_label_source"] = measurement.label_source
    sample["result_type"] = "auto_adjusted" if adjusted is not None else "auto_success"
    sample["training_press_ms"] = adjusted_press_ms if adjusted is not None else press_ms
    if adjusted is not None:
        sample["center_adjusted_press_ms"] = adjusted_press_ms
        sample["signed_landing_error_px"] = signed_error
        sample["projection_ratio"] = projection_ratio
    if stage_feedback_updates_base_curve(updated_stage):
        sample["base_curve_eligible"] = True
        model.setdefault("samples", []).append(sample)
        fit_press_model(config)
    updated_effective_distance = effective_distance_from_delta(
        previous.dx_px,
        previous.dy_px,
        config,
    )
    updated_previous = replace(
        previous,
        effective_distance_px=updated_effective_distance,
        distance_px=updated_effective_distance,
    )
    if adjusted is not None:
        # The anchored base curve (only at the initial stage) and the current
        # stage multiplier have consumed their permitted parts of this
        # feedback.  Learn only the remaining same-stage local residual.
        next_press_before_segment_update = calculate_press_ms(updated_previous, config)
        record_segment_center_correction(
            config,
            segment_distance,
            adjusted_press_ms - next_press_before_segment_update,
            signed_error,
            projection_ratio,
            reference_press_ms=next_press_before_segment_update,
            stage_bucket=updated_stage.bucket,
        )
    else:
        decay_segment_center_correction(
            config,
            segment_distance,
            updated_stage.bucket,
        )
    clear_failure_caps_near_success(config, updated_effective_distance)
    update_piece_color_model(config, previous, "auto_previous")
    update_piece_color_model(config, current_result, "auto_current")
    if bool(tuning.get("save_every_success", True)):
        save_config(config_path, config)
    print(
        f"Auto-tuned from previous jump: landing_error={landing_error:.1f}px "
        f"samples={len(model.get('samples', []))} "
        f"landing_source={measurement.label_source} "
        f"stage={updated_stage.bucket} stage_scale={updated_stage.press_scale:.3f}"
    )
    if adjusted is not None:
        corrected_press = calculate_press_ms(updated_previous, config)
        print(
            f"Center correction: signed_effective_error={signed_error:.1f}effective_px "
            f"signed_screen_error={measurement.signed_screen_error_px:.1f}px "
            f"feedback_target={adjusted_press_ms:.0f}ms "
            f"next segment press={corrected_press:.0f}ms"
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
        result = annotate_stage_context(
            detect_jump(frame, config, args.debug_dir, label, save_mask=args.save_masks),
            config,
        )
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
        calibration_stage = stage_press_context(result, config, create=True)
        base_curve_eligible = stage_feedback_updates_base_curve(calibration_stage)
        sample["base_curve_eligible"] = base_curve_eligible
        model.setdefault("samples", []).append(sample)
        if base_curve_eligible:
            fit_press_model(config)
        else:
            predicted_press_ms = calculate_press_ms(result, config)
            update_stage_press_scale(
                config,
                result,
                predicted_press_ms,
                duration_ms,
            )
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


def start_hotkey_listener(
    stop_event: threading.Event,
    pause_event: threading.Event,
    action_lock: Any | None = None,
) -> Any:
    keyboard = import_pynput_keyboard()

    def on_press(key: Any) -> bool | None:
        if key == keyboard.Key.esc:
            with (action_lock if action_lock is not None else nullcontext()):
                stop_event.set()
            print("Esc received; stopping after current action.")
            return False
        if key == keyboard.Key.f8:
            with (action_lock if action_lock is not None else nullcontext()):
                resumed = pause_event.is_set()
                if resumed:
                    pause_event.clear()
                else:
                    pause_event.set()
            if resumed:
                print("Resumed.")
            else:
                print("Paused. Press F8 to resume or Esc to exit.")
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def run_auto(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model = press_model_config(config)
    if (
        str(model.get("base_algorithm", "physics")).lower() == "learned"
        and model.get("slope_ms_per_px") is None
        and config.get("press_ms_per_px") is None
    ):
        if model.get("samples"):
            fit_press_model(config)
            model = press_model_config(config)
        if (
            model.get("slope_ms_per_px") is None
            and config.get("press_ms_per_px") is None
        ):
            print(
                "Learned base fit is unavailable after safe migration; "
                "using the configured physics reference until new base-stage feedback arrives."
            )

    # A process restart is not a new game.  Keep learned score mappings and
    # local corrections; score 0 followed by a distinct 0/1 frame performs
    # the actual new-game transition.
    begin_stage_session(config)

    dataset_path = _dataset_path(config, args.config)
    sample_index: SampleIdIndex | None = None
    try:
        sample_index = SampleIdIndex(dataset_path)
        imported = import_legacy_samples(
            dataset_path,
            model.get("samples", []),
            sample_index=sample_index,
        )
        if imported:
            print(f"Imported {imported} legacy calibration samples into {dataset_path}.")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Could not import legacy neural samples; continuing safely. {exc}")
    session_id = timestamp()
    predictor = load_neural_predictor(
        config,
        args.config,
        disabled=bool(getattr(args, "no_neural_press", False)),
    )
    stop_event = threading.Event()
    pause_event = threading.Event()
    action_lock = threading.Lock()
    listener = start_hotkey_listener(stop_event, pause_event, action_lock)
    print("Auto mode started. Press F8 to pause/resume, Esc to exit.")
    jump_count = 0
    recognition_failures = 0
    pending_jump: dict[str, Any] | None = None
    neural_landing_errors: list[dict[str, Any]] = []
    debug_policy = auto_capture_policy(config)

    def settle_pending_feedback(current_result: DetectionResult) -> None:
        nonlocal pending_jump, predictor
        if pending_jump is None:
            return
        landing_measurement = measure_landing(
            pending_jump["result"],
            current_result,
            config,
            allow_temporal_fallback=True,
        )
        record_neural_success_sample(
            config,
            args.config,
            pending_jump,
            current_result,
            landing_measurement,
            sample_index=sample_index,
        )
        if (
            predictor is not None
            and pending_jump.get("prediction_source") == "neural"
            and landing_measurement is not None
            and landing_measurement.label_confidence
            >= float(
                auto_tuning_config(config).get(
                    "landing_platform_min_confidence",
                    0.55,
                )
            )
        ):
            neural_landing_errors.append(
                {
                    "landing_error_px": landing_measurement.landing_error_px,
                    "coverage_key": coverage_key(
                        {
                            "dx_px": pending_jump["result"].dx_px,
                            "effective_distance_px": pending_jump[
                                "result"
                            ].effective_distance_px,
                        },
                        float(predictor.metadata["coverage_bin_size_px"]),
                    ),
                }
            )
            disable_neural, guard = online_guard_decision(
                neural_landing_errors,
                predictor.metadata,
                config,
            )
            if disable_neural:
                settings = neural_press_model_config(config)
                settings["enabled"] = False
                settings.setdefault("training_metrics", {})[
                    "online_guard_last"
                ] = guard
                try:
                    save_config(args.config, config)
                except Exception as exc:
                    print(f"Could not persist neural safety shutdown: {exc}")
                predictor = None
                print(
                    "Neural model disabled by online safety guard: "
                    f"median={guard['median_error_px']:.1f}px "
                    f"success={guard['success_rate']:.1%} "
                    f"(floor {guard['success_rate_floor']:.1%})."
                )
        if not args.no_auto_tune:
            record_auto_success_if_landed(
                config,
                args.config,
                pending_jump,
                current_result,
                landing_measurement,
            )
        pending_jump = None

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.15)
                continue

            window = locate_window(args.window_title, config)
            frame, client_rect = capture_window(window, config)
            label = f"auto_{jump_count:04d}"
            try:
                preview = detect_jump(
                    frame,
                    config,
                    args.debug_dir,
                    label,
                    save_debug=debug_policy == "all",
                    landing_hint=(
                        pending_jump["result"]
                        if pending_jump is not None
                        else None
                    ),
                )
            except RecognitionError as exc:
                record_neural_failure_sample(
                    config,
                    args.config,
                    pending_jump,
                    str(exc),
                    sample_index=sample_index,
                )
                if not args.no_auto_tune:
                    record_auto_failure_if_overlay(config, args.config, pending_jump, exc)
                pending_jump = None
                recognition_failures += 1
                should_pause, max_failures = recognition_failure_pause_status(
                    config,
                    recognition_failures,
                )
                print(
                    f"Recognition failed ({recognition_failures}/{max_failures}). {exc}"
                )
                if should_pause:
                    print("Recognition failure limit reached; pausing.")
                    pause_event.set()
                else:
                    time.sleep(float(args.interval))
                continue
            recognition_failures = 0

            should_pause, is_low_confidence, run_floor, threshold = confidence_run_decision(
                config,
                preview.confidence,
            )
            if is_low_confidence:
                recheck = recheck_low_confidence_detection(
                    window,
                    frame,
                    client_rect,
                    preview,
                    config,
                    args.debug_dir,
                    label,
                    stop_event,
                    pause_event,
                    pending_jump["result"] if pending_jump is not None else None,
                )
                verified, verified_rect, recheck_reason = recheck
                if verified is None:
                    landing_result = getattr(recheck, "landing_result", None)
                    if pending_jump is not None:
                        if landing_result is not None:
                            settle_pending_feedback(landing_result)
                        else:
                            record_neural_failure_sample(
                                config,
                                args.config,
                                pending_jump,
                                f"landing_recheck_rejected: {recheck_reason}",
                                sample_index=sample_index,
                            )
                            pending_jump = None
                    preview = save_detection_result_debug(
                        frame,
                        preview,
                        config,
                        args.debug_dir,
                        f"{label}_low_confidence",
                    )
                    confidence_context = (
                        f"below run floor {run_floor:.2f}"
                        if should_pause
                        else "could not be verified"
                    )
                    print(
                        f"Low confidence {preview.confidence:.2f} {confidence_context}; "
                        f"pausing. {recheck_reason}. Debug image: {preview.debug_path}"
                    )
                    pause_event.set()
                    continue
                preview = verified
                client_rect = verified_rect
                print(
                    f"Confidence recheck recovered to {preview.confidence:.2f} at threshold "
                    f"{threshold:.2f}; {recheck_reason}."
                )

            # The next target is now either high-confidence or independently
            # rechecked.  Settle the previous landing with this verified piece
            # observation before calculating another press.
            settle_pending_feedback(preview)
            preview = annotate_stage_context(preview, config)
            if preview.stage_score_confirmed is False:
                print(
                    "Score transition is awaiting a second capture; "
                    f"raw_score={preview.raw_game_score!r}. No press was sent."
                )
                time.sleep(
                    float(
                        auto_tuning_config(config).get(
                            "low_confidence_recheck_delay_s",
                            0.15,
                        )
                    )
                )
                continue

            viewport_size = (
                max(1, client_rect[2] - client_rect[0]),
                max(1, client_rect[3] - client_rect[1]),
            )
            prediction = predict_press(preview, config, viewport_size, predictor)
            press_ms = prediction.press_ms
            print_detection(
                window,
                preview,
                press_ms=press_ms,
                prediction_source=prediction.source,
            )
            try:
                press_in_window(
                    window,
                    client_rect,
                    config,
                    press_ms,
                    stop_event=stop_event,
                    pause_event=pause_event,
                    action_lock=action_lock,
                )
            except DependencyError:
                raise
            except JumpAutoError as exc:
                pending_jump = None
                print(f"Press cancelled for safety; pausing. {exc}")
                pause_event.set()
                continue
            if result_is_good_learning_candidate(config, preview):
                pending_jump = {
                    "result": preview,
                    "press_ms": press_ms,
                    "legacy_press_ms": prediction.legacy_press_ms,
                    "prediction_source": prediction.source,
                    "prediction_model_id": prediction.model_id,
                    "viewport_size": viewport_size,
                    "session_id": session_id,
                    "jump_index": jump_count,
                }
            else:
                min_conf = float(auto_tuning_config(config).get("min_confidence", 0.60))
                print(
                    f"Auto-tune deferred: press-frame confidence {preview.confidence:.2f} < {min_conf:.2f}"
                    f" (landing will not be measured for auto-tune)"
                )
                pending_jump = None
            jump_count += 1
            time.sleep(float(args.interval))
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        print(f"Auto mode ended. Completed jumps: {jump_count}")
