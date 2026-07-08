#!/usr/bin/env python3
"""
Windows desktop automation helper for the WeChat Jump mini game.

The script captures the WeChat game window, detects the current piece and the
next target, then converts pixel distance to mouse press duration. It is meant
for local learning/testing and deliberately contains no anti-detection logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "jump_config.json"
DEFAULT_DEBUG_DIR = APP_DIR / "debug"


DEFAULT_CONFIG: dict[str, Any] = {
    "window_title": "",
    "window_keywords": ["跳一跳", "微信", "WeChat"],
    "min_client_width": 280,
    "min_client_height": 420,
    "crop": {
        "left_ratio": 0.0,
        "right_ratio": 1.0,
        "top_ratio": 0.10,
        "bottom_ratio": 0.92,
    },
    "press_ms_per_px": None,
    "min_press_ms": 180,
    "max_press_ms": 1800,
    "confidence_threshold": 0.45,
    "click_point": {
        "x_ratio": 0.50,
        "y_ratio": 0.82,
    },
    "piece": {
        "hsv_lower": [95, 35, 25],
        "hsv_upper": [140, 255, 230],
        "search_top_ratio": 0.35,
        "search_bottom_ratio": 0.98,
        "min_area": 120,
        "max_area": 16000,
        "foot_offset_px": 8,
    },
    "target": {
        "diff_threshold": 16,
        "search_top_ratio": 0.05,
        "search_bottom_extra_ratio": 0.10,
        "side_gap_ratio": 0.06,
        "exclude_piece_pad_px": 28,
        "min_area": 180,
        "max_area_ratio": 0.20,
        "min_width": 18,
        "min_height": 10,
        "min_distance_ratio": 0.10,
        "max_aspect_ratio": 3.0,
        "max_surface_aspect_ratio": 2.6,
        "max_target_y_below_piece_ratio": 0.08,
        "center_y_ratio": 0.40,
        "top_surface_seed_ratio": 0.20,
        "top_surface_max_height_ratio": 0.72,
        "top_surface_color_tolerance": 34,
        "top_surface_min_area": 60,
        "top_surface_center_y_ratio": 0.50,
    },
    "overlay": {
        "dark_gray_threshold": 88,
        "min_dark_area_ratio": 0.055,
        "min_dark_width_ratio": 0.55,
        "min_dark_height_ratio": 0.12,
    },
}


class JumpAutoError(RuntimeError):
    """Base exception with a user-facing message."""


class DependencyError(JumpAutoError):
    """Raised when a required third-party dependency is missing."""


class RecognitionError(JumpAutoError):
    """Raised when image recognition fails."""


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    window_rect: tuple[int, int, int, int]
    client_rect: tuple[int, int, int, int]
    iconic: bool

    @property
    def client_width(self) -> int:
        return self.client_rect[2] - self.client_rect[0]

    @property
    def client_height(self) -> int:
        return self.client_rect[3] - self.client_rect[1]


@dataclass(frozen=True)
class DetectionResult:
    piece: tuple[int, int]
    target: tuple[int, int]
    piece_bbox: tuple[int, int, int, int]
    target_bbox: tuple[int, int, int, int]
    crop_rect: tuple[int, int, int, int]
    distance_px: float
    confidence: float
    debug_path: Path


def import_or_raise(importer: Callable[[], Any], package_name: str) -> Any:
    try:
        return importer()
    except ModuleNotFoundError as exc:
        missing_name = exc.name or package_name
        raise DependencyError(
            f"Missing dependency '{missing_name}'. Install dependencies with:\n"
            f"  {sys.executable} -m pip install -r requirements.txt"
        ) from exc


def import_cv() -> tuple[Any, Any]:
    def _load() -> tuple[Any, Any]:
        import cv2
        import numpy as np

        return cv2, np

    return import_or_raise(_load, "opencv-python/numpy")


def import_mss() -> Any:
    return import_or_raise(lambda: __import__("mss"), "mss")


def import_pyautogui() -> Any:
    return import_or_raise(lambda: __import__("pyautogui"), "pyautogui")


def import_win32() -> tuple[Any, Any, Any]:
    def _load() -> tuple[Any, Any, Any]:
        import win32api
        import win32con
        import win32gui

        return win32gui, win32con, win32api

    return import_or_raise(_load, "pywin32")


def import_pynput_keyboard() -> Any:
    def _load() -> Any:
        from pynput import keyboard

        return keyboard

    return import_or_raise(_load, "pynput")


def import_pynput_mouse() -> Any:
    def _load() -> Any:
        from pynput import mouse

        return mouse

    return import_or_raise(_load, "pynput")


def require_windows() -> None:
    if platform.system().lower() != "windows":
        raise JumpAutoError("This script is Windows-only because it uses pywin32 window APIs.")


def set_dpi_awareness() -> None:
    if platform.system().lower() != "windows":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as file:
        user_config = json.load(file)
    return deep_merge(DEFAULT_CONFIG, user_config)


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
    with mss_module.mss() as sct:
        frame = np.array(sct.grab(monitor))
    return frame[:, :, :3].copy(), current_rect


def crop_game_area(frame: Any, config: dict[str, Any]) -> tuple[Any, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    crop_config = config["crop"]
    left = int(width * float(crop_config["left_ratio"]))
    right = int(width * float(crop_config["right_ratio"]))
    top = int(height * float(crop_config["top_ratio"]))
    bottom = int(height * float(crop_config["bottom_ratio"]))
    left = int(clamp(left, 0, width - 2))
    right = int(clamp(right, left + 2, width))
    top = int(clamp(top, 0, height - 2))
    bottom = int(clamp(bottom, top + 2, height))
    return frame[top:bottom, left:right], (left, top, right, bottom)


def screen_overlay_present(crop: Any, config: dict[str, Any]) -> bool:
    cv2, np = import_cv()
    overlay_cfg = config.get("overlay") or DEFAULT_CONFIG["overlay"]
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold = int(overlay_cfg.get("dark_gray_threshold", 88))
    mask = (gray < threshold).astype(np.uint8) * 255
    mask[: int(height * 0.15), :] = 0

    kernel_width = max(9, int(width * 0.035) | 1)
    kernel_height = max(9, int(height * 0.020) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    min_area = width * height * float(overlay_cfg.get("min_dark_area_ratio", 0.055))
    min_width = width * float(overlay_cfg.get("min_dark_width_ratio", 0.55))
    min_height = height * float(overlay_cfg.get("min_dark_height_ratio", 0.12))
    for label in range(1, components):
        area = float(stats[label, cv2.CC_STAT_AREA])
        box_width = float(stats[label, cv2.CC_STAT_WIDTH])
        box_height = float(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_area and box_width >= min_width and box_height >= min_height:
            return True
    return False


def find_piece(crop: Any, config: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int, int, int], Any]:
    cv2, np = import_cv()
    piece_cfg = config["piece"]
    height, _ = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array(piece_cfg["hsv_lower"], dtype=np.uint8)
    upper = np.array(piece_cfg["hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    search_top = int(height * float(piece_cfg["search_top_ratio"]))
    search_bottom = int(height * float(piece_cfg["search_bottom_ratio"]))
    mask[:search_top, :] = 0
    mask[search_bottom:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, Any, tuple[int, int, int, int]]] = []
    min_area = float(piece_cfg["min_area"])
    max_area = float(piece_cfg["max_area"])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, width, height_box = cv2.boundingRect(contour)
        if width < 8 or height_box < 18:
            continue
        score = area + y * 0.25 + height_box * 5
        candidates.append((score, contour, (x, y, width, height_box)))

    if not candidates:
        raise RecognitionError("Could not detect the piece. Try adjusting piece HSV thresholds.")

    _, _, bbox = max(candidates, key=lambda item: item[0])
    x, y, width, height_box = bbox
    foot_offset = int(piece_cfg["foot_offset_px"])
    point = (int(x + width / 2), int(y + height_box - foot_offset))
    return point, bbox, mask


def side_mask_for_target(mask: Any, piece: tuple[int, int], config: dict[str, Any]) -> Any:
    _, np = import_cv()
    target_cfg = config["target"]
    height, width = mask.shape[:2]
    piece_x, piece_y = piece
    side_gap = int(width * float(target_cfg["side_gap_ratio"]))
    side = np.zeros_like(mask)
    if piece_x < width / 2:
        side[:, min(width, piece_x + side_gap) :] = 255
    else:
        side[:, : max(0, piece_x - side_gap)] = 255

    search_top = int(height * float(target_cfg["search_top_ratio"]))
    search_bottom = int(
        min(height, piece_y + height * float(target_cfg["search_bottom_extra_ratio"]))
    )
    side[:search_top, :] = 0
    side[search_bottom:, :] = 0
    return mask & side


def exclude_piece_area(mask: Any, piece_bbox: tuple[int, int, int, int], config: dict[str, Any]) -> Any:
    target_cfg = config["target"]
    x, y, width, height = piece_bbox
    pad = int(target_cfg["exclude_piece_pad_px"])
    mask_height, mask_width = mask.shape[:2]
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(mask_width, x + width + pad)
    bottom = min(mask_height, y + height + pad)
    mask[top:bottom, left:right] = 0
    return mask


def build_background_diff_mask(crop: Any, config: dict[str, Any]) -> Any:
    cv2, np = import_cv()
    target_cfg = config["target"]
    height, width = crop.shape[:2]
    margin = max(4, int(width * 0.04))
    sample = np.concatenate([crop[:, :margin, :], crop[:, width - margin :, :]], axis=1)
    background = np.median(sample, axis=1).reshape(height, 1, 3)
    diff = crop.astype(np.float32) - background.astype(np.float32)
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    mask = (distance > float(target_cfg["diff_threshold"])).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_edge_mask(crop: Any) -> Any:
    cv2, _ = import_cv()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, kernel, iterations=1)


def contour_mask_for_bbox(contour: Any, bbox: tuple[int, int, int, int]) -> Any:
    cv2, np = import_cv()
    x, y, width, height = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, :, 0] -= x
    shifted[:, :, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    return mask


def binary_bbox(mask: Any, origin: tuple[int, int]) -> tuple[int, int, int, int] | None:
    cv2, _ = import_cv()
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return origin[0] + x, origin[1] + y, width, height


def keep_seeded_component(mask: Any, seed_mask: Any) -> Any:
    cv2, np = import_cv()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    seed_labels = {
        int(label)
        for label in np.unique(labels[seed_mask > 0])
        if int(label) != 0
    }
    candidate_labels = seed_labels or set(range(1, count))
    best_label = max(candidate_labels, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return ((labels == best_label).astype(np.uint8)) * 255


def point_from_surface_bbox(
    bbox: tuple[int, int, int, int],
    center_y_ratio: float,
) -> tuple[int, int]:
    x, y, width, height = bbox
    return int(x + width / 2), int(y + height * center_y_ratio)


def estimate_surface_by_geometry(
    component_mask: Any,
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int, int, int], float, int] | None:
    cv2, np = import_cv()
    target_cfg = config["target"]
    x, y, width, height = bbox
    rows = np.flatnonzero(np.any(component_mask > 0, axis=1))
    if len(rows) == 0:
        return None

    top_row = int(rows[0])
    max_height_ratio = float(target_cfg.get("top_surface_max_height_ratio", 0.72))
    bottom_limit = min(height, top_row + max(8, int(height * max_height_ratio)))
    upper_mask = np.zeros_like(component_mask)
    upper_mask[top_row:bottom_limit, :] = component_mask[top_row:bottom_limit, :]

    seed_bottom = min(height, top_row + max(6, int(height * 0.18)))
    seed_mask = np.zeros_like(component_mask)
    seed_mask[top_row:seed_bottom, :] = component_mask[top_row:seed_bottom, :]
    upper_mask = keep_seeded_component(upper_mask, seed_mask)

    surface_bbox = binary_bbox(upper_mask, (x, y))
    if surface_bbox is None:
        return None

    area = int(cv2.countNonZero(upper_mask))
    center_y_ratio = float(target_cfg.get("center_y_ratio", 0.40))
    point = point_from_surface_bbox(surface_bbox, center_y_ratio)
    return point, surface_bbox, 0.55, area


def estimate_top_surface(
    crop: Any,
    contour: Any,
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int, int, int], float, int] | None:
    cv2, np = import_cv()
    target_cfg = config["target"]
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None

    component_mask = contour_mask_for_bbox(contour, bbox)
    component_area = int(cv2.countNonZero(component_mask))
    if component_area <= 0:
        return None

    rows = np.flatnonzero(np.any(component_mask > 0, axis=1))
    if len(rows) == 0:
        return None
    top_row = int(rows[0])

    seed_ratio = float(target_cfg.get("top_surface_seed_ratio", 0.20))
    seed_bottom = min(height, top_row + max(6, int(height * seed_ratio)))
    seed_mask = np.zeros_like(component_mask)
    seed_mask[top_row:seed_bottom, :] = component_mask[top_row:seed_bottom, :]
    if cv2.countNonZero(seed_mask) < 12:
        seed_bottom = min(height, top_row + max(12, int(height * 0.35)))
        seed_mask[:, :] = 0
        seed_mask[top_row:seed_bottom, :] = component_mask[top_row:seed_bottom, :]
    if cv2.countNonZero(seed_mask) == 0:
        return estimate_surface_by_geometry(component_mask, bbox, config)

    roi = crop[y : y + height, x : x + width]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    seed_color = np.median(lab[seed_mask > 0], axis=0)
    color_distance = np.sqrt(np.sum((lab - seed_color.reshape(1, 1, 3)) ** 2, axis=2))

    max_height_ratio = float(target_cfg.get("top_surface_max_height_ratio", 0.72))
    bottom_limit = min(height, top_row + max(8, int(height * max_height_ratio)))
    min_surface_area = max(
        int(target_cfg.get("top_surface_min_area", 60)),
        int(component_area * 0.04),
    )
    base_tolerance = float(target_cfg.get("top_surface_color_tolerance", 34))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best_surface = None
    best_area = 0

    for tolerance_scale in (1.0, 1.25, 1.50):
        surface_mask = ((color_distance <= base_tolerance * tolerance_scale) & (component_mask > 0)).astype(np.uint8) * 255
        surface_mask[bottom_limit:, :] = 0
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        surface_mask = cv2.bitwise_and(surface_mask, component_mask)
        surface_mask[bottom_limit:, :] = 0
        surface_mask = keep_seeded_component(surface_mask, seed_mask)
        area = int(cv2.countNonZero(surface_mask))
        if area > best_area:
            best_surface = surface_mask
            best_area = area
        if area >= min_surface_area:
            break

    if best_surface is None or best_area < min_surface_area:
        return estimate_surface_by_geometry(component_mask, bbox, config)

    surface_bbox = binary_bbox(best_surface, (x, y))
    if surface_bbox is None:
        return estimate_surface_by_geometry(component_mask, bbox, config)

    center_y_ratio = float(target_cfg.get("top_surface_center_y_ratio", 0.50))
    point = point_from_surface_bbox(surface_bbox, center_y_ratio)
    surface_ratio = best_area / max(1.0, float(component_area))
    quality = clamp(0.58 + min(0.40, surface_ratio * 0.95), 0.0, 1.0)
    return point, surface_bbox, quality, best_area


def choose_target_from_mask(
    crop: Any,
    mask: Any,
    piece: tuple[int, int],
    config: dict[str, Any],
    confidence_scale: float,
) -> tuple[tuple[int, int], tuple[int, int, int, int], float] | None:
    cv2, _ = import_cv()
    target_cfg = config["target"]
    height, width = mask.shape[:2]
    max_area = float(target_cfg["max_area_ratio"]) * width * height
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int], tuple[int, int, int, int], float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(target_cfg["min_area"]) or area > max_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < int(target_cfg["min_width"]) or box_height < int(target_cfg["min_height"]):
            continue
        aspect_ratio = max(
            box_width / max(1.0, float(box_height)),
            box_height / max(1.0, float(box_width)),
        )
        max_aspect_ratio = float(target_cfg.get("max_aspect_ratio", 3.0))
        if aspect_ratio > max_aspect_ratio:
            continue

        surface = estimate_top_surface(crop, contour, (x, y, box_width, box_height), config)
        if surface is None:
            continue
        (target_x, target_y), surface_bbox, surface_quality, surface_area = surface

        _, _, surface_width, surface_height = surface_bbox
        surface_aspect = max(
            surface_width / max(1.0, float(surface_height)),
            surface_height / max(1.0, float(surface_width)),
        )
        max_surface_aspect_ratio = float(
            target_cfg.get("max_surface_aspect_ratio", max_aspect_ratio)
        )
        if surface_aspect > max_surface_aspect_ratio:
            continue

        distance = math.dist(piece, (target_x, target_y))
        if distance < width * float(target_cfg.get("min_distance_ratio", 0.10)):
            continue
        if target_y > piece[1] + height * float(target_cfg.get("max_target_y_below_piece_ratio", 0.08)):
            continue
        area_score = min(1.0, area / max(1.0, width * height * 0.025))
        surface_score = min(1.0, surface_area / max(1.0, width * height * 0.010)) * surface_quality
        distance_score = min(1.0, distance / max(1.0, width * 0.60))
        vertical_score = 1.0 if target_y < piece[1] + height * 0.10 else 0.4
        shape_score = 1.0 - 0.25 * clamp((aspect_ratio - 1.0) / max(0.1, max_aspect_ratio - 1.0), 0.0, 1.0)
        score = (
            0.38 * area_score
            + 0.26 * distance_score
            + 0.22 * surface_score
            + 0.14 * vertical_score
        ) * shape_score
        confidence = clamp(score * confidence_scale * (0.85 + 0.15 * surface_quality), 0.0, 1.0)
        candidates.append((score, (target_x, target_y), surface_bbox, confidence))

    if not candidates:
        return None
    _, point, bbox, confidence = max(candidates, key=lambda item: item[0])
    return point, bbox, confidence


def find_target(
    crop: Any,
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int, int, int], float, Any]:
    cv2, _ = import_cv()

    diff_mask = build_background_diff_mask(crop, config)
    diff_mask = side_mask_for_target(diff_mask, piece, config)
    diff_mask = exclude_piece_area(diff_mask, piece_bbox, config)
    primary = choose_target_from_mask(crop, diff_mask, piece, config, confidence_scale=1.0)
    if primary is not None:
        point, bbox, confidence = primary
        return point, bbox, confidence, diff_mask

    edge_mask = build_edge_mask(crop)
    edge_mask = side_mask_for_target(edge_mask, piece, config)
    edge_mask = exclude_piece_area(edge_mask, piece_bbox, config)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fallback = choose_target_from_mask(crop, edge_mask, piece, config, confidence_scale=0.72)
    if fallback is None:
        raise RecognitionError("Could not detect the next target platform.")
    point, bbox, confidence = fallback
    return point, bbox, confidence, edge_mask


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def draw_debug(
    frame: Any,
    detection: DetectionResult,
    press_ms: float | None = None,
) -> Any:
    cv2, _ = import_cv()
    debug = frame.copy()
    crop_left, crop_top, crop_right, crop_bottom = detection.crop_rect
    cv2.rectangle(debug, (crop_left, crop_top), (crop_right, crop_bottom), (0, 255, 255), 2)
    cv2.circle(debug, detection.piece, 8, (255, 80, 0), -1)
    cv2.circle(debug, detection.target, 8, (0, 220, 0), -1)
    cv2.line(debug, detection.piece, detection.target, (255, 255, 255), 2)

    px, py, pw, ph = detection.piece_bbox
    tx, ty, tw, th = detection.target_bbox
    cv2.rectangle(
        debug,
        (crop_left + px, crop_top + py),
        (crop_left + px + pw, crop_top + py + ph),
        (255, 80, 0),
        2,
    )
    cv2.rectangle(
        debug,
        (crop_left + tx, crop_top + ty),
        (crop_left + tx + tw, crop_top + ty + th),
        (0, 220, 0),
        2,
    )
    label = f"distance={detection.distance_px:.1f}px confidence={detection.confidence:.2f}"
    if press_ms is not None:
        label += f" press={press_ms:.0f}ms"
    cv2.putText(debug, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
    cv2.putText(debug, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return debug


def detect_jump(
    frame: Any,
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
    press_ms: float | None = None,
    save_mask: bool = False,
) -> DetectionResult:
    cv2, _ = import_cv()
    crop, crop_rect = crop_game_area(frame, config)
    crop_left, crop_top, _, _ = crop_rect
    if screen_overlay_present(crop, config):
        raise RecognitionError("A game-over or modal overlay appears to be covering the board.")
    piece, piece_bbox, piece_mask = find_piece(crop, config)
    target, target_bbox, confidence, target_mask = find_target(crop, piece, piece_bbox, config)
    piece_full = (piece[0] + crop_left, piece[1] + crop_top)
    target_full = (target[0] + crop_left, target[1] + crop_top)
    distance = math.dist(piece_full, target_full)

    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{label}_{timestamp()}.png"
    result = DetectionResult(
        piece=piece_full,
        target=target_full,
        piece_bbox=piece_bbox,
        target_bbox=target_bbox,
        crop_rect=crop_rect,
        distance_px=distance,
        confidence=confidence,
        debug_path=debug_path,
    )
    debug = draw_debug(frame, result, press_ms=press_ms)
    cv2.imwrite(str(debug_path), debug)

    if save_mask:
        cv2.imwrite(str(debug_dir / f"{label}_{timestamp()}_piece_mask.png"), piece_mask)
        cv2.imwrite(str(debug_dir / f"{label}_{timestamp()}_target_mask.png"), target_mask)
    return result


def calculate_press_ms(distance_px: float, config: dict[str, Any]) -> float:
    ratio = config.get("press_ms_per_px")
    if ratio is None:
        raise JumpAutoError("press_ms_per_px is not configured. Run --calibrate first.")
    press_ms = distance_px * float(ratio)
    return clamp(press_ms, float(config["min_press_ms"]), float(config["max_press_ms"]))


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
    press_ms = calculate_press_ms(first_result.distance_px, config)
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
    print(f"Distance: {result.distance_px:.1f}px  Confidence: {result.confidence:.2f}")
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


def run_calibration(args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> None:
    window = locate_window(args.window_title, config)
    frame, _ = capture_window(window, config)
    result = detect_jump(frame, config, args.debug_dir, "calibrate_preview", save_mask=args.save_masks)
    print_detection(window, result)
    print()
    print("Open the debug image and confirm the markers are correct.")
    answer = input("Type 'y' to record one manual successful jump, or anything else to cancel: ")
    if answer.strip().lower() != "y":
        raise JumpAutoError("Calibration cancelled.")

    print("Now perform exactly one manual left-button long press in the WeChat game window.")
    print("The script will record the next complete left-button press/release.")
    duration_ms = record_one_manual_press()
    coefficient = duration_ms / max(1.0, result.distance_px)
    print(f"Recorded manual press: {duration_ms:.0f}ms")
    print(f"Computed coefficient: {coefficient:.4f} ms/px")
    success = input("Was that jump successful? Type 'y' to save the coefficient: ")
    if success.strip().lower() != "y":
        raise JumpAutoError("Calibration was not saved.")

    config["press_ms_per_px"] = coefficient
    if args.window_title:
        config["window_title"] = args.window_title
    save_config(config_path, config)
    print(f"Saved config: {config_path}")


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
    if config.get("press_ms_per_px") is None:
        raise JumpAutoError("press_ms_per_px is not configured. Run --calibrate first.")

    stop_event = threading.Event()
    pause_event = threading.Event()
    listener = start_hotkey_listener(stop_event, pause_event)
    print("Auto mode started. Press F8 to pause/resume, Esc to exit.")
    jump_count = 0
    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.15)
                continue

            window = locate_window(args.window_title, config)
            frame, client_rect = capture_window(window, config)
            preview = detect_jump(frame, config, args.debug_dir, f"auto_{jump_count:04d}")
            press_ms = calculate_press_ms(preview.distance_px, config)

            if preview.confidence < float(config["confidence_threshold"]):
                print(
                    f"Low confidence {preview.confidence:.2f}; pausing. "
                    f"Debug image: {preview.debug_path}"
                )
                pause_event.set()
                continue

            print_detection(window, preview, press_ms=press_ms)
            press_in_window(window, client_rect, config, press_ms)
            jump_count += 1
            time.sleep(float(args.interval))
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        print(f"Auto mode ended. Completed jumps: {jump_count}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows WeChat Jump desktop automation helper."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Capture and detect only.")
    mode.add_argument("--calibrate", action="store_true", help="Record one manual jump coefficient.")
    mode.add_argument("--auto", action="store_true", help="Run continuous automatic jumps.")
    mode.add_argument("--single-step", action="store_true", help="Run exactly one automatic jump.")
    mode.add_argument("--list-windows", action="store_true", help="Print visible window candidates.")
    parser.add_argument("--window-title", default=None, help="Substring of the target window title.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config JSON path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help=f"Debug image directory. Default: {DEFAULT_DEBUG_DIR}",
    )
    parser.add_argument("--interval", type=float, default=1.75, help="Seconds to wait after each jump.")
    parser.add_argument("--save-masks", action="store_true", help="Save intermediate mask images.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.config = args.config.resolve()
    args.debug_dir = args.debug_dir.resolve()
    try:
        set_dpi_awareness()
        if args.list_windows:
            print_window_list()
            return 0

        config = load_config(args.config)
        if args.dry_run:
            run_dry_run(args, config)
        elif args.calibrate:
            run_calibration(args, config, args.config)
        elif args.single_step:
            run_single_step(args, config)
        elif args.auto:
            run_auto(args, config)
        else:
            print("No mode selected. Start with --dry-run, --calibrate, --single-step, or --auto.")
            print("Use --list-windows to inspect window titles.")
            return 2
        return 0
    except DependencyError as exc:
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except JumpAutoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
