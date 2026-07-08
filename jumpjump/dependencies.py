from __future__ import annotations

import platform
import sys
from typing import Any, Callable

from .types import DependencyError, JumpAutoError


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