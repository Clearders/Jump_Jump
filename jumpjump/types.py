from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class JumpAutoError(RuntimeError):
    """Base exception with a user-facing message."""


class DependencyError(JumpAutoError):
    """Raised when a required third-party dependency is missing."""


class ConfigError(JumpAutoError):
    """Raised when configuration cannot be loaded, validated, or saved safely."""


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
    dx_px: float
    dy_px: float
    screen_distance_px: float
    effective_distance_px: float
    distance_px: float
    confidence: float
    debug_path: Path | None
    piece_median_hsv: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class LearningSample:
    """Internal training row reserved for online/offline model learning."""

    timestamp: str
    distance_px: float
    dx_px: float
    dy_px: float
    press_ms: float
    landing_error_px: float | None
    target: tuple[int, int]
    piece: tuple[int, int]
    confidence: float
    result_type: Literal["manual", "auto_success", "auto_adjusted", "auto_failure"]
    training_press_ms: float | None = None
