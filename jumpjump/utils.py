from __future__ import annotations

from datetime import datetime


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")