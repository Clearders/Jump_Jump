from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

from .dependencies import import_cv
from .types import JumpAutoError


_GENERATED_IMAGE_PATTERN = re.compile(
    r"^(?:"
    r"auto_\d{4}(?:_(?:low_confidence|recheck_(?:first|second)(?:_(?:failed|rejected))?))?"
    r"|calibrate(?:_\d{2})?_preview"
    r"|dry_run"
    r"|single_step(?:_preview)?"
    r"|vision_regression_[A-Za-z0-9_]+"
    r")"
    r"(?:_failed)?_\d{8}_\d{6}_\d{6}"
    r"(?:_(?:piece|target)_mask)?\.png$"
)


def debug_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["debug"]


def auto_capture_policy(config: dict[str, Any]) -> str:
    return str(debug_config(config)["auto_capture_policy"])


def is_generated_debug_image(path: Path) -> bool:
    return bool(_GENERATED_IMAGE_PATTERN.fullmatch(path.name))


def enforce_debug_retention(debug_dir: Path, config: dict[str, Any]) -> None:
    if not debug_dir.exists():
        return
    settings = debug_config(config)
    max_files = int(settings["max_files"])
    max_bytes = int(float(settings["max_size_mb"]) * 1024 * 1024)
    candidates: list[tuple[Path, int, int]] = []
    for path in debug_dir.iterdir():
        if not path.is_file() or not is_generated_debug_image(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        candidates.append((path, stat.st_mtime_ns, stat.st_size))

    candidates.sort(key=lambda item: (item[1], item[0].name))
    total_bytes = sum(item[2] for item in candidates)
    undeletable: set[Path] = set()
    while candidates and (len(candidates) > max_files or total_bytes > max_bytes):
        target_index = next(
            (index for index, item in enumerate(candidates) if item[0] not in undeletable),
            None,
        )
        if target_index is None:
            break
        path, _, size = candidates[target_index]
        try:
            path.unlink()
        except OSError as exc:
            undeletable.add(path)
            warnings.warn(
                f"Could not remove old debug image '{path}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        candidates.pop(target_index)
        total_bytes -= size


def write_debug_image(
    path: Path,
    image: Any,
    config: dict[str, Any],
) -> Path:
    cv2, _ = import_cv()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        written = bool(cv2.imwrite(str(path), image))
    except Exception as exc:
        raise JumpAutoError(f"Could not write debug image '{path}': {exc}") from exc
    if not written:
        raise JumpAutoError(f"Could not write debug image: {path}")
    enforce_debug_retention(path.parent, config)
    if not path.exists():
        raise JumpAutoError(
            f"Debug retention removed the newly written image; increase debug limits: {path}"
        )
    return path
