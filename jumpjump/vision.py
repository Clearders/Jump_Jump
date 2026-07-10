from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, deep_merge
from .debug_artifacts import write_debug_image
from .dependencies import import_cv
from .press_model import effective_distance_from_delta
from .types import DetectionResult, RecognitionError
from .utils import clamp, timestamp


@dataclass(frozen=True)
class TargetCandidate:
    point: tuple[int, int]
    bbox: tuple[int, int, int, int]
    score: float
    confidence: float
    source: str
    risks: tuple[str, ...] = ()


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


def dynamic_piece_hsv_bounds(config: dict[str, Any]) -> tuple[Any, Any] | None:
    _, np = import_cv()
    piece_cfg = config["piece"]
    if not bool(piece_cfg.get("dynamic_color_enabled", True)):
        return None
    samples = [
        sample.get("hsv")
        for sample in piece_cfg.get("color_samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("hsv"), list)
    ]
    min_samples = int(piece_cfg.get("dynamic_color_min_samples", 2))
    if len(samples) < min_samples:
        return None
    values = np.array(samples, dtype=np.float32)
    hue_margin = float(piece_cfg.get("dynamic_color_hue_margin", 14))
    saturation_margin = float(piece_cfg.get("dynamic_color_saturation_margin", 55))
    value_margin = float(piece_cfg.get("dynamic_color_value_margin", 48))
    lower = np.array(
        [
            max(0, float(values[:, 0].min()) - hue_margin),
            max(0, float(values[:, 1].min()) - saturation_margin),
            max(0, float(values[:, 2].min()) - value_margin),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(179, float(values[:, 0].max()) + hue_margin),
            min(255, float(values[:, 1].max()) + saturation_margin),
            min(255, float(values[:, 2].max()) + value_margin),
        ],
        dtype=np.uint8,
    )
    return lower, upper


def build_piece_mask(
    crop: Any,
    config: dict[str, Any],
    fallback: bool = False,
    value_upper_override: int | None = None,
) -> Any:
    cv2, np = import_cv()
    piece_cfg = config["piece"]
    height, _ = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_key = "fallback_hsv_lower" if fallback else "hsv_lower"
    upper_key = "fallback_hsv_upper" if fallback else "hsv_upper"
    lower = np.array(piece_cfg.get(lower_key, piece_cfg["hsv_lower"]), dtype=np.uint8)
    upper = np.array(piece_cfg.get(upper_key, piece_cfg["hsv_upper"]), dtype=np.uint8)
    if value_upper_override is not None:
        upper[2] = min(int(upper[2]), int(value_upper_override))
    mask = cv2.inRange(hsv, lower, upper)
    if value_upper_override is None:
        dynamic_bounds = dynamic_piece_hsv_bounds(config)
        if dynamic_bounds is not None:
            dynamic_lower, dynamic_upper = dynamic_bounds
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, dynamic_lower, dynamic_upper))

    search_top = int(height * float(piece_cfg["search_top_ratio"]))
    search_bottom = int(height * float(piece_cfg["search_bottom_ratio"]))
    mask[:search_top, :] = 0
    mask[search_bottom:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def piece_candidates_from_mask(
    mask: Any,
    crop: Any,
    config: dict[str, Any],
) -> list[tuple[float, Any, tuple[int, int, int, int]]]:
    cv2, np = import_cv()
    piece_cfg = config["piece"]
    mask_height, mask_width = mask.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, Any, tuple[int, int, int, int]]] = []
    min_area = float(piece_cfg["min_area"])
    max_area = float(piece_cfg["max_area"])
    min_width = int(piece_cfg.get("min_width", 8))
    min_height = int(piece_cfg.get("min_height", 18))
    max_width = int(mask_width * float(piece_cfg.get("max_width_ratio", 0.18)))
    min_height_width_ratio = float(piece_cfg.get("min_height_width_ratio", 1.05))
    edge_reject_px = int(piece_cfg.get("edge_reject_px", 4))
    preferred_hue = float(piece_cfg.get("preferred_hue", 122))
    preferred_hue_tolerance = float(piece_cfg.get("preferred_hue_tolerance", 38))
    min_median_saturation = float(piece_cfg.get("min_median_saturation", 45))
    preferred_max_value = float(piece_cfg.get("preferred_max_value", 155))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, width, height_box = cv2.boundingRect(contour)
        if x <= edge_reject_px or x + width >= mask_width - edge_reject_px:
            continue
        if width < min_width or height_box < min_height:
            continue
        if max_width > 0 and width > max_width:
            continue
        height_width_ratio = height_box / max(1.0, float(width))
        if height_width_ratio < min_height_width_ratio:
            continue
        if y + height_box > mask_height:
            continue

        contour_mask = np.zeros((height_box, width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(contour_mask, [shifted], -1, 255, -1)
        values = hsv[y : y + height_box, x : x + width][contour_mask > 0]
        if len(values) == 0:
            continue
        median_hue, median_saturation, median_value = np.median(values, axis=0)
        if median_saturation < min_median_saturation:
            continue

        hue_delta = abs(float(median_hue) - preferred_hue)
        hue_delta = min(hue_delta, 180.0 - hue_delta)
        hue_score = clamp(1.0 - hue_delta / max(1.0, preferred_hue_tolerance), 0.0, 1.0)
        saturation_score = clamp((float(median_saturation) - min_median_saturation) / 85.0, 0.0, 1.0)
        value_score = clamp((preferred_max_value - float(median_value)) / 100.0, 0.0, 1.0)
        ratio_score = clamp((height_width_ratio - min_height_width_ratio) / 1.2, 0.0, 1.0)
        fill_score = clamp(area / max(1.0, width * height_box), 0.0, 1.0)
        size_score = min(1.0, area / max(1.0, max_area * 0.45))
        vertical_score = y / max(1.0, mask_height)
        score = (
            3200.0 * hue_score
            + 2600.0 * saturation_score
            + 1800.0 * value_score
            + 1700.0 * ratio_score
            + 1200.0 * fill_score
            + 900.0 * size_score
            + 450.0 * vertical_score
        )
        candidates.append((score, contour, (x, y, width, height_box)))
    return candidates


def sample_piece_median_hsv(crop: Any, bbox: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    cv2, np = import_cv()
    x, y, width, height = bbox
    left = int(x + width * 0.18)
    right = int(x + width * 0.82)
    top = int(y + height * 0.10)
    bottom = int(y + height * 0.88)
    if right <= left or bottom <= top:
        return None
    patch = crop[top:bottom, left:right]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(hsv) == 0:
        return None
    median_hsv = np.median(hsv, axis=0)
    return float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])


def update_piece_color_model(
    config: dict[str, Any],
    result: DetectionResult,
    source: str,
) -> bool:
    hsv = result.piece_median_hsv
    if hsv is None:
        return False
    piece_cfg = config["piece"]
    if not bool(piece_cfg.get("dynamic_color_enabled", True)):
        return False
    samples = piece_cfg.setdefault("color_samples", [])
    samples.append(
        {
            "timestamp": timestamp(),
            "source": source,
            "hsv": [round(float(hsv[0]), 2), round(float(hsv[1]), 2), round(float(hsv[2]), 2)],
            "confidence": round(float(result.confidence), 3),
        }
    )
    max_samples = int(piece_cfg.get("dynamic_color_max_samples", 24))
    if len(samples) > max_samples:
        del samples[:-max_samples]
    return True


def find_piece(crop: Any, config: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int, int, int], Any]:
    piece_cfg = config["piece"]
    mask = build_piece_mask(crop, config, fallback=False)
    candidate_sets = [
        (score, contour, bbox, mask)
        for score, contour, bbox in piece_candidates_from_mask(mask, crop, config)
    ]

    fallback_mask = build_piece_mask(crop, config, fallback=True)
    candidate_sets.extend(
        (score, contour, bbox, fallback_mask)
        for score, contour, bbox in piece_candidates_from_mask(fallback_mask, crop, config)
    )

    if piece_cfg.get("core_value_upper") is not None:
        core_mask = build_piece_mask(
            crop,
            config,
            fallback=True,
            value_upper_override=int(piece_cfg["core_value_upper"]),
        )
        candidate_sets.extend(
            (score, contour, bbox, core_mask)
            for score, contour, bbox in piece_candidates_from_mask(core_mask, crop, config)
        )

    if not candidate_sets:
        raise RecognitionError("Could not detect the piece. Try adjusting piece HSV thresholds.")

    _, _, bbox, selected_mask = max(candidate_sets, key=lambda item: item[0])
    x, y, width, height_box = bbox
    foot_offset = int(piece_cfg["foot_offset_px"])
    point = (int(x + width / 2), int(y + height_box - foot_offset))
    return point, bbox, selected_mask


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
    sample_float = sample.astype(np.float32)
    sample_median = np.median(sample_float, axis=1, keepdims=True)
    sample_std = np.std(sample_float, axis=1, keepdims=True)
    sample_std = np.maximum(sample_std, 1.0)
    deviation = np.abs(sample_float - sample_median) / sample_std
    inlier_mask = deviation < 2.0
    inlier_mask_any = np.all(inlier_mask, axis=2, keepdims=True)
    masked_sample = np.where(inlier_mask_any, sample_float, sample_median)
    background = np.median(masked_sample, axis=1).reshape(height, 1, 3)
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


def median_lab_in_rect(crop: Any, rect: tuple[int, int, int, int]) -> Any | None:
    cv2, np = import_cv()
    height, width = crop.shape[:2]
    left, top, right, bottom = rect
    left = clamp(left, 0, width)
    right = clamp(right, 0, width)
    top = clamp(top, 0, height)
    bottom = clamp(bottom, 0, height)
    if right <= left or bottom <= top:
        return None
    patch = crop[int(top) : int(bottom), int(left) : int(right)]
    if patch.size == 0:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if len(lab) == 0:
        return None
    return np.median(lab, axis=0)


def side_platform_color_matches_target(
    crop: Any,
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    target: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> bool:
    _, np = import_cv()
    target_cfg = config["target"]
    piece_x, _ = piece
    piece_left, piece_top, piece_width, piece_height = piece_bbox
    sample_px = max(10, int(target_cfg.get("current_platform_side_sample_px", 34)))
    band_top = int(piece_top + piece_height * 0.50)
    band_bottom = int(piece_top + piece_height + max(8, piece_height * 0.10))
    if target[0] >= piece_x:
        base_rect = (
            piece_left + piece_width + 2,
            band_top,
            piece_left + piece_width + sample_px,
            band_bottom,
        )
    else:
        base_rect = (
            piece_left - sample_px,
            band_top,
            piece_left - 2,
            band_bottom,
        )

    target_left, target_top, target_width, target_height = target_bbox
    target_rect = (
        target_left,
        target_top,
        target_left + target_width,
        target_top + target_height,
    )
    base_lab = median_lab_in_rect(crop, base_rect)
    target_lab = median_lab_in_rect(crop, target_rect)
    if base_lab is None or target_lab is None:
        return False

    color_distance = float(np.linalg.norm(base_lab - target_lab))
    tolerance = float(target_cfg.get("current_platform_color_tolerance_lab", 28))
    return color_distance <= tolerance


def looks_like_current_platform_target(
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    target: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
    mask_width: int,
    crop: Any | None = None,
) -> bool:
    target_cfg = config["target"]
    x, y, width, height = target_bbox
    pad_x = max(8, int(width * float(target_cfg.get("current_platform_exclude_pad_ratio", 0.25))))
    pad_y = max(8, int(height * 0.12))
    piece_x, piece_y = piece
    max_target_above_piece = mask_width * float(
        target_cfg.get("current_platform_max_target_above_piece_ratio", 0.045)
    )
    if target[1] < piece_y - max_target_above_piece:
        return False
    if piece_x < x:
        edge_gap_x = x - piece_x
    elif piece_x > x + width:
        edge_gap_x = piece_x - (x + width)
    else:
        edge_gap_x = 0
    max_edge_gap_x = max(
        6,
        int(width * float(target_cfg.get("current_platform_edge_gap_ratio", 0.18))),
    )
    if edge_gap_x > max_edge_gap_x:
        if crop is None:
            return False
        distance = math.dist(piece, target)
        max_color_distance = mask_width * float(
            target_cfg.get("current_platform_color_max_distance_ratio", 0.18)
        )
        return (
            distance <= max_color_distance
            and side_platform_color_matches_target(
                crop,
                piece,
                piece_bbox,
                target,
                target_bbox,
                config,
            )
        )
    near_bbox = (
        x - pad_x <= piece_x <= x + width + pad_x
        and y - pad_y <= piece_y <= y + height + pad_y
    )
    if not near_bbox:
        return False
    distance = math.dist(piece, target)
    max_distance = mask_width * float(target_cfg.get("current_platform_max_distance_ratio", 0.26))
    return distance <= max_distance


def constrain_surface_bbox(
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    target_cfg = config["target"]
    x, y, width, height = bbox
    max_height_to_width = float(target_cfg.get("top_surface_max_height_to_width", 0.68))
    max_height = max(8, int(width * max_height_to_width))
    if height > max_height:
        height = max_height
    return x, y, width, height


def constrained_surface_from_mask(
    surface_mask: Any,
    origin: tuple[int, int],
    config: dict[str, Any],
) -> tuple[Any, tuple[int, int, int, int], int] | None:
    cv2, np = import_cv()
    surface_bbox = binary_bbox(surface_mask, origin)
    if surface_bbox is None:
        return None

    constrained_bbox = constrain_surface_bbox(surface_bbox, config)
    x, y, width, height = constrained_bbox
    origin_x, origin_y = origin
    local_left = int(clamp(x - origin_x, 0, surface_mask.shape[1]))
    local_top = int(clamp(y - origin_y, 0, surface_mask.shape[0]))
    local_right = int(clamp(local_left + width, local_left, surface_mask.shape[1]))
    local_bottom = int(clamp(local_top + height, local_top, surface_mask.shape[0]))
    if local_right <= local_left or local_bottom <= local_top:
        return None

    constrained_mask = np.zeros_like(surface_mask)
    constrained_mask[local_top:local_bottom, local_left:local_right] = surface_mask[
        local_top:local_bottom,
        local_left:local_right,
    ]
    constrained_bbox = binary_bbox(constrained_mask, origin)
    if constrained_bbox is None:
        return None
    constrained_bbox = constrain_surface_bbox(constrained_bbox, config)
    area = int(cv2.countNonZero(constrained_mask))
    if area <= 0:
        return None
    return constrained_mask, constrained_bbox, area


def bbox_fill_ratio(area: float, bbox: tuple[int, int, int, int]) -> float:
    _, _, width, height = bbox
    return float(area) / max(1.0, float(width * height))


def bbox_edge_touch_count(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin: int = 2,
) -> int:
    x, y, width, height = bbox
    touches = 0
    if x <= margin:
        touches += 1
    if y <= margin:
        touches += 1
    if x + width >= frame_width - margin:
        touches += 1
    if y + height >= frame_height - margin:
        touches += 1
    return touches


def focus_far_edge_surface_bbox(
    bbox: tuple[int, int, int, int],
    piece: tuple[int, int],
    mask_width: int,
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    target_cfg = config["target"]
    x, y, width, height = bbox
    min_focus_width = mask_width * float(
        target_cfg.get("far_edge_surface_focus_width_ratio", 0.48)
    )
    if width < min_focus_width:
        return bbox

    piece_x, _ = piece
    trim_ratio = float(target_cfg.get("far_edge_surface_focus_trim_ratio", 0.30))
    min_width = max(18, int(target_cfg.get("min_width", 18)))
    trim_px = min(max(0, int(width * trim_ratio)), max(0, width - min_width))
    if trim_px <= 0:
        return bbox

    if piece_x < mask_width / 2 and x + width >= mask_width - 2:
        return x + trim_px, y, width - trim_px, height
    if piece_x >= mask_width / 2 and x <= 2:
        return x, y, width - trim_px, height
    return bbox


def top_surface_point_from_bbox(
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[int, int]:
    target_cfg = config["target"]
    _, _, width, height = bbox
    aspect = width / max(1.0, float(height))
    if aspect > 1.6:
        center_y_ratio = 0.44
    elif aspect > 1.0:
        center_y_ratio = 0.48
    else:
        center_y_ratio = float(target_cfg.get("top_surface_center_y_ratio", 0.50))
    return point_from_surface_bbox(bbox, center_y_ratio)


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

    constrained = constrained_surface_from_mask(upper_mask, (x, y), config)
    if constrained is None:
        return None
    _, surface_bbox, area = constrained

    _, _, geo_w, geo_h = surface_bbox
    geo_aspect = geo_w / max(1.0, float(geo_h))
    if geo_aspect > 1.6:
        center_y_ratio = 0.38
    elif geo_aspect > 1.0:
        center_y_ratio = 0.42
    else:
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
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    seed_color = np.median(lab[seed_mask > 0], axis=0)
    seed_hsv = np.median(hsv[seed_mask > 0], axis=0)
    color_distance = np.sqrt(np.sum((lab - seed_color.reshape(1, 1, 3)) ** 2, axis=2))
    hue_delta = np.abs(hsv[:, :, 0] - float(seed_hsv[0]))
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    saturation_delta = np.abs(hsv[:, :, 1] - float(seed_hsv[1]))
    value_delta = np.abs(hsv[:, :, 2] - float(seed_hsv[2]))

    max_height_ratio = float(target_cfg.get("top_surface_max_height_ratio", 0.72))
    bottom_limit = min(height, top_row + max(8, int(height * max_height_ratio)))
    min_surface_area = max(
        int(target_cfg.get("top_surface_min_area", 60)),
        int(component_area * 0.04),
    )
    base_tolerance = float(target_cfg.get("top_surface_color_tolerance", 34))
    hue_tolerance = float(target_cfg.get("top_surface_hue_tolerance", 18))
    saturation_tolerance = float(target_cfg.get("top_surface_saturation_tolerance", 72))
    value_tolerance = float(target_cfg.get("top_surface_value_tolerance", 52))
    min_hue_saturation = float(target_cfg.get("top_surface_min_saturation_for_hue", 24))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    surfaces: list[tuple[Any, int]] = []

    for tolerance_scale in (1.0, 1.25, 1.50):
        color_match = color_distance <= base_tolerance * tolerance_scale
        tone_match = (
            (saturation_delta <= saturation_tolerance * tolerance_scale)
            & (value_delta <= value_tolerance * tolerance_scale)
        )
        if float(seed_hsv[1]) >= min_hue_saturation:
            tone_match &= hue_delta <= hue_tolerance * tolerance_scale
        surface_mask = (color_match & tone_match & (component_mask > 0)).astype(np.uint8) * 255
        surface_mask[bottom_limit:, :] = 0
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        surface_mask = cv2.bitwise_and(surface_mask, component_mask)
        surface_mask[bottom_limit:, :] = 0
        surface_mask = keep_seeded_component(surface_mask, seed_mask)
        area = int(cv2.countNonZero(surface_mask))
        surfaces.append((surface_mask, area))

    if not surfaces:
        return estimate_surface_by_geometry(component_mask, bbox, config)

    best_surface, best_area = max(surfaces, key=lambda item: item[1])

    if best_surface is None or best_area < min_surface_area:
        return estimate_surface_by_geometry(component_mask, bbox, config)

    constrained = constrained_surface_from_mask(best_surface, (x, y), config)
    if constrained is None:
        return estimate_surface_by_geometry(component_mask, bbox, config)
    _, surface_bbox, best_area = constrained

    _, _, surface_w, surface_h = surface_bbox
    aspect = surface_w / max(1.0, float(surface_h))
    if aspect > 1.6:
        center_y_ratio = 0.44
    elif aspect > 1.0:
        center_y_ratio = 0.48
    else:
        center_y_ratio = float(target_cfg.get("top_surface_center_y_ratio", 0.50))

    point = point_from_surface_bbox(surface_bbox, center_y_ratio)
    surface_ratio = best_area / max(1.0, float(component_area))
    fill_ratio = bbox_fill_ratio(best_area, surface_bbox)
    quality = clamp(
        0.46
        + min(0.32, surface_ratio * 0.80)
        + min(0.22, fill_ratio * 0.35),
        0.0,
        1.0,
    )
    return point, surface_bbox, quality, best_area


def target_candidate_risk_multiplier(
    crop: Any,
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    target: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
    mask_width: int,
    mask_height: int,
    distance: float,
) -> tuple[float, tuple[str, ...]]:
    target_cfg = config["target"]
    piece_x, piece_y = piece
    piece_left, piece_top, piece_width, piece_height = piece_bbox
    target_x, target_y = target
    target_left, target_top, target_width, target_height = target_bbox
    risks: list[str] = []
    multiplier = 1.0

    below_limit = mask_height * float(target_cfg.get("max_target_y_below_piece_ratio", 0.08))
    if target_y > piece_y + below_limit:
        overshoot = (target_y - piece_y - below_limit) / max(1.0, mask_height * 0.22)
        multiplier *= clamp(1.0 - 0.36 * clamp(overshoot, 0.0, 1.0), 0.58, 1.0)
        risks.append("below_piece")

    if looks_like_current_platform_target(
        piece,
        piece_bbox,
        target,
        target_bbox,
        config,
        mask_width,
        crop,
    ):
        multiplier *= float(target_cfg.get("current_platform_risk_confidence_scale", 0.24))
        risks.append("current_platform")
    else:
        horizontal_band_px = max(
            18.0,
            mask_width * float(target_cfg.get("current_platform_horizontal_band_ratio", 0.055)),
        )
        close_distance = mask_width * float(
            target_cfg.get("current_platform_near_distance_ratio", 0.22)
        )
        target_center_y = target_top + target_height / 2.0
        platform_band_top = piece_top + piece_height * 0.48
        platform_band_bottom = piece_top + piece_height + max(8.0, piece_height * 0.18)
        horizontally_adjacent = (
            (target_x < piece_x and target_left + target_width <= piece_left + piece_width * 0.25)
            or (target_x > piece_x and target_left >= piece_left + piece_width * 0.75)
        )
        same_height_band = abs(target_y - piece_y) <= horizontal_band_px
        overlaps_platform_band = platform_band_top <= target_center_y <= platform_band_bottom
        if distance <= close_distance and horizontally_adjacent and same_height_band and overlaps_platform_band:
            multiplier *= float(
                target_cfg.get("current_platform_band_confidence_scale", 0.34)
            )
            risks.append("current_platform_band")

    color_distance_limit = mask_width * float(
        target_cfg.get("current_platform_color_max_distance_ratio", 0.14)
    )
    if distance <= color_distance_limit and side_platform_color_matches_target(
        crop,
        piece,
        piece_bbox,
        target,
        target_bbox,
        config,
    ):
        multiplier *= float(target_cfg.get("current_platform_color_confidence_scale", 0.55))
        risks.append("current_platform_color")

    edge_touches = bbox_edge_touch_count(target_bbox, mask_width, mask_height)
    if edge_touches >= 2:
        multiplier *= float(target_cfg.get("multi_edge_touch_confidence_scale", 0.68))
        risks.append("multi_edge_touch")
    elif edge_touches == 1:
        multiplier *= float(target_cfg.get("edge_touch_confidence_scale", 0.86))
        risks.append("edge_touch")

    return clamp(multiplier, 0.0, 1.0), tuple(risks)


def collect_target_candidates(
    crop: Any,
    mask: Any,
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
    confidence_scale: float,
    source: str,
) -> list[TargetCandidate]:
    cv2, _ = import_cv()
    target_cfg = config["target"]
    height, width = mask.shape[:2]
    max_area = float(target_cfg["max_area_ratio"]) * width * height
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[TargetCandidate] = []
    min_component_fill_ratio = float(target_cfg.get("min_component_fill_ratio", 0.035))
    min_surface_fill_ratio = float(target_cfg.get("min_surface_fill_ratio", 0.12))

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(target_cfg["min_area"]) or area > max_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < int(target_cfg["min_width"]) or box_height < int(target_cfg["min_height"]):
            continue
        component_bbox = (x, y, box_width, box_height)
        component_fill = bbox_fill_ratio(area, component_bbox)
        if component_fill < min_component_fill_ratio:
            continue
        if bbox_edge_touch_count(component_bbox, width, height) >= 3:
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
        original_surface_bbox = surface_bbox
        surface_bbox = focus_far_edge_surface_bbox(surface_bbox, piece, width, config)
        if surface_bbox != original_surface_bbox:
            surface_area = int(
                min(
                    float(surface_area) * surface_bbox[2] / max(1.0, float(original_surface_bbox[2])),
                    float(surface_bbox[2] * surface_bbox[3]),
                )
            )
            target_x, target_y = top_surface_point_from_bbox(surface_bbox, config)

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
        surface_fill = bbox_fill_ratio(surface_area, surface_bbox)
        if surface_fill < min_surface_fill_ratio:
            continue

        distance = math.dist(piece, (target_x, target_y))
        if distance < width * float(target_cfg.get("min_distance_ratio", 0.10)):
            continue
        risk_multiplier, risks = target_candidate_risk_multiplier(
            crop,
            piece,
            piece_bbox,
            (target_x, target_y),
            surface_bbox,
            config,
            width,
            height,
            distance,
        )
        if risk_multiplier <= 0:
            continue
        area_score = min(1.0, area / max(1.0, width * height * 0.025))
        surface_score = min(1.0, surface_area / max(1.0, width * height * 0.010)) * surface_quality
        distance_score = min(1.0, distance / max(1.0, width * 0.60))
        vertical_overflow = max(0.0, target_y - (piece[1] + height * 0.10))
        vertical_score = clamp(1.0 - vertical_overflow / max(1.0, height * 0.45), 0.42, 1.0)
        shape_score = 1.0 - 0.25 * clamp((aspect_ratio - 1.0) / max(0.1, max_aspect_ratio - 1.0), 0.0, 1.0)
        fill_score = 0.5 * clamp(component_fill / 0.30, 0.0, 1.0) + 0.5 * clamp(
            surface_fill / 0.42,
            0.0,
            1.0,
        )
        edge_touches = bbox_edge_touch_count(surface_bbox, width, height)
        edge_score = 1.0 if edge_touches == 0 else 0.92 if edge_touches == 1 else 0.78
        score = (
            0.26 * area_score
            + 0.25 * distance_score
            + 0.26 * surface_score
            + 0.13 * vertical_score
            + 0.10 * fill_score
        ) * shape_score * edge_score * confidence_scale * risk_multiplier
        confidence = clamp(score * (0.76 + 0.24 * surface_quality), 0.0, 1.0)
        candidates.append(
            TargetCandidate(
                point=(target_x, target_y),
                bbox=surface_bbox,
                score=score,
                confidence=confidence,
                source=source,
                risks=risks,
            )
        )

    return candidates


def choose_target_from_mask(
    crop: Any,
    mask: Any,
    piece: tuple[int, int],
    piece_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
    confidence_scale: float,
) -> tuple[tuple[int, int], tuple[int, int, int, int], float] | None:
    candidates = collect_target_candidates(
        crop,
        mask,
        piece,
        piece_bbox,
        config,
        confidence_scale,
        "mask",
    )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: candidate.score)
    return best.point, best.bbox, best.confidence


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
    candidates = collect_target_candidates(
        crop,
        diff_mask,
        piece,
        piece_bbox,
        config,
        confidence_scale=1.0,
        source="diff",
    )

    edge_mask = build_edge_mask(crop)
    edge_mask = side_mask_for_target(edge_mask, piece, config)
    edge_mask = exclude_piece_area(edge_mask, piece_bbox, config)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates.extend(
        collect_target_candidates(
            crop,
            edge_mask,
            piece,
            piece_bbox,
            config,
            confidence_scale=0.72,
            source="edge",
        )
    )
    if not candidates:
        raise RecognitionError("Could not detect the next target platform.")
    best = max(candidates, key=lambda candidate: (candidate.score, candidate.confidence))
    source_mask = edge_mask if best.source == "edge" else diff_mask
    return best.point, best.bbox, best.confidence, source_mask


def recognition_strategy_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("default", config),
        (
            "piece_wide",
            deep_merge(
                config,
                {
                    "piece": {
                        "search_top_ratio": 0.14,
                        "hsv_lower": [90, 42, 20],
                        "fallback_hsv_lower": [80, 38, 12],
                    }
                },
            ),
        ),
        (
            "target_strict",
            deep_merge(
                config,
                {
                    "target": {
                        "search_top_ratio": 0.22,
                        "diff_threshold": 20,
                        "min_distance_ratio": 0.14,
                        "current_platform_exclude_pad_ratio": 0.32,
                    }
                },
            ),
        ),
        (
            "target_wide",
            deep_merge(
                config,
                {
                    "target": {
                        "search_top_ratio": 0.14,
                        "search_bottom_extra_ratio": 0.18,
                        "diff_threshold": 12,
                        "max_area_ratio": 0.28,
                    }
                },
            ),
        ),
        (
            "wide_all",
            deep_merge(
                config,
                {
                    "piece": {
                        "search_top_ratio": 0.12,
                        "hsv_lower": [90, 42, 20],
                        "fallback_hsv_lower": [80, 38, 12],
                    },
                    "target": {
                        "search_top_ratio": 0.14,
                        "search_bottom_extra_ratio": 0.18,
                        "diff_threshold": 12,
                        "max_area_ratio": 0.28,
                    },
                },
            ),
        ),
    ]


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
    label = (
        f"eff={detection.effective_distance_px:.1f}px "
        f"screen={detection.screen_distance_px:.1f}px "
        f"confidence={detection.confidence:.2f}"
    )
    if press_ms is not None:
        label += f" press={press_ms:.0f}ms"
    cv2.putText(debug, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
    cv2.putText(debug, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return debug


def save_recognition_failure_debug(
    frame: Any,
    crop_rect: tuple[int, int, int, int],
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
    message: str,
) -> Path:
    cv2, _ = import_cv()
    debug_path = debug_dir / f"{label}_failed_{timestamp()}.png"
    debug = frame.copy()
    crop_left, crop_top, crop_right, crop_bottom = crop_rect
    cv2.rectangle(debug, (crop_left, crop_top), (crop_right, crop_bottom), (0, 255, 255), 2)
    text = message[:120]
    cv2.putText(debug, "recognition failed", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
    cv2.putText(debug, "recognition failed", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(debug, text, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(debug, text, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return write_debug_image(debug_path, debug, config)


def save_detection_debug(
    frame: Any,
    detection: DetectionResult,
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
    press_ms: float | None = None,
    strategy: str = "default",
) -> Path:
    cv2, _ = import_cv()
    debug_path = debug_dir / f"{label}_{timestamp()}.png"
    debug = draw_debug(frame, detection, press_ms=press_ms)
    if strategy != "default":
        cv2.putText(debug, f"strategy={strategy}", (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(debug, f"strategy={strategy}", (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return write_debug_image(debug_path, debug, config)


def detect_jump(
    frame: Any,
    config: dict[str, Any],
    debug_dir: Path,
    label: str,
    press_ms: float | None = None,
    save_mask: bool = False,
    save_debug: bool = True,
) -> DetectionResult:
    cv2, _ = import_cv()
    crop, crop_rect = crop_game_area(frame, config)
    crop_left, crop_top, _, _ = crop_rect
    if screen_overlay_present(crop, config):
        debug_path = save_recognition_failure_debug(
            frame,
            crop_rect,
            config,
            debug_dir,
            label,
            "A game-over or modal overlay appears to be covering the board.",
        )
        raise RecognitionError(
            f"A game-over or modal overlay appears to be covering the board. Debug image: {debug_path}"
        )
    last_error: RecognitionError | None = None
    selected_strategy = "default"
    best_attempt: tuple[
        str,
        tuple[int, int],
        tuple[int, int, int, int],
        Any,
        tuple[int, int],
        tuple[int, int, int, int],
        float,
        Any,
    ] | None = None
    strategy_accept_confidence = float(
        config["target"].get("strategy_accept_confidence", config.get("confidence_threshold", 0.45))
    )
    for strategy_name, strategy_config in recognition_strategy_configs(config):
        try:
            piece, piece_bbox, piece_mask = find_piece(crop, strategy_config)
            target, target_bbox, confidence, target_mask = find_target(
                crop,
                piece,
                piece_bbox,
                strategy_config,
            )
            attempt = (
                strategy_name,
                piece,
                piece_bbox,
                piece_mask,
                target,
                target_bbox,
                confidence,
                target_mask,
            )
            if best_attempt is None or confidence > best_attempt[6]:
                best_attempt = attempt
            if confidence >= strategy_accept_confidence:
                break
        except RecognitionError as exc:
            last_error = exc
    if best_attempt is None:
        message = str(last_error) if last_error is not None else "Recognition failed."
        if save_mask:
            write_debug_image(
                debug_dir / f"{label}_{timestamp()}_piece_mask.png",
                build_piece_mask(crop, config, fallback=True),
                config,
            )
        debug_path = save_recognition_failure_debug(
            frame,
            crop_rect,
            config,
            debug_dir,
            label,
            message,
        )
        raise RecognitionError(f"{message} Debug image: {debug_path}") from last_error
    (
        selected_strategy,
        piece,
        piece_bbox,
        piece_mask,
        target,
        target_bbox,
        confidence,
        target_mask,
    ) = best_attempt
    piece_full = (piece[0] + crop_left, piece[1] + crop_top)
    target_full = (target[0] + crop_left, target[1] + crop_top)
    dx = float(target_full[0] - piece_full[0])
    dy = float(target_full[1] - piece_full[1])
    screen_distance = math.hypot(dx, dy)
    effective_distance = effective_distance_from_delta(dx, dy, config)

    result = DetectionResult(
        piece=piece_full,
        target=target_full,
        piece_bbox=piece_bbox,
        target_bbox=target_bbox,
        crop_rect=crop_rect,
        dx_px=dx,
        dy_px=dy,
        screen_distance_px=screen_distance,
        effective_distance_px=effective_distance,
        distance_px=effective_distance,
        confidence=confidence,
        debug_path=None,
        piece_median_hsv=sample_piece_median_hsv(crop, piece_bbox),
    )
    if save_mask:
        write_debug_image(
            debug_dir / f"{label}_{timestamp()}_piece_mask.png",
            piece_mask,
            config,
        )
        write_debug_image(
            debug_dir / f"{label}_{timestamp()}_target_mask.png",
            target_mask,
            config,
        )
    if save_debug:
        debug_path = save_detection_debug(
            frame,
            result,
            config,
            debug_dir,
            label,
            press_ms=press_ms,
            strategy=selected_strategy,
        )
        result = replace(result, debug_path=debug_path)
    return result
