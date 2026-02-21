from __future__ import annotations

from math import pi, sqrt
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import DetectionConfig as BaseHoughConfig
from src.processor_circles import CircleDetector as BaseHoughDetector

from .config import ContourSettings, HoughSettings, PolicySettings, WatershedSettings
from .models import Circle

_BASE_HOUGH_DETECTOR = BaseHoughDetector(BaseHoughConfig())


def detect_hough_circles(gray: np.ndarray, cfg: HoughSettings) -> List[Circle]:
    """Detect circles using the shared Hough ensemble from `src`.

    Reusing the original detector keeps behavior close to your stronger `src` baseline
    while allowing `src_v2` policy routing around it.
    """
    blur = cv2.medianBlur(gray, _odd(15))
    min_r = max(1, int(cfg.min_radius))
    max_r = max(min_r + 1, int(cfg.max_radius))
    circles = _BASE_HOUGH_DETECTOR.detect_ensemble(
        gray=gray,
        blurred=blur,
        dp=max(0.5, float(cfg.dp)),
        minDist=max(1, int(cfg.min_dist)),
        param1=max(1, int(cfg.param1)),
        param2=max(1, int(cfg.param2)),
        minRadius=min_r,
        maxRadius=max_r,
    )
    return _dedupe([Circle(int(x), int(y), int(r)) for x, y, r in circles])


def detect_contour_circles(mask: np.ndarray, contour_cfg: ContourSettings, policy_cfg: PolicySettings) -> Tuple[List[Circle], float]:
    """Detect near-circular components from the binary mask.

    Returns:
    - candidate circles
    - merge_score: fraction of significant regions that look like merged coins
      (used by policy to infer overlap complexity).
    """
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    circles: List[Circle] = []
    significant = 0
    merge_like = 0
    image_area = float(mask.shape[0] * mask.shape[1])

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(contour_cfg.min_area):
            continue

        significant += 1
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue

        circularity = float((4.0 * pi * area) / (perimeter * perimeter))
        _, _, w, h = cv2.boundingRect(contour)
        aspect = float(max(w, h) / max(1, min(w, h)))

        area_ratio = area / max(1.0, image_area)
        if (
            area_ratio >= float(policy_cfg.contour_merge_area_ratio)
            and circularity < float(policy_cfg.contour_merge_circularity)
            and aspect >= 1.30
        ):
            # Large elongated blobs are often multiple touching coins.
            merge_like += 1

        if circularity < float(contour_cfg.min_circularity):
            continue
        if aspect > float(contour_cfg.max_aspect_ratio):
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circles.append(Circle(int(round(cx)), int(round(cy)), int(round(radius))))

    merge_score = float(merge_like / max(1, significant))
    return _dedupe(circles), merge_score


def detect_watershed_circles(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    watershed_cfg: WatershedSettings,
    contour_cfg: ContourSettings,
) -> Tuple[List[Circle], np.ndarray]:
    """Split merged foreground regions with watershed and fit circles per region."""
    if cv2.countNonZero(mask) <= 0:
        return [], np.zeros_like(img_bgr)

    work = mask.copy()
    open_k = _odd(watershed_cfg.open_kernel)
    close_k = _odd(watershed_cfg.close_kernel)
    # Stabilize seeds before distance transform.
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, np.ones((open_k, open_k), dtype=np.uint8))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((close_k, close_k), dtype=np.uint8))

    dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    max_dist = float(np.max(dist)) if dist.size > 0 else 0.0
    if max_dist <= 1e-6:
        return [], np.zeros_like(img_bgr)

    fg_ratio = float(np.clip(watershed_cfg.fg_ratio, 0.20, 0.85))
    # "sure_fg" are seed cores; clipped ratio prevents extremely weak/strong seeding.
    sure_fg = np.where(dist >= (fg_ratio * max_dist), 255, 0).astype(np.uint8)
    sure_fg = _remove_small_components(sure_fg, int(max(1, watershed_cfg.min_seed_area)))

    sure_bg = cv2.dilate(work, np.ones((3, 3), dtype=np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_components, markers = cv2.connectedComponents(sure_fg)
    if num_components <= 1:
        return [], np.zeros_like(img_bgr)

    markers = markers + 1
    markers[unknown > 0] = 0
    markers = cv2.watershed(img_bgr.copy(), markers.astype(np.int32))

    circles: List[Circle] = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        region = np.where(markers == label, 255, 0).astype(np.uint8)
        if cv2.countNonZero(region) < int(max(1, contour_cfg.min_area, watershed_cfg.min_region_area)):
            continue

        contours_info = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < float(max(contour_cfg.min_area, watershed_cfg.min_region_area)):
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circles.append(Circle(int(round(cx)), int(round(cy)), int(round(radius))))

    markers_rgb = _markers_to_rgb(markers)
    return _dedupe(circles), markers_rgb


def count_hough_overlap_pairs(circles: Sequence[Circle], distance_scale: float) -> int:
    """Count close circle pairs as a proxy for overlap/touching density."""
    overlap_pairs = 0
    scale = float(max(0.10, distance_scale))
    for i in range(len(circles)):
        c1 = circles[i]
        for j in range(i + 1, len(circles)):
            c2 = circles[j]
            dist = sqrt(float((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2))
            if dist < (c1.r + c2.r) * scale:
                overlap_pairs += 1
    return overlap_pairs


def merge_circle_sets(primary: Sequence[Circle], secondary: Sequence[Circle]) -> List[Circle]:
    """Union of two detector outputs with geometric de-duplication."""
    out: List[Circle] = []
    for circle in list(primary) + list(secondary):
        if any(_is_duplicate(circle, existing) for existing in out):
            continue
        out.append(circle)
    out.sort(key=lambda c: (c.y, c.x))
    return out


def draw_overlay(
    img_bgr: np.ndarray,
    circles: Sequence[Circle],
    method_label: str,
    coin_labels: Optional[Sequence[Optional[int]]] = None,
) -> np.ndarray:
    """Render circles and optional denomination labels for debugging/UI."""
    overlay = img_bgr.copy()

    def _put_text_with_outline(
        image: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_scale: float,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            1,
            cv2.LINE_AA,
        )

    for idx, circle in enumerate(circles, start=1):
        cv2.circle(overlay, (circle.x, circle.y), circle.r, (0, 255, 0), 2)
        cv2.circle(overlay, (circle.x, circle.y), 2, (0, 0, 255), -1)
        denom_text = ""
        if coin_labels is not None and idx - 1 < len(coin_labels):
            denom = coin_labels[idx - 1]
            denom_text = "?" if denom is None else f" {int(denom)}c"
        _put_text_with_outline(overlay, f"{idx}{denom_text}", (circle.x + 3, circle.y - 4), 0.50)

    _put_text_with_outline(overlay, f"method: {method_label}", (10, 24), 0.65)
    _put_text_with_outline(overlay, f"coins: {len(circles)}", (10, 50), 0.65)
    return overlay


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only connected components above the minimum area threshold."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    out = np.zeros_like(mask)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(max(1, min_area)):
            out[labels == i] = 255
    return out


def _markers_to_rgb(markers: np.ndarray) -> np.ndarray:
    """Colorize watershed labels for human inspection in the visualizer."""
    h, w = markers.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label in np.unique(markers):
        if label <= 1:
            continue
        color = _label_color(int(label))
        out[markers == label] = color
    out[markers == -1] = (255, 255, 255)
    return out


def _label_color(label: int) -> Tuple[int, int, int]:
    """Deterministic pseudo-color for a label id."""
    b = (53 * label + 80) % 255
    g = (97 * label + 20) % 255
    r = (151 * label + 180) % 255
    return int(b), int(g), int(r)


def _dedupe(circles: Sequence[Circle]) -> List[Circle]:
    """Remove duplicate detections and return stable top-to-bottom ordering."""
    out: List[Circle] = []
    for circle in circles:
        if any(_is_duplicate(circle, existing) for existing in out):
            continue
        out.append(circle)
    out.sort(key=lambda c: (c.y, c.x))
    return out


def _is_duplicate(a: Circle, b: Circle) -> bool:
    """Heuristic duplicate test based on center distance and radius similarity."""
    dist = sqrt(float((a.x - b.x) ** 2 + (a.y - b.y) ** 2))
    center_thr = 0.50 * float(max(a.r, b.r))
    radius_thr = 0.42 * float(max(a.r, b.r))
    return dist <= center_thr and abs(float(a.r - b.r)) <= radius_thr


def _odd(value: int) -> int:
    """Return nearest odd integer >= 1 for OpenCV kernel sizes."""
    v = int(max(1, value))
    return v if v % 2 == 1 else v + 1
