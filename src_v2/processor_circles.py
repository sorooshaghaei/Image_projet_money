from dataclasses import dataclass
from math import pi
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.processor_circles import CircleDetector as HoughEnsembleDetector

from .config import DetectionConfig

CircleInt = Tuple[int, int, int]


@dataclass(frozen=True)
class GeometryDetectionOutput:
    circles: List[CircleInt]
    distance_map: np.ndarray
    binary_mask: np.ndarray
    markers_viz: np.ndarray
    edge_like_components: int


class CoinGeometryDetector:
    """
    Classical geometry pipeline:
    1) border-background color distance mask,
    2) watershed instance separation,
    3) contour-based geometric filtering,
    4) Hough-ensemble proposals,
    5) adaptive arbitration between mask and Hough outputs.
    """

    def __init__(self, config: DetectionConfig):
        self._cfg = config
        self._hough_ensemble = HoughEnsembleDetector(config)

    def detect(
        self,
        img_bgr: np.ndarray,
        gray: np.ndarray,
        blur: np.ndarray,
        *,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
        scene_profile: str,
        scene_metrics: Dict[str, float],
    ) -> GeometryDetectionOutput:
        distance_map, mask = self._build_background_distance_mask(img_bgr, scene_profile)
        markers = self._split_touching_instances(mask, img_bgr)

        mask_circles, edge_like = self._circles_from_markers(
            markers=markers,
            min_radius=minRadius,
            max_radius=maxRadius,
            image_area=img_bgr.shape[0] * img_bgr.shape[1],
        )

        mask_circles = self._merge_duplicates(mask_circles)
        hough_circles = self._hough_ensemble.detect_ensemble(
            gray=gray,
            blurred=blur,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        circles = self._choose_hybrid_circles(
            mask_circles=mask_circles,
            hough_circles=hough_circles,
            scene_profile=scene_profile,
            scene_metrics=scene_metrics,
        )
        if not circles:
            circles = self._add_hough_fallback(
                gray=gray,
                blur=blur,
                existing=mask_circles,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius,
            )
        circles = self._merge_duplicates(circles)
        circles.sort(key=lambda c: (c[1], c[0]))

        return GeometryDetectionOutput(
            circles=circles,
            distance_map=distance_map,
            binary_mask=mask,
            markers_viz=self._markers_to_rgb(markers),
            edge_like_components=edge_like,
        )

    def _choose_hybrid_circles(
        self,
        *,
        mask_circles: Sequence[CircleInt],
        hough_circles: Sequence[CircleInt],
        scene_profile: str,
        scene_metrics: Dict[str, float],
    ) -> List[CircleInt]:
        if not hough_circles and mask_circles:
            return list(mask_circles)
        if not hough_circles and not mask_circles:
            return []

        if not bool(self._cfg.HYBRID_POLICY_ENABLED):
            return list(hough_circles)

        hough_count = int(len(hough_circles))
        mask_count = int(len(mask_circles))
        count_gap = hough_count - mask_count

        border_std = float(scene_metrics.get("border_std", 0.0))
        border_edge_ratio = float(scene_metrics.get("border_edge_ratio", 0.0))
        edge_density = float(scene_metrics.get("edge_density", 0.0))

        prefer_mask = (
            scene_profile == "strict"
            and count_gap >= int(self._cfg.HYBRID_MASK_OVER_HOUGH_MIN_GAP)
            and hough_count >= int(self._cfg.HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT)
            and mask_count <= int(self._cfg.HYBRID_MASK_OVER_HOUGH_MAX_MASK_COUNT)
            and border_std >= float(self._cfg.HYBRID_MASK_OVER_HOUGH_MIN_BORDER_STD)
            and border_edge_ratio >= float(self._cfg.HYBRID_MASK_OVER_HOUGH_MIN_BORDER_EDGE)
            and (
                mask_count >= int(self._cfg.HYBRID_MASK_OVER_HOUGH_MIN_MASK_COUNT)
                or edge_density >= float(self._cfg.HYBRID_MASK_ZERO_ALLOWED_MIN_EDGE_DENSITY)
            )
        )
        return list(mask_circles) if prefer_mask else list(hough_circles)

    def _add_hough_fallback(
        self,
        *,
        gray: np.ndarray,
        blur: np.ndarray,
        existing: Sequence[CircleInt],
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> List[CircleInt]:
        if not bool(self._cfg.HOUGH_FALLBACK_ENABLED):
            return list(existing)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=max(0.5, float(dp)),
            minDist=max(1, int(minDist)),
            param1=max(1, int(param1)),
            param2=max(1, int(param2)),
            minRadius=max(0, int(minRadius)),
            maxRadius=max(int(minRadius) + 1, int(maxRadius)),
        )
        if circles is None:
            return list(existing)

        detected = np.squeeze(circles, axis=0)
        if detected.ndim == 1:
            detected = np.expand_dims(detected, axis=0)

        out: List[CircleInt] = list(existing)
        added = 0
        edge_map = cv2.Canny(gray, threshold1=70, threshold2=160)
        for x, y, r in detected:
            cand = (int(round(x)), int(round(y)), int(round(r)))
            if any(self._is_duplicate(cand, ref) for ref in out):
                continue
            if not self._hough_support_ok(gray=gray, edge_map=edge_map, circle=cand):
                continue
            out.append(cand)
            added += 1
            if added >= int(max(0, self._cfg.HOUGH_MAX_ADDED_CIRCLES)):
                break

        return self._merge_duplicates(out)

    def _build_background_distance_mask(self, img_bgr: np.ndarray, scene_profile: str) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]

        border = max(4, int(round(min(h, w) * float(self._cfg.SCENE_BORDER_FRACTION))))
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[:border, :] = 1
        border_mask[-border:, :] = 1
        border_mask[:, :border] = 1
        border_mask[:, -border:] = 1

        bg_pixels = lab[border_mask > 0]
        if bg_pixels.size == 0:
            bg_pixels = lab.reshape(-1, 3)
        bg_med = np.median(bg_pixels, axis=0).astype(np.float32)

        diff = lab - bg_med[None, None, :]
        dist = np.sqrt(
            (float(self._cfg.BG_DIST_WEIGHT_L) * diff[:, :, 0]) ** 2
            + (float(self._cfg.BG_DIST_WEIGHT_A) * diff[:, :, 1]) ** 2
            + (float(self._cfg.BG_DIST_WEIGHT_B) * diff[:, :, 2]) ** 2
        )

        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu_thr, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        offset = int(self._cfg.MASK_OTSU_OFFSET)
        if scene_profile == "strict":
            offset += int(self._cfg.ADAPTIVE_STRICT_MASK_OFFSET)
        elif scene_profile == "relaxed":
            offset += int(self._cfg.ADAPTIVE_RELAXED_MASK_OFFSET)

        final_thr = int(max(self._cfg.MASK_MIN_DISTANCE_THRESHOLD, min(255, round(float(otsu_thr) + float(offset)))))
        mask = np.where(dist_u8 >= final_thr, 255, 0).astype(np.uint8)

        open_k = max(1, int(self._cfg.MASK_OPEN_KERNEL))
        close_k = max(1, int(self._cfg.MASK_CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), dtype=np.uint8))

        if int(self._cfg.MASK_ERODE_ITERS) > 0:
            mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=int(self._cfg.MASK_ERODE_ITERS))
        if int(self._cfg.MASK_DILATE_ITERS) > 0:
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=int(self._cfg.MASK_DILATE_ITERS))

        mask = self._remove_bad_components(mask)
        return dist_u8, mask

    def _remove_bad_components(self, mask: np.ndarray) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask

        h, w = mask.shape[:2]
        img_area = float(h * w)
        min_area = int(max(1, self._cfg.MASK_MIN_COMPONENT_AREA))
        max_area = float(self._cfg.MASK_MAX_COMPONENT_AREA_RATIO) * img_area

        out = np.zeros_like(mask)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            out[labels == i] = 255
        return out

    def _split_touching_instances(self, mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        if cv2.countNonZero(mask) <= 0:
            return np.zeros_like(mask, dtype=np.int32)

        work = mask.copy()
        open_k = max(1, int(self._cfg.WATERSHED_OPEN_KERNEL))
        close_k = max(1, int(self._cfg.WATERSHED_CLOSE_KERNEL))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, np.ones((open_k, open_k), dtype=np.uint8))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((close_k, close_k), dtype=np.uint8))

        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist)) if dist.size > 0 else 0.0
        if max_dist <= 1e-6:
            return self._connected_component_markers(work)

        fg_ratio = float(np.clip(self._cfg.WATERSHED_DT_FG_RATIO, 0.20, 0.85))
        sure_fg = np.where(dist >= fg_ratio * max_dist, 255, 0).astype(np.uint8)
        sure_fg = self._remove_small_seeds(sure_fg)

        sure_bg = cv2.dilate(work, np.ones((3, 3), dtype=np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num, markers = cv2.connectedComponents(sure_fg)
        if num <= 1:
            return self._connected_component_markers(work)

        markers = markers + 1
        markers[unknown > 0] = 0
        markers = cv2.watershed(img_bgr.copy(), markers.astype(np.int32))
        out = np.where(markers > 1, markers, 0).astype(np.int32)
        return out if np.max(out) > 0 else self._connected_component_markers(work)

    def _remove_small_seeds(self, seeds: np.ndarray) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
        if num <= 1:
            return seeds

        min_seed = int(max(1, self._cfg.WATERSHED_MIN_SEED_AREA))
        out = np.zeros_like(seeds)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_seed:
                out[labels == i] = 255
        return out

    def _connected_component_markers(self, mask: np.ndarray) -> np.ndarray:
        num, labels = cv2.connectedComponents(mask)
        if num <= 1:
            return np.zeros_like(mask, dtype=np.int32)
        return labels.astype(np.int32)

    def _circles_from_markers(
        self,
        *,
        markers: np.ndarray,
        min_radius: int,
        max_radius: int,
        image_area: int,
    ) -> Tuple[List[CircleInt], int]:
        circles_scored: List[Tuple[float, CircleInt]] = []
        edge_like_components = 0

        geom_min_radius = int(max(min_radius, self._cfg.GEOM_MIN_RADIUS))
        geom_max_radius = int(min(max_radius, self._cfg.GEOM_MAX_RADIUS))
        geom_min_area = int(max(self._cfg.GEOM_MIN_AREA, pi * (geom_min_radius**2) * 0.45))
        geom_max_area = float(self._cfg.GEOM_MAX_AREA_RATIO) * float(image_area)

        for label in np.unique(markers):
            if label <= 0:
                continue
            region = np.where(markers == label, 255, 0).astype(np.uint8)
            if cv2.countNonZero(region) <= 0:
                continue

            contours_info = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            area = float(cv2.contourArea(contour))
            if area < geom_min_area or area > geom_max_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float((4.0 * pi * area) / (perimeter * perimeter))
            if circularity < float(self._cfg.GEOM_MIN_CIRCULARITY):
                continue

            aspect_ratio = 1.0
            if len(contour) >= 5:
                _, (maj, min_), _ = cv2.fitEllipse(contour)
                if min_ > 1e-6:
                    aspect_ratio = float(max(maj, min_) / min(maj, min_))
            if aspect_ratio >= float(self._cfg.EDGE_STACK_ASPECT_RATIO):
                edge_like_components += 1
            if aspect_ratio > float(self._cfg.GEOM_MAX_ASPECT_RATIO):
                continue

            (x, y), r = cv2.minEnclosingCircle(contour)
            r = float(r)
            if r < geom_min_radius or r > geom_max_radius:
                continue

            fill_ratio = float(area / (pi * (r**2) + 1e-6))
            if fill_ratio < float(self._cfg.GEOM_MIN_FILL_RATIO) or fill_ratio > float(self._cfg.GEOM_MAX_FILL_RATIO):
                continue

            score = circularity + 0.35 * max(0.0, min(1.0, fill_ratio))
            circles_scored.append((score, (int(round(x)), int(round(y)), int(round(r)))))

        circles_scored.sort(key=lambda it: it[0], reverse=True)
        return [c for _, c in circles_scored], edge_like_components

    def _merge_duplicates(self, circles: Sequence[CircleInt]) -> List[CircleInt]:
        out: List[CircleInt] = []
        for cand in circles:
            if any(self._is_duplicate(cand, ref) for ref in out):
                continue
            out.append(cand)
        out.sort(key=lambda c: (c[1], c[0]))
        return out

    def _is_duplicate(self, a: CircleInt, b: CircleInt) -> bool:
        ax, ay, ar = a
        bx, by, br = b
        max_r = float(max(ar, br))
        center_dist = float(np.hypot(float(ax - bx), float(ay - by)))
        radius_diff = float(abs(ar - br))

        center_thr = float(self._cfg.GEOM_DUPLICATE_CENTER_SCALE) * max_r + 2.0
        radius_thr = float(self._cfg.GEOM_DUPLICATE_RADIUS_SCALE) * max_r + 2.0
        return center_dist <= center_thr and radius_diff <= radius_thr

    def _hough_support_ok(self, *, gray: np.ndarray, edge_map: np.ndarray, circle: CircleInt) -> bool:
        x, y, r = circle
        h, w = gray.shape[:2]
        x = int(np.clip(x, 0, max(0, w - 1)))
        y = int(np.clip(y, 0, max(0, h - 1)))
        r = int(np.clip(r, 3, max(3, min(h, w) - 2)))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2

        ring = np.logical_and(dist2 >= (max(1, r - 2) ** 2), dist2 <= ((r + 2) ** 2))
        ring_edges = np.logical_and(ring, edge_map > 0)

        angular_coverage = 0.0
        if np.any(ring_edges):
            ys, xs = np.nonzero(ring_edges)
            angles = np.arctan2(ys.astype(np.float32) - float(y), xs.astype(np.float32) - float(x))
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
            bins = np.floor((angles / (2.0 * np.pi)) * 72).astype(np.int32)
            bins = np.clip(bins, 0, 71)
            angular_coverage = float(np.unique(bins).size / 72.0)

        inner = dist2 <= int((0.70 * r) ** 2)
        outer = np.logical_and(dist2 >= ((r + 3) ** 2), dist2 <= ((r + 8) ** 2))
        inner_mean = float(np.mean(gray[inner])) if np.any(inner) else 0.0
        outer_mean = float(np.mean(gray[outer])) if np.any(outer) else inner_mean
        contrast = float(abs(inner_mean - outer_mean) / 255.0)

        return (
            angular_coverage >= float(self._cfg.HOUGH_REQUIRED_ANGULAR_COVERAGE)
            or contrast >= float(self._cfg.HOUGH_REQUIRED_CONTRAST)
        )

    def _markers_to_rgb(self, markers: np.ndarray) -> np.ndarray:
        h, w = markers.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        labels = [int(v) for v in np.unique(markers) if int(v) > 0]
        for label in labels:
            color = (
                (53 * label + 67) % 255,
                (97 * label + 29) % 255,
                (191 * label + 11) % 255,
            )
            out[markers == label] = color
        return out
