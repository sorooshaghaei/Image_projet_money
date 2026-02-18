from math import acos, pi, sqrt
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectionConfig

CircleInt = Tuple[int, int, int]
ScoredCircle = Tuple[float, float, float, float]


class CircleDetector:
    """Multi-source circle proposal and filtering pipeline."""

    def __init__(self, config: DetectionConfig) -> None:
        self._cfg = config

    def detect_ensemble(
        self,
        gray: np.ndarray,
        blurred: np.ndarray,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> List[CircleInt]:
        """
        Multi-source circle proposal pipeline (strict Hough + loose Hough + contour fallback).

        Why ensemble:
        - strict Hough gives precision,
        - loose Hough recovers missed coins,
        - contour fallback helps difficult low-contrast cases.
        """
        support_edges = cv2.Canny(gray, threshold1=45, threshold2=130)

        strict_raw = self._detect_hough_pass(
            blurred=blurred,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
            score=3.0,
        )
        loose_raw = self._detect_hough_pass(
            blurred=blurred,
            dp=max(0.6, dp * self._cfg.HOUGH_LOOSE_DP_SCALE),
            minDist=max(1, int(round(minDist * self._cfg.HOUGH_LOOSE_MIN_DIST_SCALE))),
            param1=param1,
            param2=max(1, int(round(param2 * self._cfg.HOUGH_LOOSE_PARAM2_SCALE))),
            minRadius=minRadius,
            maxRadius=maxRadius,
            score=2.0,
        )
        contour_raw = self._detect_contour_circles(gray=gray, minRadius=minRadius, maxRadius=maxRadius)

        strict = self._merge_circle_candidates(strict_raw)
        loose = self._merge_circle_candidates(loose_raw)
        contour = self._merge_circle_candidates(contour_raw)

        if contour and strict:
            median_radius = float(np.median([c[2] for c in strict]))
            min_allowed_r = max(float(minRadius), 0.60 * median_radius)
            max_allowed_r = min(float(maxRadius) * 1.20, 1.70 * median_radius)
            contour = [cand for cand in contour if min_allowed_r <= cand[2] <= max_allowed_r]

        if not strict:
            # If strict pass misses everything, keep best-supported fallback detections.
            fallback = self._merge_circle_candidates(loose_raw + contour_raw)
            filtered = self._filter_circles_by_support(gray, support_edges, fallback)
            return self._suppress_nested_overlaps(filtered)

        final_circles = list(strict)
        median_radius = float(np.median([c[2] for c in strict]))
        for loose_circle in loose:
            if any(self._is_same_circle_int(loose_circle, c) for c in final_circles):
                continue
            loose_r = float(loose_circle[2])
            if loose_r < 0.55 * median_radius or loose_r > 1.90 * median_radius:
                continue
            support, _, _ = self._circle_support_score(gray, support_edges, loose_circle)
            if support >= self._cfg.LOOSE_MIN_SUPPORT_SCORE:
                final_circles.append(loose_circle)

        # Very small detection sets are often under-segmented; allow one contour backup.
        if len(final_circles) <= 2:
            for contour_circle in contour:
                if any(self._is_same_circle_int(contour_circle, c) for c in final_circles):
                    continue
                support, _, _ = self._circle_support_score(gray, support_edges, contour_circle)
                if support < self._cfg.CIRCLE_MIN_SUPPORT_SCORE:
                    continue
                final_circles.append(contour_circle)
                break

        final_as_candidates = [(float(x), float(y), float(r), 3.0) for x, y, r in final_circles]
        merged = self._merge_circle_candidates(final_as_candidates)
        filtered = self._filter_circles_by_support(gray, support_edges, merged)
        return self._suppress_nested_overlaps(filtered)

    def _detect_hough_pass(
        self,
        blurred: np.ndarray,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
        score: float,
    ) -> List[ScoredCircle]:
        """Run one HoughCircles pass and attach a confidence weight (`score`)."""
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )
        if circles is None:
            return []

        detected = np.squeeze(circles, axis=0)
        if detected.ndim == 1:
            detected = np.expand_dims(detected, axis=0)

        out: List[ScoredCircle] = []
        for x, y, r in detected:
            out.append((float(x), float(y), float(r), float(score)))
        return out

    def _detect_contour_circles(self, gray: np.ndarray, minRadius: int, maxRadius: int) -> List[ScoredCircle]:
        """
        Contour-based circle proposals used as backup when Hough is incomplete.

        Uses circularity + fill-ratio filters to reject irregular blobs.
        """
        eq = cv2.equalizeHist(gray)
        edges = cv2.Canny(eq, threshold1=45, threshold2=130)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        min_r = max(3.0, minRadius * self._cfg.CONTOUR_MIN_RADIUS_SCALE)
        max_r = max(min_r + 1.0, maxRadius * self._cfg.CONTOUR_MAX_RADIUS_SCALE)
        min_area = np.pi * (min_r**2) * 0.55
        max_area = np.pi * (max_r**2) * 1.30

        out: List[ScoredCircle] = []
        contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self._cfg.CONTOUR_MIN_CIRCULARITY:
                continue

            (x, y), r = cv2.minEnclosingCircle(contour)
            if r < min_r or r > max_r:
                continue

            fill_ratio = float(area / (np.pi * (r**2) + 1e-6))
            if fill_ratio < self._cfg.CONTOUR_MIN_FILL_RATIO:
                continue

            score = 1.0 + float(circularity) + 0.2 * min(1.0, fill_ratio)
            out.append((float(x), float(y), float(r), score))
        return out

    def _circle_support_score(self, gray: np.ndarray, edge_map: np.ndarray, circle: CircleInt) -> Tuple[float, float, float]:
        """Score how likely a circle is real using angular edge coverage + contrast."""
        x, y, r = circle
        h, w = gray.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        r = int(np.clip(r, 3, max(3, min(w, h) - 2)))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2

        ring_in = max(1, r - 2)
        ring_out = r + 2
        ring = np.logical_and(dist2 >= (ring_in**2), dist2 <= (ring_out**2))

        inner = dist2 <= int((0.72 * r) ** 2)
        outer_in = r + 3
        outer_out = r + 8
        outer = np.logical_and(dist2 >= (outer_in**2), dist2 <= (outer_out**2))

        # Angular coverage is more robust than edge density when edges are thin/fragmented.
        ring_edges = np.logical_and(ring, edge_map > 0)
        angular_coverage = 0.0
        if np.any(ring_edges):
            ys, xs = np.nonzero(ring_edges)
            angles = np.arctan2(ys.astype(np.float32) - float(y), xs.astype(np.float32) - float(x))
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
            bin_count = 72  # 5-degree bins over 360 degrees.
            bins = np.floor((angles / (2.0 * np.pi)) * bin_count).astype(np.int32)
            bins = np.clip(bins, 0, bin_count - 1)
            angular_coverage = float(np.unique(bins).size / float(bin_count))

        inner_mean = float(np.mean(gray[inner])) if np.any(inner) else 0.0
        outer_mean = float(np.mean(gray[outer])) if np.any(outer) else inner_mean
        contrast = abs(inner_mean - outer_mean) / 255.0

        edge_score = float(np.clip(angular_coverage / 0.42, 0.0, 1.0))
        contrast_score = float(np.clip(contrast / 0.20, 0.0, 1.0))
        support = 0.65 * edge_score + 0.35 * contrast_score
        return support, angular_coverage, contrast

    def _filter_circles_by_support(
        self,
        gray: np.ndarray,
        edge_map: np.ndarray,
        circles: Sequence[CircleInt],
    ) -> List[CircleInt]:
        """Drop weak candidates, keep strongest plausible circles."""
        if not circles:
            return []

        scored: List[Tuple[CircleInt, float]] = []
        for circle in circles:
            support, angular_coverage, contrast = self._circle_support_score(gray, edge_map, circle)
            if (
                support >= self._cfg.CIRCLE_MIN_SUPPORT_SCORE
                and (
                    angular_coverage >= self._cfg.CIRCLE_MIN_ANGULAR_COVERAGE
                    or contrast >= self._cfg.CIRCLE_MIN_CONTRAST
                )
            ):
                scored.append((circle, support))

        if not scored:
            # Fail-safe: if all circles are rejected, keep original candidates ordered by support.
            for circle in circles:
                support, _, _ = self._circle_support_score(gray, edge_map, circle)
                scored.append((circle, support))

        scored.sort(key=lambda item: item[1], reverse=True)
        filtered = [circle for circle, _ in scored]
        if len(filtered) >= 4:
            median_radius = float(np.median([c[2] for c in filtered]))
            bounded = [c for c in filtered if 0.45 * median_radius <= c[2] <= 2.20 * median_radius]
            if bounded:
                filtered = bounded
        return filtered

    def _suppress_nested_overlaps(self, circles: Sequence[CircleInt]) -> List[CircleInt]:
        """Remove small circles that are mostly contained in larger detections."""
        if len(circles) <= 1:
            return list(circles)

        # Keep larger circles first, then discard small circles that mostly overlap bigger ones.
        sorted_by_radius = sorted(circles, key=lambda c: c[2], reverse=True)
        kept: List[CircleInt] = []
        for candidate in sorted_by_radius:
            if any(self._is_nested_overlap(candidate, ref) for ref in kept):
                continue
            kept.append(candidate)

        kept.sort(key=lambda c: (c[1], c[0]))
        return kept

    def _is_nested_overlap(self, candidate: CircleInt, reference: CircleInt) -> bool:
        """Heuristic test for duplicate/nested circle hypotheses."""
        cx, cy, cr = candidate
        rx, ry, rr = reference
        if cr >= 0.92 * rr:
            return False

        d = float(np.hypot(cx - rx, cy - ry))
        if d >= cr + rr:
            return False

        # Candidate fully inside a larger reference circle.
        if d + cr <= 1.05 * rr:
            return True

        overlap = self._circle_intersection_area(cr, rr, d)
        overlap_ratio_small = overlap / (pi * (cr**2) + 1e-6)
        return overlap_ratio_small >= 0.52 and cr <= 0.80 * rr

    def _circle_intersection_area(self, r1: float, r2: float, d: float) -> float:
        """Exact geometric overlap area between two circles."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return pi * min(r1, r2) ** 2
        if d <= 1e-6:
            return pi * min(r1, r2) ** 2

        alpha = acos(np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0))
        beta = acos(np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0))
        lens = 0.5 * sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
        return (r1 * r1 * alpha) + (r2 * r2 * beta) - lens

    def _merge_circle_candidates(self, candidates: Sequence[ScoredCircle]) -> List[CircleInt]:
        """Merge near-duplicate candidates while preserving highest-score proposals."""
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
        kept: List[ScoredCircle] = []
        for candidate in sorted_candidates:
            if candidate[2] <= 0.0:
                continue
            if any(self._is_same_circle(candidate, selected) for selected in kept):
                continue
            kept.append(candidate)

        circles = [(int(round(x)), int(round(y)), int(round(r))) for x, y, r, _ in kept if r > 0.0]
        circles.sort(key=lambda c: (c[1], c[0]))
        return circles

    def _is_same_circle(self, a: ScoredCircle, b: ScoredCircle) -> bool:
        """Near-equality test for two circle hypotheses."""
        ax, ay, ar, _ = a
        bx, by, br, _ = b
        center_dist = float(np.hypot(ax - bx, ay - by))
        max_r = max(ar, br)
        radius_delta = abs(ar - br)

        center_threshold = self._cfg.MERGE_CENTER_DIST_SCALE * max_r + 3.0
        radius_threshold = self._cfg.MERGE_RADIUS_DIFF_SCALE * max_r + 2.0
        return center_dist <= center_threshold and radius_delta <= radius_threshold

    def _is_same_circle_int(self, a: CircleInt, b: CircleInt) -> bool:
        """Integer wrapper over `_is_same_circle`."""
        ax, ay, ar = a
        bx, by, br = b
        return self._is_same_circle(
            (float(ax), float(ay), float(ar), 0.0),
            (float(bx), float(by), float(br), 0.0),
        )
