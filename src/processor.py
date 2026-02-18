from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectionConfig
from .models import PipelineResult, PipelineStep


class CoinProcessor:
    """Resize, grayscale, normalize, optional invert, blur, Hough circles, ratio-based classify, and draw."""

    # Manual denomination radius ratios (normalized to 1c = 1.0). No absolute mm is used.
    _COIN_RADIUS_RATIO: Dict[int, float] = {
        1: 1.000,
        2: 1.154,
        5: 1.308,
        10: 1.215,
        20: 1.369,
        50: 1.492,
        100: 1.431,
        200: 1.585,
    }
    _RATIO_ERROR_THRESHOLD: float = 0.11
    _COLOR_BRONZE = "bronze"
    _COLOR_GOLD = "gold"
    _COLOR_BIMETAL_1E = "bimetal_gold_ring"
    _COLOR_BIMETAL_2E = "bimetal_silver_ring"
    _COLOR_UNKNOWN = "unknown"

    _COLOR_TO_DENOMS: Dict[str, Tuple[int, ...]] = {
        _COLOR_BRONZE: (1, 2, 5),
        _COLOR_GOLD: (10, 20, 50),
        _COLOR_BIMETAL_1E: (100,),
        _COLOR_BIMETAL_2E: (200,),
        _COLOR_UNKNOWN: (1, 2, 5, 10, 20, 50, 100, 200),
    }

    def __init__(self, config: DetectionConfig):
        self._cfg = config

    def execute(self, img: np.ndarray, filename: str = "Unknown") -> Optional[PipelineResult]:
        if img is None or img.size == 0:
            return None

        img_resized = self._resize(img)
        return self.detect_with_params(
            img_bgr_resized=img_resized,
            dp=self._cfg.HOUGH_DP,
            minDist=self._cfg.HOUGH_MIN_DIST,
            param1=self._cfg.HOUGH_PARAM1,
            param2=self._cfg.HOUGH_PARAM2,
            minRadius=self._cfg.HOUGH_MIN_RADIUS,
            maxRadius=self._cfg.HOUGH_MAX_RADIUS,
            filename=filename,
        )

    def detect_with_params(
        self,
        img_bgr_resized: np.ndarray,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
        filename: str = "LIVE_TUNE",
    ) -> PipelineResult:
        steps: List[PipelineStep] = []

        display_img = img_bgr_resized.copy()
        steps.append(PipelineStep("1. Original", img_bgr_resized, "rgb"))

        gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        mean_brightness = float(np.mean(gray))
        inverted = False
        if mean_brightness < 110:
            gray = cv2.bitwise_not(gray)
            inverted = True
            steps.append(PipelineStep("2a. Inverted (Low Brightness)", gray, "gray"))
        else:
            steps.append(PipelineStep("2. Grayscale", gray, "gray"))

        blurred = cv2.medianBlur(gray, self._cfg.BLUR_KERNEL_SIZE)
        steps.append(PipelineStep("3. Median Blur", blurred, "gray"))

        minRadius = int(max(0, minRadius))
        maxRadius = int(max(minRadius + 1, maxRadius))
        minDist = int(max(1, minDist))
        param1 = int(max(1, param1))
        param2 = int(max(1, param2))
        dp = float(max(0.1, dp))

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

        mask = np.zeros_like(gray)
        coin_count = 0
        coin_labels: List[Optional[int]] = []
        coin_color_labels: List[str] = []
        coin_candidate_denoms: List[List[int]] = []
        coin_tags: List[str] = []
        coin_radii: List[float] = []
        radius_ratio_matrix: List[List[float]] = []
        ratio_fit_errors: List[Optional[float]] = []
        labeled_count = 0
        estimated_value_eur = 0.0

        if circles is not None:
            circles = np.uint16(np.around(circles))[0, :]
            coin_count = int(circles.shape[0])
            coin_radii = [float(c[2]) for c in circles]
            coin_tags = [self._coin_tag(i) for i in range(coin_count)]
            color_features = [self._extract_coin_color_features(img_bgr_resized, c) for c in circles]
            coin_color_labels = self.classify_coin_color_groups(color_features)
            coin_candidate_denoms = [
                list(self._COLOR_TO_DENOMS.get(lbl, self._COLOR_TO_DENOMS[self._COLOR_UNKNOWN]))
                for lbl in coin_color_labels
            ]

            radius_ratio_matrix = self._build_radius_ratio_matrix(coin_radii)
            coin_labels, ratio_fit_errors = self.classify_coin_values_by_ratios(
                ratio_matrix=radius_ratio_matrix,
                candidate_denoms_per_coin=coin_candidate_denoms,
            )

            labeled_count = sum(1 for label in coin_labels if label is not None)
            estimated_value_eur = float(sum(label for label in coin_labels if label is not None)) / 100.0

            for idx, (x, y, r) in enumerate(circles):
                x_i, y_i, r_i = int(x), int(y), int(r)
                cv2.circle(display_img, (x_i, y_i), r_i, (0, 255, 0), 3)
                cv2.circle(display_img, (x_i, y_i), 2, (0, 0, 255), 3)
                cv2.circle(mask, (x_i, y_i), r_i, 255, -1)

                tag = coin_tags[idx]
                label = coin_labels[idx]
                if label is None:
                    draw_text = f"{tag}"
                else:
                    draw_text = f"{tag}:{label}c"

                cv2.putText(
                    display_img,
                    draw_text,
                    (x_i - max(10, r_i // 2), y_i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        steps.append(PipelineStep("4. Detected Circles + Labels", display_img, "rgb"))
        steps.append(PipelineStep("5. Mask (Debug)", mask, "gray"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=inverted,
            source_filename=filename,
            estimated_value_eur=estimated_value_eur,
            labeled_coin_count=labeled_count,
            coin_labels=coin_labels,
            coin_color_labels=coin_color_labels,
            radius_ratio_matrix=radius_ratio_matrix,
            ratio_fit_errors=ratio_fit_errors,
            coin_tags=coin_tags,
            coin_radii=coin_radii,
            coin_candidate_denoms=coin_candidate_denoms,
        )

    def classify_coin_color_groups(self, features: Sequence[Dict[str, float]]) -> List[str]:
        labels: List[str] = []
        for feat in features:
            copper = feat["copper_pct"]
            gold = feat["gold_pct"]
            silver = feat["silver_pct"]
            ring_gold = feat["ring_gold_pct"]
            center_gold = feat["center_gold_pct"]
            ring_silver = feat["ring_silver_pct"]
            center_silver = feat["center_silver_pct"]

            if copper >= 0.38 and gold <= 0.45:
                labels.append(self._COLOR_BRONZE)
                continue
            if ring_gold >= 0.44 and center_silver >= 0.44:
                labels.append(self._COLOR_BIMETAL_1E)
                continue
            if ring_silver >= 0.44 and center_gold >= 0.44:
                labels.append(self._COLOR_BIMETAL_2E)
                continue
            if gold >= 0.45 and silver <= 0.50:
                labels.append(self._COLOR_GOLD)
                continue
            labels.append(self._COLOR_UNKNOWN)
        return labels

    def classify_coin_values_by_ratios(
        self,
        ratio_matrix: Sequence[Sequence[float]],
        candidate_denoms_per_coin: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[Optional[int]], List[Optional[float]]]:
        """Classify each coin by ratio consistency, constrained by color-derived denomination candidates."""
        n = len(ratio_matrix)
        if n == 0:
            return [], []

        all_denoms = sorted(self._COIN_RADIUS_RATIO.keys())
        if candidate_denoms_per_coin is None or len(candidate_denoms_per_coin) != n:
            candidate_denoms_per_coin = [all_denoms for _ in range(n)]

        labels: List[Optional[int]] = [None] * n
        fit_errors: List[Optional[float]] = [None] * n

        for i in range(n):
            coin_candidates = list(candidate_denoms_per_coin[i]) or all_denoms
            best_denom = None
            best_score = float("inf")

            for d_anchor in coin_candidates:
                # For each observed ratio r_j/r_i, find best etalon ratio among candidates of coin j.
                per_pair_errors: List[float] = []
                for j in range(n):
                    if j == i:
                        continue
                    obs = float(ratio_matrix[j][i])
                    other_candidates = list(candidate_denoms_per_coin[j]) or all_denoms
                    expected_candidates = [
                        self._COIN_RADIUS_RATIO[d_other] / self._COIN_RADIUS_RATIO[d_anchor]
                        for d_other in other_candidates
                    ]
                    err = min(abs(obs - exp) for exp in expected_candidates)
                    per_pair_errors.append(float(err))

                score = float(np.mean(per_pair_errors)) if per_pair_errors else 0.0
                if score < best_score:
                    best_score = score
                    best_denom = d_anchor

            fit_errors[i] = best_score
            if best_denom is not None and best_score <= self._RATIO_ERROR_THRESHOLD:
                labels[i] = int(best_denom)

        return labels, fit_errors

    def _extract_coin_color_features(self, img_bgr: np.ndarray, circle: np.ndarray) -> Dict[str, float]:
        x, y, r = [int(v) for v in circle]
        h, w = img_bgr.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        r = int(np.clip(r, 1, min(w, h) - 1))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        mask_coin = dist2 <= (r ** 2)
        mask_center = dist2 <= int((0.45 * r) ** 2)
        mask_ring = np.logical_and(mask_coin, ~mask_center)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        coin_hsv = hsv[mask_coin]
        ring_hsv = hsv[mask_ring]
        center_hsv = hsv[mask_center]

        if coin_hsv.size == 0:
            return {
                "copper_pct": 0.0,
                "gold_pct": 0.0,
                "silver_pct": 0.0,
                "ring_gold_pct": 0.0,
                "center_gold_pct": 0.0,
                "ring_silver_pct": 0.0,
                "center_silver_pct": 0.0,
            }

        copper_mask = self._mask_copper(coin_hsv)
        gold_mask = self._mask_gold(coin_hsv)
        silver_mask = self._mask_silver(coin_hsv)
        return {
            "copper_pct": float(np.mean(copper_mask)) if copper_mask.size else 0.0,
            "gold_pct": float(np.mean(gold_mask)) if gold_mask.size else 0.0,
            "silver_pct": float(np.mean(silver_mask)) if silver_mask.size else 0.0,
            "ring_gold_pct": self._pct_gold(ring_hsv),
            "center_gold_pct": self._pct_gold(center_hsv),
            "ring_silver_pct": self._pct_silver(ring_hsv),
            "center_silver_pct": self._pct_silver(center_hsv),
        }

    def _mask_copper(self, hsv_pixels: np.ndarray) -> np.ndarray:
        h = hsv_pixels[:, 0]
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]
        return (h >= 5) & (h <= 25) & (s >= 70) & (v >= 45)

    def _mask_gold(self, hsv_pixels: np.ndarray) -> np.ndarray:
        h = hsv_pixels[:, 0]
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]
        return (h >= 15) & (h <= 40) & (s >= 45) & (v >= 60)

    def _mask_silver(self, hsv_pixels: np.ndarray) -> np.ndarray:
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]
        return (s <= 60) & (v >= 65)

    def _pct_gold(self, hsv_pixels: np.ndarray) -> float:
        if hsv_pixels.size == 0:
            return 0.0
        return float(np.mean(self._mask_gold(hsv_pixels)))

    def _pct_silver(self, hsv_pixels: np.ndarray) -> float:
        if hsv_pixels.size == 0:
            return 0.0
        return float(np.mean(self._mask_silver(hsv_pixels)))

    def _build_radius_ratio_matrix(self, radii: Sequence[float]) -> List[List[float]]:
        n = len(radii)
        if n == 0:
            return []

        matrix: List[List[float]] = []
        for i in range(n):
            row: List[float] = []
            for j in range(n):
                den = float(radii[j])
                ratio = float(radii[i] / den) if den > 1e-6 else 0.0
                row.append(round(ratio, 4))
            matrix.append(row)
        return matrix

    def _coin_tag(self, idx: int) -> str:
        # A..Z, AA..AZ, BA.. etc.
        n = idx + 1
        chars: List[str] = []
        while n > 0:
            n -= 1
            chars.append(chr(ord("A") + (n % 26)))
            n //= 26
        return "".join(reversed(chars))

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == 0:
            return img
        scale = self._cfg.TARGET_WIDTH / w
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(h * scale)))
