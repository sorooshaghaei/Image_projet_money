from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .coin_metadata import COLOR_UNKNOWN
from .config import DetectionConfig
from .models import PipelineResult, PipelineStep
from .processor_circles import CircleDetector
from .processor_color import CoinColorClassifier
from .processor_scale import ScaleValueClassifier


@dataclass
class _ClassificationResult:
    coin_labels: List[Optional[int]]
    coin_color_labels: List[str]
    coin_candidate_denoms: List[List[int]]
    radius_ratio_matrix: List[List[float]]
    ratio_fit_errors: List[Optional[float]]
    labeled_count: int
    estimated_value_eur: float


class CoinProcessor:
    """
    Core computer-vision pipeline for euro coin detection and denomination inference.

    Design choices:
    - Detect circles first (geometry is more stable than color).
    - Use color only to constrain plausible denominations.
    - Fit one global px/mm scale per image for consistent value assignment.
    """

    _COLOR_MISMATCH_PENALTY: float = 0.10
    # High-confidence bimetal guardrail to avoid mapping 1EUR/2EUR to tiny bronze coins.
    _BIMETAL_HIGH_CONFIDENCE: float = 0.58
    _BIMETAL_STRONG_2E_MARGIN: float = 0.10

    def __init__(self, config: DetectionConfig):
        self._cfg = config
        self._circle_detector = CircleDetector(config)
        self._color_classifier = CoinColorClassifier(
            bimetal_high_confidence=self._BIMETAL_HIGH_CONFIDENCE,
            bimetal_strong_2e_margin=self._BIMETAL_STRONG_2E_MARGIN,
        )
        self._scale_classifier = ScaleValueClassifier(
            color_mismatch_penalty=self._COLOR_MISMATCH_PENALTY,
        )

    def execute(self, img: np.ndarray, filename: str = "Unknown") -> Optional[PipelineResult]:
        """Default execution path using configuration-defined Hough parameters."""
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
        classify: bool = True,
    ) -> PipelineResult:
        """
        Full pipeline entry with explicit tunable parameters.

        Used by:
        - batch processing (`classify=True`)
        - interactive tuner (`classify` optionally off for speed)
        """
        steps: List[PipelineStep] = []
        display_img = img_bgr_resized.copy()
        steps.append(PipelineStep("1. Original", img_bgr_resized, "rgb"))

        gray, blurred, inverted = self._prepare_gray_and_blur(img_bgr_resized, steps)
        dp, minDist, param1, param2, minRadius, maxRadius = self._sanitize_hough_params(
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        circles = self._circle_detector.detect_ensemble(
            gray=gray,
            blurred=blurred,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        mask = np.zeros_like(gray)
        coin_count = len(circles)
        coin_tags = [self._coin_tag(i) for i in range(coin_count)]
        coin_radii = [float(c[2]) for c in circles]

        if classify:
            class_result = self._classify_circles(img_bgr_resized, circles, coin_radii)
        else:
            class_result = self._build_unclassified_result(coin_count)

        if circles:
            self._draw_coin_annotations(
                display_img=display_img,
                mask=mask,
                circles=circles,
                coin_tags=coin_tags,
                coin_labels=class_result.coin_labels,
            )

        steps.append(PipelineStep("4. Detected Circles + Labels", display_img, "rgb"))
        steps.append(PipelineStep("5. Mask (Debug)", mask, "gray"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=inverted,
            source_filename=filename,
            estimated_value_eur=class_result.estimated_value_eur,
            labeled_coin_count=class_result.labeled_count,
            coin_labels=class_result.coin_labels,
            coin_color_labels=class_result.coin_color_labels,
            radius_ratio_matrix=class_result.radius_ratio_matrix,
            ratio_fit_errors=class_result.ratio_fit_errors,
            coin_tags=coin_tags,
            coin_radii=coin_radii,
            coin_candidate_denoms=class_result.coin_candidate_denoms,
        )

    def classify_coin_color_groups(self, features: Sequence[Dict[str, float]]) -> Tuple[List[str], List[Dict[str, float]]]:
        """Map raw color features to color-group labels + per-group confidence scores."""
        return self._color_classifier.classify_color_groups(features)

    def classify_coin_values_by_scale_mm(
        self,
        radii_px: Sequence[float],
        candidate_denoms_per_coin: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[float], float]:
        """
        Fit one global scale (px per mm diameter) and assign each coin to the closest real euro diameter.
        Color-derived candidates are used as a soft penalty, not a hard constraint.
        """
        return self._scale_classifier.classify(radii_px, candidate_denoms_per_coin)

    def _prepare_gray_and_blur(
        self,
        img_bgr_resized: np.ndarray,
        steps: List[PipelineStep],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        mean_brightness = float(np.mean(gray))
        inverted = False
        if mean_brightness < 110:
            # Dark images can produce weak ring edges; inversion often improves circle gradients.
            gray = cv2.bitwise_not(gray)
            inverted = True
            steps.append(PipelineStep("2a. Inverted (Low Brightness)", gray, "gray"))
        else:
            steps.append(PipelineStep("2. Grayscale", gray, "gray"))

        blur_kernel = int(max(3, self._cfg.BLUR_KERNEL_SIZE))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.medianBlur(gray, blur_kernel)
        steps.append(PipelineStep("3. Median Blur", blurred, "gray"))
        return gray, blurred, inverted

    def _sanitize_hough_params(
        self,
        *,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> Tuple[float, int, int, int, int, int]:
        safe_min_radius = int(max(0, minRadius))
        safe_max_radius = int(max(safe_min_radius + 1, maxRadius))
        safe_min_dist = int(max(1, minDist))
        safe_param1 = int(max(1, param1))
        safe_param2 = int(max(1, param2))
        safe_dp = float(max(0.1, dp))
        return safe_dp, safe_min_dist, safe_param1, safe_param2, safe_min_radius, safe_max_radius

    def _classify_circles(
        self,
        img_bgr_resized: np.ndarray,
        circles: Sequence[Tuple[int, int, int]],
        coin_radii: Sequence[float],
    ) -> _ClassificationResult:
        color_features = [self._color_classifier.extract_coin_features(img_bgr=img_bgr_resized, circle=np.asarray(c)) for c in circles]
        coin_color_labels, color_group_scores = self.classify_coin_color_groups(color_features)
        coin_candidate_denoms = [
            self._color_classifier.candidate_denoms_from_group_scores(scores)
            for scores in color_group_scores
        ]

        radius_ratio_matrix = self._build_radius_ratio_matrix(coin_radii)
        raw_labels, ratio_fit_errors, _ = self.classify_coin_values_by_scale_mm(
            radii_px=coin_radii,
            candidate_denoms_per_coin=coin_candidate_denoms,
        )
        coin_labels: List[Optional[int]] = []
        for label, err in zip(raw_labels, ratio_fit_errors):
            if err is None or float(err) > float(self._cfg.MAX_LABEL_REL_ERROR):
                coin_labels.append(None)
            else:
                coin_labels.append(int(label))

        estimated_value_eur = float(sum(int(label) for label in coin_labels if label is not None)) / 100.0
        labeled_count = sum(1 for label in coin_labels if label is not None)

        return _ClassificationResult(
            coin_labels=coin_labels,
            coin_color_labels=coin_color_labels,
            coin_candidate_denoms=coin_candidate_denoms,
            radius_ratio_matrix=radius_ratio_matrix,
            ratio_fit_errors=[float(err) for err in ratio_fit_errors],
            labeled_count=labeled_count,
            estimated_value_eur=estimated_value_eur,
        )

    def _build_unclassified_result(self, coin_count: int) -> _ClassificationResult:
        return _ClassificationResult(
            coin_labels=[None] * coin_count,
            coin_color_labels=[COLOR_UNKNOWN] * coin_count,
            coin_candidate_denoms=[[] for _ in range(coin_count)],
            radius_ratio_matrix=[],
            ratio_fit_errors=[None] * coin_count,
            labeled_count=0,
            estimated_value_eur=0.0,
        )

    def _draw_coin_annotations(
        self,
        display_img: np.ndarray,
        mask: np.ndarray,
        circles: Sequence[Tuple[int, int, int]],
        coin_tags: Sequence[str],
        coin_labels: Sequence[Optional[int]],
    ) -> None:
        h_img, w_img = display_img.shape[:2]
        for idx, (x, y, r) in enumerate(circles):
            x_i, y_i, r_i = int(x), int(y), int(r)
            cv2.circle(display_img, (x_i, y_i), r_i, (0, 0, 0), 5)
            cv2.circle(display_img, (x_i, y_i), r_i, (0, 255, 0), 2)
            cv2.circle(display_img, (x_i, y_i), 3, (255, 255, 255), -1)
            cv2.circle(display_img, (x_i, y_i), 2, (0, 0, 255), -1)
            cv2.circle(mask, (x_i, y_i), r_i, 255, -1)

            tag = coin_tags[idx]
            label = coin_labels[idx] if idx < len(coin_labels) else None
            draw_text = f"{tag}" if label is None else f"{tag}:{label}c"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.55, min(0.9, r_i / 45.0))
            thickness = max(1, int(round(font_scale * 2.0)))
            (text_w, text_h), baseline = cv2.getTextSize(draw_text, font, font_scale, thickness)

            text_x = x_i - (text_w // 2)
            text_y = y_i - r_i - 8
            if text_y - text_h - baseline < 2:
                text_y = y_i + r_i + text_h + 8

            text_x = int(np.clip(text_x, 2, max(2, w_img - text_w - 2)))
            text_y = int(np.clip(text_y, text_h + baseline + 2, h_img - 2))

            pad = 3
            top_left = (max(0, text_x - pad), max(0, text_y - text_h - baseline - pad))
            bottom_right = (min(w_img - 1, text_x + text_w + pad), min(h_img - 1, text_y + pad))
            cv2.rectangle(display_img, top_left, bottom_right, (0, 0, 0), -1)
            cv2.putText(
                display_img,
                draw_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_img,
                draw_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def _build_radius_ratio_matrix(self, radii: Sequence[float]) -> List[List[float]]:
        """Pairwise radius ratios used for reporting/debug diagnostics."""
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
        """Resize image to configured width while preserving aspect ratio."""
        h, w = img.shape[:2]
        if w == 0:
            return img
        scale = self._cfg.TARGET_WIDTH / w
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(h * scale)))
