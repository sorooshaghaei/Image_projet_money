from typing import Optional, List

import cv2
import numpy as np

from .config import DetectionConfig
from .models import PipelineResult, PipelineStep


class CoinProcessor:
    """Resize, grayscale, normalize, optional invert, blur, Hough circles, and draw."""

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

        if circles is not None:
            circles = np.uint16(np.around(circles))
            coin_count = circles.shape[1]

            for x, y, r in circles[0, :]:
                cv2.circle(display_img, (int(x), int(y)), int(r), (0, 255, 0), 3)
                cv2.circle(display_img, (int(x), int(y)), 2, (0, 0, 255), 3)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        steps.append(PipelineStep("4. Detected Circles", display_img, "rgb"))
        steps.append(PipelineStep("5. Mask (Debug)", mask, "gray"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=inverted,
            source_filename=filename,
        )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == 0:
            return img
        scale = self._cfg.TARGET_WIDTH / w
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(h * scale)))
