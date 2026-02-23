"""Pure preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    hist_norm_bgr: np.ndarray
    hist_norm_rgb: np.ndarray
    clahe_bgr: np.ndarray
    clahe_rgb: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    hist_norm_debug: dict[str, Any]


class ImagePreprocessing:
    def __init__(
        self,
        clahe_enabled: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        histogram_normalization_enabled: bool = False,
        histogram_clip_limit: float = 2.2,
        histogram_tile_grid_size: tuple[int, int] = (8, 8),
        histogram_stretch_percentiles: tuple[float, float] = (1.0, 99.0),
        blur_mode: str = "gauss",
        gauss_ksize: int = 5,
        gauss_sigma: float = 2.0,
    ):
        self._clahe_enabled = bool(clahe_enabled)
        self._clahe_clip_limit = float(clahe_clip_limit)
        self._clahe_tile_grid_size = clahe_tile_grid_size
        self._histogram_normalization_enabled = bool(histogram_normalization_enabled)
        self._histogram_clip_limit = float(histogram_clip_limit)
        self._histogram_tile_grid_size = histogram_tile_grid_size
        self._histogram_stretch_percentiles = histogram_stretch_percentiles
        self._blur_mode = normalize_blur_mode(blur_mode)
        self._gauss_ksize = int(gauss_ksize)
        self._gauss_sigma = float(gauss_sigma)

    @property
    def blur_step_name(self) -> str:
        return "Gaussian Blur" if self._blur_mode == "gauss" else "Median Blur"

    def process(self, image_bgr: np.ndarray) -> PreprocessingResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        hist_norm_bgr = image_bgr
        hist_norm_rgb = image_rgb
        hist_norm_debug = {"enabled": False, "mode": "removed"}
        clahe_bgr = image_bgr
        clahe_rgb = image_rgb

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ksize = normalize_odd_ksize(self._gauss_ksize)
        if self._blur_mode == "gauss":
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), self._gauss_sigma)
        else:
            blurred = gray if ksize <= 1 else cv2.medianBlur(gray, ksize)

        return PreprocessingResult(
            image_bgr=image_bgr,
            image_rgb=image_rgb,
            hist_norm_bgr=hist_norm_bgr,
            hist_norm_rgb=hist_norm_rgb,
            clahe_bgr=clahe_bgr,
            clahe_rgb=clahe_rgb,
            gray=gray,
            blurred=blurred,
            hist_norm_debug=hist_norm_debug,
        )


def normalize_blur_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"gauss", "gaussian"}:
        return "gauss"
    if normalized == "median":
        return "median"
    raise ValueError("Invalid blur_mode. Expected one of: 'gauss', 'gaussian', 'median'.")


def normalize_odd_ksize(ksize: int) -> int:
    k = int(ksize)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1


def auto_hough_param1_from_gradient(
    gray_u8: np.ndarray,
    blur_ksize: int = 5,
    perc: float = 90.0,
    scale: float = 1.0,
    clamp: tuple[int, int] = (20, 250),
) -> int:
    if gray_u8.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)

    if blur_ksize >= 3:
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray_proc = cv2.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)
    else:
        gray_proc = gray_u8

    gx = cv2.Scharr(gray_proc, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_proc, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)

    non_zero = magnitude[magnitude > 1e-3]
    if non_zero.size == 0:
        return 100

    threshold = np.percentile(non_zero, perc) * scale
    return int(np.clip(threshold, clamp[0], clamp[1]))
