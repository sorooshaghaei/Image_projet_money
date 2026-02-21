from __future__ import annotations

import cv2
import numpy as np

from .config import ContourSettings


def resize_if_needed(img_bgr: np.ndarray, target_width: int) -> np.ndarray:
    """Downscale only when needed.

    Why:
    - keeps runtime bounded on very large images
    - avoids upscaling, which usually hurts edge/contour quality
    """
    h, w = img_bgr.shape[:2]
    if w <= max(1, int(target_width)):
        return img_bgr.copy()

    scale = float(target_width / max(1, w))
    return cv2.resize(img_bgr, (int(target_width), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Build a normalized grayscale image for detectors.

    The optional inversion keeps coin interiors generally brighter than background so
    threshold-based stages behave more consistently across lighting setups.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if float(np.mean(gray)) < 105.0:
        gray = cv2.bitwise_not(gray)
    return gray


def build_edge_map(gray: np.ndarray, *, threshold1: int = 60, threshold2: int = 150) -> np.ndarray:
    """Canny edges are used for texture/complexity scoring (policy features)."""
    return cv2.Canny(gray, threshold1=int(threshold1), threshold2=int(threshold2))


def build_binary_mask(gray: np.ndarray, cfg: ContourSettings) -> np.ndarray:
    """Create a foreground mask for contour and watershed branches."""
    k = _odd(cfg.blur_kernel)
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Otsu automatically picks a threshold per scene.
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = _border_mask(gray.shape[0], gray.shape[1])
    border_white_ratio = float(np.mean(bw[border > 0] > 0))
    if border_white_ratio > 0.50:
        # If border is mostly white we likely segmented background as foreground; flip it.
        bw = cv2.bitwise_not(bw)

    mk = max(1, int(cfg.morph_kernel))
    kernel = np.ones((mk, mk), dtype=np.uint8)
    # Open removes speckles; close reconnects fragmented coin regions.
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return bw


def _border_mask(h: int, w: int, frac: float = 0.08) -> np.ndarray:
    """Binary mask selecting only outer border pixels."""
    border = max(4, int(round(min(h, w) * frac)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    return mask


def _odd(value: int) -> int:
    """OpenCV kernels for blur/morph are expected to be odd-sized."""
    v = int(max(1, value))
    return v if v % 2 == 1 else v + 1
