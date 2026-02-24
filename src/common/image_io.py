"""Image reading and geometric normalization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr_or_raise(image_path: Path) -> np.ndarray:
    """Read image with OpenCV and raise explicit error if loading fails."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path.resolve()}")
    return image_bgr


def letterbox_resize_to_canvas(image_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize with aspect-ratio preservation and pad onto fixed canvas."""
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image has invalid dimensions")

    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas
