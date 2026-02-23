"""Image preprocessing helpers used by the analyzer."""

from __future__ import annotations

from onefiler import ImagePreprocessing, PreprocessingResult, auto_hough_param1_from_gradient
from onefiler import _normalize_blur_mode as _legacy_normalize_blur_mode
from onefiler import _normalize_odd_ksize as _legacy_normalize_odd_ksize


def normalize_blur_mode(mode: str) -> str:
    """Normalize blur mode aliases to the canonical pipeline value."""

    return _legacy_normalize_blur_mode(mode)


def normalize_odd_ksize(ksize: int) -> int:
    """Return a positive odd kernel size for OpenCV blur operations."""

    return _legacy_normalize_odd_ksize(ksize)


__all__ = [
    "PreprocessingResult",
    "ImagePreprocessing",
    "auto_hough_param1_from_gradient",
    "normalize_blur_mode",
    "normalize_odd_ksize",
]
