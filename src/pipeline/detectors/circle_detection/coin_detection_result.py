"""Detection-stage result models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoinDetectionResult:
    """Detection stage payload with both data and visualization artifacts."""

    circles: np.ndarray | None
    circle_count: int
    hough_params: dict[str, float | int]
    sweep_debug: dict[str, int | float | list[int] | str]
    sweep_results: list[tuple[int, dict[str, int | float]]]
    circles_overlay: np.ndarray
