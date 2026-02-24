"""Detector-related policy/heuristic helpers.

This module centralizes reusable policy functions that are tightly coupled to
detector behavior. It re-exports canonical implementations from detector
submodules to avoid logic duplication.
"""

from __future__ import annotations

from .circle_detection.coin_detector import (
    auto_minradius_plateau,
    circle_nesting_score,
    run_hough_with_params,
)
from .color_analysis.coin_analyzer import (
    choose_dynamic_sat_delta_threshold,
    radius_spread_metric,
)

__all__ = [
    "run_hough_with_params",
    "circle_nesting_score",
    "auto_minradius_plateau",
    "choose_dynamic_sat_delta_threshold",
    "radius_spread_metric",
]
