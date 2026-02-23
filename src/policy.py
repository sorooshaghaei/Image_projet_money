"""Backward-compatible shim for detector policies.

Canonical location is now :mod:`src.detectors.policy`.
This file intentionally re-exports the same symbols so old imports continue to
work while the project transitions to the new structure.
"""

from __future__ import annotations

from src.detectors.policy import (
    auto_minradius_plateau,
    choose_dynamic_sat_delta_threshold,
    circle_nesting_score,
    radius_spread_metric,
    run_hough_with_params,
)

__all__ = [
    "run_hough_with_params",
    "circle_nesting_score",
    "auto_minradius_plateau",
    "choose_dynamic_sat_delta_threshold",
    "radius_spread_metric",
]
