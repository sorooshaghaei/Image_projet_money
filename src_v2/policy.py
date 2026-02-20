from typing import Literal, Tuple

import numpy as np

from .config import PolicySettings

BackgroundLabel = Literal["easy", "medium", "difficult"]
MethodLabel = Literal["contours", "hough", "watershed", "hough+watershed"]


def classify_background(border_cv: float, edge_density: float, cfg: PolicySettings) -> Tuple[BackgroundLabel, float]:
    """
    Label scene difficulty from border texture and edge density.

    Lower score means cleaner background.
    """
    edge_term = min(1.0, max(0.0, edge_density * 2.5))
    texture_score = float(np.clip(0.65 * border_cv + 0.35 * edge_term, 0.0, 1.0))

    if texture_score < float(cfg.easy_threshold):
        return "easy", texture_score
    if texture_score < float(cfg.medium_threshold):
        return "medium", texture_score
    return "difficult", texture_score


def choose_auto_method(background_label: BackgroundLabel, likely_overlap: bool, hough_count: int) -> MethodLabel:
    """Route to an algorithm according to the requested strategy."""
    if likely_overlap:
        return "hough+watershed" if hough_count > 0 else "watershed"
    if background_label == "easy":
        return "contours"
    return "hough"
