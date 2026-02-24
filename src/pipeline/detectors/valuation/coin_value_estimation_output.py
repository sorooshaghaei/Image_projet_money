"""Output payload for full value-estimation stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoinValueEstimationOutput:
    """Full output payload used by analyzer/runner for value stage."""

    split_rgb: np.ndarray
    split_stats: list[dict]
    value_labeled_rgb: np.ndarray
    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int
