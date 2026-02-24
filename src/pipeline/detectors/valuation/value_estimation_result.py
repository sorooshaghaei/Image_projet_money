"""Value estimation result model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueEstimationResult:
    """Intermediate denomination inference outputs before rendering."""

    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int
