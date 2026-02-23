"""Configuration model for coin color analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoinAnalyzerConfig:
    """Configuration knobs for inner/border coin color classification."""

    border_ratio: float = 0.24
    sat_delta_threshold: float | None = None
    bimetal_mode: str = "hybrid"
    material_mode: str = "hsv"
