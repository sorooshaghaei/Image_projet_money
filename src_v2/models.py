from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PipelineStep:
    """One visual stage of the pipeline (used for debug/visualization panels)."""

    name: str
    image: np.ndarray
    cmap: str


@dataclass
class PipelineResult:
    """Full output bundle produced for one image."""

    steps: List[PipelineStep]
    coin_count: int
    is_inverted: bool
    source_filename: str
    estimated_value_eur: float = 0.0
    labeled_coin_count: int = 0
    coin_labels: List[Optional[int]] = field(default_factory=list)
    coin_color_labels: List[str] = field(default_factory=list)
    radius_ratio_matrix: List[List[float]] = field(default_factory=list)
    ratio_fit_errors: List[Optional[float]] = field(default_factory=list)
    coin_tags: List[str] = field(default_factory=list)
    coin_radii: List[float] = field(default_factory=list)
    coin_candidate_denoms: List[List[int]] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    quality_rejected: bool = False
    scene_profile: str = "default"
    scene_metrics: Dict[str, float] = field(default_factory=dict)
