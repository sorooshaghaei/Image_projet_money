from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    r: int


@dataclass(frozen=True)
class SceneMetrics:
    border_cv: float
    edge_density: float
    texture_score: float
    contour_merge_score: float
    hough_overlap_pairs: int
    likely_overlap: bool
    background_label: str


@dataclass
class DebugFrames:
    overlay: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    binary_mask: np.ndarray
    watershed_markers: np.ndarray


@dataclass
class AnalysisResult:
    source_path: str
    short_path: str
    selected_method: str
    circles: List[Circle]
    metrics: SceneMetrics
    frames: DebugFrames
    estimated_value_eur: float = 0.0
    labeled_coin_count: int = 0
    coin_labels: List[Optional[int]] = field(default_factory=list)
    coin_color_labels: List[str] = field(default_factory=list)
    ratio_fit_errors: List[Optional[float]] = field(default_factory=list)
