from dataclasses import dataclass
from typing import List

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
