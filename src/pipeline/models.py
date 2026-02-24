"""Core pipeline result models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PipelineStep:
    """One visualization step image in the processing pipeline."""

    name: str
    image: np.ndarray
    cmap: str


@dataclass
class AnalysisResult:
    """Final per-image pipeline result used by CLI and viewer."""

    source_path: Path
    steps: list[PipelineStep]
    circle_count: int
    hough_params: dict[str, float | int]
    debug_info: dict[str, Any]


PipelineResult = AnalysisResult
