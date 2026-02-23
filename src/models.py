"""Typed data models for analyzer outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PipelineStep:
    """Single visual step emitted by the analysis pipeline."""

    name: str
    image: np.ndarray
    cmap: str


@dataclass(frozen=True)
class AnalysisResult:
    """Structured output returned by :class:`Analyzer`."""

    source_path: Path | None
    steps: list[PipelineStep]
    circle_count: int
    hough_params: dict[str, float | int]
    debug_info: dict[str, Any]

    @property
    def predicted_value_cents(self) -> int:
        return int(self.debug_info.get("total_cents", 0))

    @property
    def value_counts(self) -> dict[str, int]:
        counts = self.debug_info.get("value_counts", {})
        if not isinstance(counts, dict):
            return {}
        return {str(k): int(v) for k, v in counts.items()}
