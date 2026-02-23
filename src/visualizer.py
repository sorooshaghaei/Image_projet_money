"""Optional interactive viewer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import onefiler as _legacy

from src.models import AnalysisResult


def _to_legacy_result(result: AnalysisResult) -> _legacy.PipelineResult:
    source_path = result.source_path if result.source_path is not None else Path("<memory>")
    return _legacy.PipelineResult(
        source_path=source_path,
        steps=[_legacy.PipelineStep(name=s.name, image=s.image, cmap=s.cmap) for s in result.steps],
        circle_count=int(result.circle_count),
        hough_params=dict(result.hough_params),
        debug_info=dict(result.debug_info),
    )


class DebugViewer:
    """Thin wrapper over the existing one-file interactive viewer."""

    def __init__(
        self,
        results: Sequence[AnalysisResult],
        cols: int = 3,
        final_only: bool = False,
        debug_export_dir: Path | None = None,
    ):
        self._viewer = _legacy.OneFileViewer(
            [_to_legacy_result(result) for result in results],
            cols=cols,
            final_only=final_only,
            debug_export_dir=debug_export_dir,
        )

    def show(self) -> None:
        """Open the interactive matplotlib viewer."""

        self._viewer.show()


__all__ = ["DebugViewer"]
