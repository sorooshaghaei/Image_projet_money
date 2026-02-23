"""Filesystem and serialization helpers used by runner/viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onefiler as _legacy

from src.models import AnalysisResult

read_bgr_or_raise = _legacy.read_bgr_or_raise
letterbox_resize_to_canvas = _legacy.letterbox_resize_to_canvas
save_pipeline_figure = _legacy.save_pipeline_figure


def to_serializable(value: Any) -> Any:
    """Convert pipeline debug values into JSON-safe structures."""

    return _legacy._to_serializable(value)


def format_total_cents(total_cents: int) -> str:
    """Format cents as ``X EUR YY c``."""

    return _legacy._format_total_cents(total_cents)


def format_signed_cents(diff_cents: int) -> str:
    """Format signed cents as ``+X EUR YY c`` or ``-X EUR YY c``."""

    return _legacy._format_signed_cents(diff_cents)


def group_from_relative_path(relative_path: Path) -> str:
    """Infer group name from the first relative path component."""

    return _legacy._group_from_relative_path(relative_path)


def parse_eval_groups(raw_groups: list[str] | None) -> set[str] | None:
    """Parse ``--eval-groups`` CLI tokens into normalized group names."""

    return _legacy._parse_eval_groups(raw_groups)


def _to_legacy_result(result: AnalysisResult) -> _legacy.PipelineResult:
    legacy_steps = [_legacy.PipelineStep(name=s.name, image=s.image, cmap=s.cmap) for s in result.steps]
    source_path = result.source_path if result.source_path is not None else Path("<memory>")
    return _legacy.PipelineResult(
        source_path=source_path,
        steps=legacy_steps,
        circle_count=int(result.circle_count),
        hough_params=dict(result.hough_params),
        debug_info=dict(result.debug_info),
    )


def build_debug_dump_payload(
    result: AnalysisResult,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> dict[str, Any]:
    """Build the debug JSON payload for one analysis result."""

    return _legacy.build_debug_dump_payload(
        result=_to_legacy_result(result),
        step_index=step_index,
        final_only=final_only,
        panel_text=panel_text,
    )


def export_result_debug(
    result: AnalysisResult,
    export_root: Path,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> tuple[Path, Path]:
    """Export debug payload and human-readable panel snapshot."""

    return _legacy.export_result_debug(
        result=_to_legacy_result(result),
        export_root=export_root,
        step_index=step_index,
        final_only=final_only,
        panel_text=panel_text,
    )


__all__ = [
    "read_bgr_or_raise",
    "letterbox_resize_to_canvas",
    "save_pipeline_figure",
    "to_serializable",
    "format_total_cents",
    "format_signed_cents",
    "group_from_relative_path",
    "parse_eval_groups",
    "build_debug_dump_payload",
    "export_result_debug",
]
