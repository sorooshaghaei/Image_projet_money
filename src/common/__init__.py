"""Common shared utilities."""

from .debug_export import build_debug_dump_payload, export_result_debug, to_serializable
from .formatters import (
    fmt_optional_score,
    format_cents_compact,
    format_diff_cents_compact,
    format_signed_cents,
    format_total_cents,
    group_from_relative_path,
    parse_eval_groups,
)
from .image_io import letterbox_resize_to_canvas, read_bgr_or_raise
from .plotting import is_non_interactive_backend, plt, save_pipeline_figure

__all__ = [
    "read_bgr_or_raise",
    "letterbox_resize_to_canvas",
    "is_non_interactive_backend",
    "plt",
    "save_pipeline_figure",
    "to_serializable",
    "build_debug_dump_payload",
    "export_result_debug",
    "group_from_relative_path",
    "parse_eval_groups",
    "format_total_cents",
    "format_signed_cents",
    "fmt_optional_score",
    "format_cents_compact",
    "format_diff_cents_compact",
]
