"""Matplotlib backend setup and pipeline figure rendering."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib
import numpy as np

from src.pipeline.models import AnalysisResult


def _module_available(module_name: str) -> bool:
    """Return whether a Python module can be imported."""
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


def _configure_matplotlib_backend() -> str:
    """Select best available backend with safe headless fallback."""
    current_backend = str(matplotlib.get_backend()).strip().lower()
    if current_backend and current_backend not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
        if not current_backend.startswith("module://matplotlib_inline"):
            return str(matplotlib.get_backend()).strip()

    candidates = [
        ("TkAgg", _module_available("tkinter")),
        (
            "QtAgg",
            any(_module_available(module_name) for module_name in ("PyQt6", "PySide6", "PyQt5", "PySide2")),
        ),
    ]
    for backend_name, available in candidates:
        if not available:
            continue
        try:
            matplotlib.use(backend_name, force=True)
            return backend_name
        except Exception:
            continue

    try:
        matplotlib.use("WebAgg", force=True)
        return "WebAgg"
    except Exception:
        pass

    matplotlib.use("Agg", force=True)
    return "Agg"


_MPL_BACKEND = _configure_matplotlib_backend()
import matplotlib.pyplot as plt


def is_non_interactive_backend(backend_name: str) -> bool:
    """Check whether Matplotlib backend cannot open interactive windows."""
    backend = str(backend_name).strip().lower()
    if backend.startswith("module://matplotlib_inline"):
        return True

    non_interactive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    return backend in non_interactive_backends


def save_pipeline_figure(
    result: AnalysisResult,
    output_path: Path,
    cols: int = 3,
    final_only: bool = False,
) -> None:
    """Save full (or final-only) pipeline panel grid as one image file."""
    cols = max(1, int(cols))
    rows = 1 if final_only else max(1, ceil(max(1, len(result.steps)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows))
    axes_matrix = _normalize_axes(axes, rows, cols)
    steps = result.steps[-1:] if final_only else result.steps

    for flat_idx in range(rows * cols):
        row, col = divmod(flat_idx, cols)
        ax = axes_matrix[row, col]
        ax.axis("off")
        if flat_idx >= len(steps):
            continue
        step = steps[flat_idx]
        if step.cmap == "gray":
            ax.imshow(step.image, cmap="gray")
        else:
            ax.imshow(step.image)
        ax.set_title(step.name if not final_only else f"Final: {step.name}", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _normalize_axes(axes, rows: int, cols: int) -> np.ndarray:
    if rows == 1 and cols == 1:
        return np.array([[axes]])
    if rows == 1:
        return np.array([axes])
    if cols == 1:
        return np.array([[ax] for ax in axes])
    return axes
