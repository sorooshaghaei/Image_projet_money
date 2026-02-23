"""I/O, plotting backend, serialization and formatting helpers."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

from src.dataset import normalize_group_name
from src.detectors import _coin_marker_token
from src.models import AnalysisResult


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


def read_bgr_or_raise(image_path: Path) -> np.ndarray:
    """Read image with OpenCV and raise explicit error if loading fails."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path.resolve()}")
    return image_bgr


def letterbox_resize_to_canvas(image_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize with aspect-ratio preservation and pad onto fixed canvas."""
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image has invalid dimensions")

    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


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


def to_serializable(value: Any) -> Any:
    """Convert nested runtime objects into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
        }
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return repr(value)


def build_debug_dump_payload(
    result: AnalysisResult,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> dict[str, Any]:
    """Build structured debug payload for export/viewer snapshots."""
    steps = result.steps[-1:] if final_only else result.steps
    step_count = len(steps)
    clamped_step_index = 0
    step_name = "n/a"
    step_shape = None
    if step_count > 0:
        clamped_step_index = max(0, min(int(step_index), step_count - 1))
        step = steps[clamped_step_index]
        step_name = step.name
        step_shape = [int(dim) for dim in step.image.shape]

    info = result.debug_info if isinstance(result.debug_info, dict) else {}
    relative_path = str(info.get("relative_path", result.source_path.name))
    predictions = info.get("value_predictions", {})
    coin_rows: list[dict[str, Any]] = []
    if isinstance(predictions, dict):
        coin_ids: list[int] = []
        for raw_key in predictions.keys():
            try:
                coin_ids.append(int(raw_key))
            except (TypeError, ValueError):
                continue
        for coin_id in sorted(coin_ids):
            pred = predictions.get(coin_id, {})
            if pred == {} and coin_id in predictions:
                pred = predictions[coin_id]
            elif pred == {} and str(coin_id) in predictions:
                pred = predictions[str(coin_id)]
            if not isinstance(pred, dict):
                continue
            coin_rows.append(
                {
                    "marker": _coin_marker_token(coin_id),
                    "coin_id": int(coin_id),
                    "best_label": str(pred.get("best_label", "?")),
                    "best_denom": str(pred.get("best_denom", "?")),
                    "best_prob": float(pred.get("best_prob", 0.0)),
                    "family": str(pred.get("family", "unknown")),
                }
            )

    payload: dict[str, Any] = {
        "source_path": str(result.source_path),
        "relative_path": relative_path,
        "viewer": {
            "final_only": bool(final_only),
            "step_index": int(clamped_step_index),
            "step_count": int(step_count),
            "step_name": step_name,
            "step_image_shape": step_shape,
        },
        "metrics": {
            "status": str(info.get("status", "n/a")),
            "predicted_coin_count": int(info.get("predicted_coin_count", result.circle_count)),
            "true_coin_count": to_serializable(info.get("true_coin_count")),
            "coin_diff": to_serializable(info.get("coin_diff")),
            "predicted_value_cents": int(info.get("predicted_value_cents", info.get("total_cents", 0))),
            "true_value_cents": to_serializable(info.get("true_value_cents")),
            "value_diff_cents": to_serializable(info.get("value_diff_cents")),
        },
        "hough_params": to_serializable(result.hough_params),
        "coin_predictions": coin_rows,
        "raw_debug_info": to_serializable(info),
    }
    if panel_text is not None:
        payload["panel_text"] = panel_text
    return payload


def export_result_debug(
    result: AnalysisResult,
    export_root: Path,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> tuple[Path, Path]:
    """Export debug payload as `.json` plus human-readable `.txt` snapshot."""
    payload = build_debug_dump_payload(
        result=result,
        step_index=step_index,
        final_only=final_only,
        panel_text=panel_text,
    )

    relative_path = Path(str(payload.get("relative_path", result.source_path.name)))
    out_subdir = Path(export_root) / relative_path.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    step_no = int(payload["viewer"]["step_index"]) + 1
    tag = "final" if final_only else f"s{step_no:02d}"
    json_path = out_subdir / f"{relative_path.stem}_debug_{tag}.json"
    text_path = out_subdir / f"{relative_path.stem}_debug_{tag}.txt"

    json_blob = json.dumps(payload, indent=2, ensure_ascii=True)
    json_path.write_text(json_blob, encoding="utf-8")

    panel_blob = panel_text if panel_text is not None else "(panel text not captured)"
    text_blob = (
        "DEBUG PANEL SNAPSHOT\n"
        "====================\n"
        f"{panel_blob}\n\n"
        "FULL DEBUG PAYLOAD (JSON)\n"
        "=========================\n"
        f"{json_blob}\n"
    )
    text_path.write_text(text_blob, encoding="utf-8")
    return json_path, text_path


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


def group_from_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) <= 1:
        return ""
    return normalize_group_name(parts[0])


def parse_eval_groups(raw_groups: list[str] | None) -> set[str] | None:
    if raw_groups is None:
        return None
    groups: set[str] = set()
    for token in raw_groups:
        for chunk in token.split(","):
            group = normalize_group_name(chunk.strip())
            if group:
                groups.add(group)
    return groups if groups else None


def format_total_cents(total_cents: int) -> str:
    euros = total_cents // 100
    cents = total_cents % 100
    return f"{euros} EUR {cents:02d} c"


def format_signed_cents(diff_cents: int) -> str:
    sign = "+" if diff_cents >= 0 else "-"
    euros_abs = abs(int(diff_cents)) // 100
    cents_abs = abs(int(diff_cents)) % 100
    return f"{sign}{euros_abs} EUR {cents_abs:02d} c"


def fmt_optional_score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"
