"""Structured debug export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline.detectors import _coin_marker_token
from src.pipeline.models import AnalysisResult


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
