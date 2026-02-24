"""Reporting helpers for evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DIFFICULTY_REPORT_FIELDS = [
    "file",
    "group",
    "status",
    "difficulty",
    "coin_pred",
    "coin_true",
    "coin_diff",
    "coin_abs_diff",
    "value_pred_cents",
    "value_true_cents",
    "value_diff_cents",
    "value_abs_diff_cents",
]


def write_difficulty_report(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-image difficulty report to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIFFICULTY_REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in DIFFICULTY_REPORT_FIELDS})


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write pretty JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
