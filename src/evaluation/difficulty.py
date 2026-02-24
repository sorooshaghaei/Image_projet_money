"""Difficulty classification and aggregate statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.evaluation.metrics import EvaluationItem


MEDIUM_MAX_COIN_ABS_DIFF_DEFAULT = 1
MEDIUM_MAX_VALUE_ABS_DIFF_CENTS_DEFAULT = 100


def classify_difficulty(
    item: EvaluationItem,
    *,
    medium_max_coin_abs_diff: int = MEDIUM_MAX_COIN_ABS_DIFF_DEFAULT,
    medium_max_value_abs_diff_cents: int = MEDIUM_MAX_VALUE_ABS_DIFF_CENTS_DEFAULT,
) -> str:
    """Classify one evaluation row into easy/medium/hard."""
    coin_threshold = max(0, int(medium_max_coin_abs_diff))
    value_threshold = max(0, int(medium_max_value_abs_diff_cents))
    coin_abs_diff = abs(int(item.diff))
    value_abs_diff_cents = None if item.value_abs_diff_cents is None else int(item.value_abs_diff_cents)

    if coin_abs_diff == 0 and (value_abs_diff_cents is None or value_abs_diff_cents == 0):
        return "easy"
    if coin_abs_diff <= coin_threshold and (value_abs_diff_cents is None or value_abs_diff_cents <= value_threshold):
        return "medium"
    return "hard"


def format_difficulty_label(label: str) -> str:
    """Render a Rich-friendly colored difficulty label."""
    if label == "easy":
        return "[green]EASY[/green]"
    if label == "medium":
        return "[yellow]MEDIUM[/yellow]"
    if label == "hard":
        return "[red]HARD[/red]"
    return "-"


def build_difficulty_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate overall + per-group stats from per-image labels."""
    valid_labels = {"easy", "medium", "hard"}
    labeled = [row for row in rows if str(row.get("difficulty")) in valid_labels]
    total = len(labeled)

    overall_counts: Counter[str] = Counter(str(row["difficulty"]) for row in labeled)
    overall: dict[str, dict[str, float | int]] = {}
    for label in ("easy", "medium", "hard"):
        subset = [row for row in labeled if str(row["difficulty"]) == label]
        count = len(subset)
        coin_abs_values = [float(row["coin_abs_diff"]) for row in subset if row.get("coin_abs_diff") is not None]
        value_abs_values = [
            float(row["value_abs_diff_cents"]) / 100.0 for row in subset if row.get("value_abs_diff_cents") is not None
        ]
        overall[label] = {
            "count": count,
            "share_pct": (100.0 * count / total) if total > 0 else 0.0,
            "avg_coin_abs_diff": (sum(coin_abs_values) / len(coin_abs_values)) if coin_abs_values else 0.0,
            "avg_value_abs_diff_eur": (sum(value_abs_values) / len(value_abs_values)) if value_abs_values else 0.0,
        }

    by_group_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in labeled:
        by_group_counter[str(row.get("group", ""))][str(row["difficulty"])] += 1

    by_group: dict[str, dict[str, float | int]] = {}
    for group, counter in by_group_counter.items():
        group_total = sum(counter.values())
        hard_count = int(counter.get("hard", 0))
        by_group[group] = {
            "easy": int(counter.get("easy", 0)),
            "medium": int(counter.get("medium", 0)),
            "hard": hard_count,
            "total": int(group_total),
            "hard_share_pct": (100.0 * hard_count / group_total) if group_total > 0 else 0.0,
        }

    return {
        "total_labeled": total,
        "overall": overall,
        "by_group": by_group,
        "raw_counts": dict(overall_counts),
    }
