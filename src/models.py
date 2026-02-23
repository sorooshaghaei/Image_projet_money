"""Dataclasses for pipeline and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset import GroundTruthEntry


@dataclass(frozen=True)
class EvaluationItem:
    """One evaluated prediction aligned with one ground-truth record."""

    relative_path: Path
    group: str
    filename: str
    predicted: int
    expected: int
    predicted_value_cents: int
    expected_value_cents: int | None

    @property
    def diff(self) -> int:
        return self.predicted - self.expected

    @property
    def abs_diff(self) -> int:
        return abs(self.diff)

    @property
    def is_correct(self) -> bool:
        return self.diff == 0

    @property
    def has_value_ground_truth(self) -> bool:
        return self.expected_value_cents is not None

    @property
    def value_diff_cents(self) -> int | None:
        if self.expected_value_cents is None:
            return None
        return int(self.predicted_value_cents) - int(self.expected_value_cents)

    @property
    def value_abs_diff_cents(self) -> int | None:
        diff = self.value_diff_cents
        return None if diff is None else abs(diff)

    @property
    def value_is_correct(self) -> bool:
        diff = self.value_diff_cents
        return False if diff is None else diff == 0


class Evaluator:
    """Accumulates evaluation items and computes summary metrics."""

    VALUE_TOLERANCE_CENTS = 100
    VALUE_SOFT_CAP_CENTS = 400

    def __init__(self):
        self._items: list[EvaluationItem] = []
        self._skipped_missing_ground_truth = 0
        self._skipped_filtered_group = 0

    @property
    def evaluated_count(self) -> int:
        return len(self._items)

    @property
    def skipped_missing_ground_truth(self) -> int:
        return self._skipped_missing_ground_truth

    @property
    def skipped_filtered_group(self) -> int:
        return self._skipped_filtered_group

    def add_match(
        self,
        relative_path: Path,
        group: str,
        predicted: int,
        ground_truth: GroundTruthEntry,
        predicted_value_cents: int = 0,
    ) -> EvaluationItem:
        """Register one prediction/ground-truth pair and return stored item."""
        item = EvaluationItem(
            relative_path=relative_path,
            group=group,
            filename=ground_truth.filename,
            predicted=int(predicted),
            expected=int(ground_truth.coin_count),
            predicted_value_cents=int(predicted_value_cents),
            expected_value_cents=None if ground_truth.value_cents is None else int(ground_truth.value_cents),
        )
        self._items.append(item)
        return item

    def add_missing_ground_truth(self) -> None:
        """Increment skip counter for samples missing annotations."""
        self._skipped_missing_ground_truth += 1

    def add_filtered_group(self) -> None:
        """Increment skip counter for filtered-out groups."""
        self._skipped_filtered_group += 1

    def _compute_metrics(self, items: list[EvaluationItem]) -> dict[str, float | int | None]:
        """Compute coin/value metrics for a given item subset."""
        value_tolerance_cents = int(self.VALUE_TOLERANCE_CENTS)
        value_soft_cap_cents = int(max(self.VALUE_SOFT_CAP_CENTS, value_tolerance_cents + 1))

        def _value_quality_score(abs_diff_cents: int) -> float:
            diff = float(max(0, int(abs_diff_cents)))
            tol = float(value_tolerance_cents)
            cap = float(value_soft_cap_cents)
            if diff <= tol:
                return 100.0
            if diff >= cap:
                return 0.0
            return float(100.0 * (1.0 - (diff - tol) / (cap - tol)))

        if not items:
            return {
                "evaluated": 0,
                "coin_correct": 0,
                "coin_accuracy": 0.0,
                "coin_mae": 0.0,
                "coin_total_abs_error": 0,
                "value_evaluated": 0,
                "value_correct": 0,
                "value_accuracy": 0.0,
                "value_correct_exact": 0,
                "value_accuracy_exact": 0.0,
                "value_mae_cents": 0.0,
                "value_mae_eur": 0.0,
                "value_total_abs_error_cents": 0,
                "value_tolerance_cents": value_tolerance_cents,
                "value_soft_cap_cents": value_soft_cap_cents,
                "value_error_score": 0.0,
                "coin_score": 0.0,
                "value_score": None,
                "general_score": 0.0,
                "correct": 0,
                "accuracy": 0.0,
                "mae": 0.0,
                "total_abs_error": 0,
            }

        evaluated = len(items)
        coin_correct = sum(1 for item in items if item.is_correct)
        coin_total_abs_error = sum(item.abs_diff for item in items)
        coin_accuracy = (coin_correct / evaluated) * 100.0
        coin_mae = coin_total_abs_error / evaluated

        value_items = [item for item in items if item.has_value_ground_truth]
        value_evaluated = len(value_items)
        value_correct_exact = sum(1 for item in value_items if item.value_is_correct)
        value_correct = sum(1 for item in value_items if int(item.value_abs_diff_cents or 0) <= value_tolerance_cents)
        value_total_abs_error_cents = sum(int(item.value_abs_diff_cents or 0) for item in value_items)
        value_accuracy_exact = (value_correct_exact / value_evaluated) * 100.0 if value_evaluated > 0 else 0.0
        value_accuracy = (value_correct / value_evaluated) * 100.0 if value_evaluated > 0 else 0.0
        value_mae_cents = (value_total_abs_error_cents / value_evaluated) if value_evaluated > 0 else 0.0
        value_mae_eur = value_mae_cents / 100.0

        coin_error_score = 100.0 / (1.0 + coin_mae)
        coin_score = 0.60 * coin_accuracy + 0.40 * coin_error_score

        if value_evaluated > 0:
            quality_scores = [_value_quality_score(int(item.value_abs_diff_cents or 0)) for item in value_items]
            value_error_score = float(np.mean(quality_scores)) if quality_scores else 0.0
            value_score = 0.55 * value_accuracy + 0.45 * value_error_score
            general_score = 0.50 * coin_score + 0.50 * value_score
        else:
            value_error_score = 0.0
            value_score = None
            general_score = coin_score

        return {
            "evaluated": evaluated,
            "coin_correct": coin_correct,
            "coin_accuracy": coin_accuracy,
            "coin_mae": coin_mae,
            "coin_total_abs_error": coin_total_abs_error,
            "value_evaluated": value_evaluated,
            "value_correct": value_correct,
            "value_accuracy": value_accuracy,
            "value_correct_exact": value_correct_exact,
            "value_accuracy_exact": value_accuracy_exact,
            "value_mae_cents": value_mae_cents,
            "value_mae_eur": value_mae_eur,
            "value_total_abs_error_cents": value_total_abs_error_cents,
            "value_tolerance_cents": value_tolerance_cents,
            "value_soft_cap_cents": value_soft_cap_cents,
            "value_error_score": value_error_score,
            "coin_score": coin_score,
            "value_score": value_score,
            "general_score": general_score,
            "correct": coin_correct,
            "accuracy": coin_accuracy,
            "mae": coin_mae,
            "total_abs_error": coin_total_abs_error,
        }

    def summary(self) -> dict[str, float | int | None]:
        """Return global metrics over all evaluated items."""
        return self._compute_metrics(self._items)

    def summary_by_group(self) -> dict[str, dict[str, float | int | None]]:
        """Return metrics grouped by dataset group id."""
        grouped: dict[str, list[EvaluationItem]] = {}
        for item in self._items:
            grouped.setdefault(item.group, []).append(item)

        out: dict[str, dict[str, float | int | None]] = {}
        for group, items in grouped.items():
            out[group] = self._compute_metrics(items)
        return out


class Evaluation(Evaluator):
    """Alias class kept for pipeline stage naming."""


@dataclass
class PipelineStep:
    """One visualization step image in the processing pipeline."""

    name: str
    image: np.ndarray
    cmap: str


@dataclass
class AnalysisResult:
    """Final per-image pipeline result used by runner and viewer."""

    source_path: Path
    steps: list[PipelineStep]
    circle_count: int
    hough_params: dict[str, float | int]
    debug_info: dict[str, Any]


PipelineResult = AnalysisResult
