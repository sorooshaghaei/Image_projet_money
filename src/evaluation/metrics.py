"""Evaluation models and aggregate metric computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.ground_truth import GroundTruthEntry


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
        if not items:
            return {
                "evaluated": 0,
                "coin_correct": 0,
                "coin_accuracy": 0.0,
                "coin_total_abs_error": 0,
                "value_evaluated": 0,
                "value_precision": 0.0,
                "value_recall": 0.0,
                "value_f1": 0.0,
                "value_mae_cents": 0.0,
                "value_mae_eur": 0.0,
                "value_mse_cents2": 0.0,
                "value_mse_eur2": 0.0,
                "value_total_abs_error_cents": 0,
                "value_total_sq_error_cents2": 0.0,
                "value_true_positive_cents": 0,
                "value_false_positive_cents": 0,
                "value_false_negative_cents": 0,
                "coin_score": 0.0,
                "value_score": None,
                "general_score": None,
                "correct": 0,
                "accuracy": 0.0,
                "mae": None,
                "mse": None,
                "total_abs_error": 0,
            }

        evaluated = len(items)
        coin_correct = sum(1 for item in items if item.is_correct)
        coin_total_abs_error = sum(item.abs_diff for item in items)
        coin_accuracy = (coin_correct / evaluated) * 100.0

        value_items = [item for item in items if item.has_value_ground_truth]
        value_evaluated = len(value_items)
        value_true_positive_cents = sum(
            min(max(int(item.predicted_value_cents), 0), max(int(item.expected_value_cents or 0), 0))
            for item in value_items
        )
        value_false_positive_cents = sum(
            max(int(item.predicted_value_cents) - int(item.expected_value_cents or 0), 0) for item in value_items
        )
        value_false_negative_cents = sum(
            max(int(item.expected_value_cents or 0) - int(item.predicted_value_cents), 0) for item in value_items
        )
        value_precision = (
            (value_true_positive_cents / (value_true_positive_cents + value_false_positive_cents)) * 100.0
            if (value_true_positive_cents + value_false_positive_cents) > 0
            else 0.0
        )
        value_recall = (
            (value_true_positive_cents / (value_true_positive_cents + value_false_negative_cents)) * 100.0
            if (value_true_positive_cents + value_false_negative_cents) > 0
            else 0.0
        )
        value_f1 = (
            (2.0 * value_precision * value_recall) / (value_precision + value_recall)
            if (value_precision + value_recall) > 0.0
            else 0.0
        )
        value_total_abs_error_cents = sum(int(item.value_abs_diff_cents or 0) for item in value_items)
        value_total_sq_error_cents2 = sum(int(item.value_diff_cents or 0) ** 2 for item in value_items)
        value_mae_cents = (value_total_abs_error_cents / value_evaluated) if value_evaluated > 0 else 0.0
        value_mse_cents2 = (value_total_sq_error_cents2 / value_evaluated) if value_evaluated > 0 else 0.0
        value_mae_eur = value_mae_cents / 100.0
        value_mse_eur2 = value_mse_cents2 / 10000.0

        coin_score = coin_accuracy
        value_score = None
        general_score = None

        return {
            "evaluated": evaluated,
            "coin_correct": coin_correct,
            "coin_accuracy": coin_accuracy,
            "coin_total_abs_error": coin_total_abs_error,
            "value_evaluated": value_evaluated,
            "value_precision": value_precision,
            "value_recall": value_recall,
            "value_f1": value_f1,
            "value_mae_cents": value_mae_cents,
            "value_mae_eur": value_mae_eur,
            "value_mse_cents2": value_mse_cents2,
            "value_mse_eur2": value_mse_eur2,
            "value_total_abs_error_cents": value_total_abs_error_cents,
            "value_total_sq_error_cents2": value_total_sq_error_cents2,
            "value_true_positive_cents": value_true_positive_cents,
            "value_false_positive_cents": value_false_positive_cents,
            "value_false_negative_cents": value_false_negative_cents,
            "coin_score": coin_score,
            "value_score": value_score,
            "general_score": general_score,
            "correct": coin_correct,
            "accuracy": coin_accuracy,
            "mae": None,
            "mse": None,
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
