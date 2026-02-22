from dataclasses import dataclass
from pathlib import Path

from .ground_truth import GroundTruthEntry


@dataclass(frozen=True)
class EvaluationItem:
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
        self._skipped_missing_ground_truth += 1

    def add_filtered_group(self) -> None:
        self._skipped_filtered_group += 1

    def _compute_metrics(self, items: list[EvaluationItem]) -> dict[str, float | int | None]:
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
                "value_mae_cents": 0.0,
                "value_mae_eur": 0.0,
                "value_total_abs_error_cents": 0,
                "coin_score": 0.0,
                "value_score": None,
                "general_score": 0.0,
                # Backward-compatible aliases.
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
        value_correct = sum(1 for item in value_items if item.value_is_correct)
        value_total_abs_error_cents = sum(int(item.value_abs_diff_cents or 0) for item in value_items)
        value_accuracy = (value_correct / value_evaluated) * 100.0 if value_evaluated > 0 else 0.0
        value_mae_cents = (value_total_abs_error_cents / value_evaluated) if value_evaluated > 0 else 0.0
        value_mae_eur = value_mae_cents / 100.0

        coin_error_score = 100.0 / (1.0 + coin_mae)
        coin_score = 0.60 * coin_accuracy + 0.40 * coin_error_score

        if value_evaluated > 0:
            value_error_score = 100.0 / (1.0 + value_mae_eur)
            value_score = 0.60 * value_accuracy + 0.40 * value_error_score
            general_score = 0.50 * coin_score + 0.50 * value_score
        else:
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
            "value_mae_cents": value_mae_cents,
            "value_mae_eur": value_mae_eur,
            "value_total_abs_error_cents": value_total_abs_error_cents,
            "coin_score": coin_score,
            "value_score": value_score,
            "general_score": general_score,
            # Backward-compatible aliases.
            "correct": coin_correct,
            "accuracy": coin_accuracy,
            "mae": coin_mae,
            "total_abs_error": coin_total_abs_error,
        }

    def summary(self) -> dict[str, float | int | None]:
        return self._compute_metrics(self._items)

    def summary_by_group(self) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[EvaluationItem]] = {}
        for item in self._items:
            grouped.setdefault(item.group, []).append(item)

        out: dict[str, dict[str, float | int | None]] = {}
        for group, items in grouped.items():
            out[group] = self._compute_metrics(items)
        return out


class Evaluation(Evaluator):
    """Pipeline stage class alias: ImagePreprocessing -> CoinDetector -> CoinValueEstimator -> Evaluation."""
