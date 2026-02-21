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

    @property
    def diff(self) -> int:
        return self.predicted - self.expected

    @property
    def abs_diff(self) -> int:
        return abs(self.diff)

    @property
    def is_correct(self) -> bool:
        return self.diff == 0


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
    ) -> EvaluationItem:
        item = EvaluationItem(
            relative_path=relative_path,
            group=group,
            filename=ground_truth.filename,
            predicted=int(predicted),
            expected=int(ground_truth.coin_count),
        )
        self._items.append(item)
        return item

    def add_missing_ground_truth(self) -> None:
        self._skipped_missing_ground_truth += 1

    def add_filtered_group(self) -> None:
        self._skipped_filtered_group += 1

    def summary(self) -> dict[str, float | int]:
        if not self._items:
            return {
                "evaluated": 0,
                "correct": 0,
                "accuracy": 0.0,
                "mae": 0.0,
                "total_abs_error": 0,
            }

        correct = sum(1 for item in self._items if item.is_correct)
        total_abs_error = sum(item.abs_diff for item in self._items)
        evaluated = len(self._items)
        accuracy = (correct / evaluated) * 100.0
        mae = total_abs_error / evaluated
        return {
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": accuracy,
            "mae": mae,
            "total_abs_error": total_abs_error,
        }

    def summary_by_group(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[EvaluationItem]] = {}
        for item in self._items:
            grouped.setdefault(item.group, []).append(item)

        out: dict[str, dict[str, float | int]] = {}
        for group, items in grouped.items():
            correct = sum(1 for item in items if item.is_correct)
            total_abs_error = sum(item.abs_diff for item in items)
            evaluated = len(items)
            accuracy = (correct / evaluated) * 100.0 if evaluated > 0 else 0.0
            mae = (total_abs_error / evaluated) if evaluated > 0 else 0.0
            out[group] = {
                "evaluated": evaluated,
                "correct": correct,
                "accuracy": accuracy,
                "mae": mae,
                "total_abs_error": total_abs_error,
            }
        return out
