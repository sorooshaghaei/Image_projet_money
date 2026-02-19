from dataclasses import dataclass
from typing import List, Optional

import cv2
import pandas as pd

from .config import DetectionConfig, RuntimeConfig
from .dataset import DatasetRepository
from .io_utils import ImagePathResolver
from .models import PipelineResult
from .processor import CoinProcessor
from .visualization import HoughTuningBrowser, PipelineVisualizer


@dataclass
class RunStats:
    """Accumulates detection/value metrics across the full dataset run."""

    processed: int = 0
    exact_count_matches: int = 0
    total_abs_error: int = 0
    total_true_count: int = 0
    total_pred_count: int = 0
    matched_count_sum: int = 0
    total_detected_coins: int = 0
    total_labeled_coins: int = 0
    processed_value: int = 0
    total_abs_value_error: float = 0.0
    total_true_value: float = 0.0
    quality_warned_images: int = 0
    quality_rejected_images: int = 0

    def update(
        self,
        pred_count: int,
        true_count: int,
        labeled_count: int,
        value_diff: Optional[float],
        true_value: Optional[float],
    ):
        diff = pred_count - true_count
        self.processed += 1
        self.total_abs_error += abs(diff)
        self.total_true_count += max(0, int(true_count))
        self.total_pred_count += max(0, int(pred_count))
        self.matched_count_sum += min(max(0, int(true_count)), max(0, int(pred_count)))
        self.total_detected_coins += max(0, int(pred_count))
        self.total_labeled_coins += max(0, min(int(labeled_count), int(pred_count)))
        if diff == 0:
            self.exact_count_matches += 1

        if value_diff is not None and true_value is not None:
            self.processed_value += 1
            self.total_abs_value_error += abs(value_diff)
            self.total_true_value += abs(true_value)

    def register_quality(self, warnings_count: int, rejected: bool):
        if warnings_count > 0:
            self.quality_warned_images += 1
        if rejected:
            self.quality_rejected_images += 1

    def print_summary(self):
        if self.processed <= 0:
            print("\n" + "=" * 90)
            print("Summary")
            print("-" * 90)
            print("[WARN] No images processed.")
            print(f"QGate Warned:     {self.quality_warned_images}")
            print(f"QGate Rejected:   {self.quality_rejected_images}")
            print("=" * 90)
            return

        count_accuracy = (self.exact_count_matches / self.processed) * 100.0
        count_mae = self.total_abs_error / self.processed
        recall = (self.matched_count_sum / self.total_true_count) * 100.0 if self.total_true_count > 0 else 0.0
        precision = (self.matched_count_sum / self.total_pred_count) * 100.0 if self.total_pred_count > 0 else 0.0
        f1 = 0.0
        if recall + precision > 1e-9:
            f1 = (2.0 * recall * precision) / (recall + precision)

        labeled_coverage = (self.total_labeled_coins / self.total_detected_coins) * 100.0 if self.total_detected_coins > 0 else 0.0

        value_accuracy = 0.0
        value_mae = 0.0
        value_rel_error_pct = 0.0
        if self.processed_value > 0:
            value_mae = self.total_abs_value_error / self.processed_value
            if self.total_true_value > 1e-9:
                value_rel_error_pct = (self.total_abs_value_error / self.total_true_value) * 100.0
                value_accuracy = max(0.0, 100.0 - value_rel_error_pct)

        general_accuracy = (recall + value_accuracy) / 2.0 if self.processed_value > 0 else recall

        print("\n" + "=" * 90)
        print("Summary")
        print("-" * 90)
        print(f"Images:           {self.processed}")
        print(f"Count Accuracy:   {count_accuracy:.2f}%")
        print(f"Count MAE:        {count_mae:.2f} coins/image")
        print(f"Recall:           {recall:.2f}%")
        print(f"Precision:        {precision:.2f}%")
        print(f"F1 Score:         {f1:.2f}%")
        print(f"Labeled Coverage: {labeled_coverage:.2f}%")
        print(f"QGate Warned:     {self.quality_warned_images}")
        print(f"QGate Rejected:   {self.quality_rejected_images}")
        if self.processed_value > 0:
            print(f"Value Samples:    {self.processed_value}")
            print(f"Value MAE:        {value_mae:.3f} EUR/image")
            print(f"Value Rel Error:  {value_rel_error_pct:.2f}%")
            print(f"Value Accuracy:   {value_accuracy:.2f}%")
        else:
            print("Value Accuracy:   N/A")
        print(f"General Accuracy: {general_accuracy:.2f}%")
        print("=" * 90)


class PipelineApp:
    """Coordinates dataset loading, processing, reporting, and optional UI browsing."""

    _TABLE_RULE = "-" * 108

    def __init__(self, runtime: RuntimeConfig):
        self._runtime = runtime
        self._detector_cfg = DetectionConfig()
        self._processor = CoinProcessor(self._detector_cfg)
        self._dataset_repo = DatasetRepository()
        self._path_resolver = ImagePathResolver(runtime.IMAGE_DIRECTORY)
        self._visualizer = PipelineVisualizer()

    def run(self):
        df = self._dataset_repo.to_dataframe()
        print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

        stats = RunStats()
        results_all: List[PipelineResult] = []

        self._print_header()

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            filename = row["image"]
            true_count = int(row["pieces"])
            true_value = row["value_eur"]
            group = row["group"]

            image_path = self._path_resolver.resolve(filename, group)
            if not image_path:
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERR ] Unreadable: {filename}")
                continue

            result = self._processor.execute(img, filename)
            if not result:
                continue

            stats.register_quality(
                warnings_count=len(result.quality_warnings),
                rejected=bool(result.quality_rejected),
            )

            true_value_text = f"{float(true_value):.2f}" if pd.notna(true_value) else "-"
            file_col = self._truncate_filename(filename, width=28)

            if result.quality_rejected:
                print(
                    f"{idx:>3}  {file_col:<28} {group:<4} "
                    f"{'-':>4} {true_count:>4} {'-':>+4} {'-':>4} "
                    f"{'-':>7} {true_value_text:>7} {'-':>7} {'QER':<3}"
                )
                if result.quality_warnings:
                    print(f"  qgate: {', '.join(result.quality_warnings)}")
                results_all.append(result)
                continue

            pred = int(result.coin_count)
            diff = pred - true_count
            pred_value = float(result.estimated_value_eur)
            value_diff = None
            true_value_num = None
            if pd.notna(true_value):
                true_value_num = float(true_value)
                value_diff = pred_value - true_value_num

            stats.update(
                pred_count=pred,
                true_count=true_count,
                labeled_count=result.labeled_coin_count,
                value_diff=value_diff,
                true_value=true_value_num,
            )

            status = "OK" if diff == 0 else "ERR"
            value_diff_text = f"{value_diff:+.2f}" if value_diff is not None else "-"
            print(
                f"{idx:>3}  {file_col:<28} {group:<4} "
                f"{pred:>4} {true_count:>4} {diff:>+4} {result.labeled_coin_count:>4} "
                f"{pred_value:>7.2f} {true_value_text:>7} {value_diff_text:>7} {status:<3}"
            )

            if result.quality_warnings:
                print(f"  qgate: {', '.join(result.quality_warnings)}")
            if result.scene_metrics:
                print(
                    f"  scene: profile={result.scene_profile} "
                    f"edge={float(result.scene_metrics.get('edge_density', 0.0)):.3f} "
                    f"lap={float(result.scene_metrics.get('laplacian_var', 0.0)):.1f}"
                )

            if result.coin_labels and (diff != 0 or result.labeled_coin_count < result.coin_count):
                label_hist = self._format_label_hist(result.coin_labels, expected_count=result.coin_count)
                print(f"  labels: {label_hist}")

            results_all.append(result)

            if self._runtime.SAVE_STEPS:
                saved = self._visualizer.save_pipeline_steps(result, self._runtime.OUT_DIR)
                if saved:
                    print(f"[SAVED] {saved}")

        stats.print_summary()

        if self._runtime.BROWSE_TUNE:
            HoughTuningBrowser(self._processor, results_all).show()

    def _print_header(self):
        print("\n" + self._TABLE_RULE)
        print(f"{'#':>3}  {'FILE':<28} {'GRP':<4} {'PRED':>4} {'TRUE':>4} {'DIFF':>4} {'LAB':>4} {'P_EUR':>7} {'T_EUR':>7} {'D_EUR':>7} {'ST':<3}")
        print(self._TABLE_RULE)

    def _truncate_filename(self, filename: str, width: int = 28) -> str:
        if len(filename) <= width:
            return filename
        return f"...{filename[-(width - 3):]}"

    def _format_label_hist(self, labels, expected_count: Optional[int] = None):
        counts = {}
        unknown_count = 0
        for label in labels:
            if label is None:
                unknown_count += 1
                continue
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values()) + unknown_count
        if expected_count is not None:
            assert total == expected_count, f"Label summary mismatch: {total} vs expected {expected_count}"

        if not counts and unknown_count == 0:
            return "none"
        parts = [f"{den}c x{counts[den]}" for den in sorted(counts)]
        if unknown_count > 0:
            parts.append(f"unknown x{unknown_count}")
        return ", ".join(parts)


class AppRunner:
    """Thin wrapper to keep a stable app entrypoint."""

    def main(self):
        PipelineApp(RuntimeConfig()).run()


if __name__ == "__main__":
    AppRunner().main()
