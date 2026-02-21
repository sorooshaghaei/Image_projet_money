from dataclasses import dataclass
from pathlib import Path
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

    def update(
        self,
        pred_count: int,
        true_count: int,
        labeled_count: int,
        value_diff: Optional[float],
        true_value: Optional[float],
    ):
        """Ingest per-image results and update aggregate counters."""
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

    def print_summary(self):
        """Compute and print detection/classification summary metrics."""
        if self.processed <= 0:
            print("[WARN] No images processed. Check IMAGE_DIRECTORY path.")
            return

        count_accuracy = (self.exact_count_matches / self.processed) * 100.0
        count_mae = self.total_abs_error / self.processed
        detection_recall = (self.matched_count_sum / self.total_true_count) * 100.0 if self.total_true_count > 0 else 0.0
        detection_precision = (
            (self.matched_count_sum / self.total_pred_count) * 100.0 if self.total_pred_count > 0 else 0.0
        )
        detection_f1 = 0.0
        if detection_recall + detection_precision > 1e-9:
            detection_f1 = (2.0 * detection_recall * detection_precision) / (detection_recall + detection_precision)
        label_coverage = (
            (self.total_labeled_coins / self.total_detected_coins) * 100.0 if self.total_detected_coins > 0 else 0.0
        )

        value_accuracy = 0.0
        value_mae = 0.0
        value_rel_error_pct = 0.0
        if self.processed_value > 0:
            value_mae = self.total_abs_value_error / self.processed_value
            if self.total_true_value > 1e-9:
                value_rel_error_pct = (self.total_abs_value_error / self.total_true_value) * 100.0
                value_accuracy = max(0.0, 100.0 - value_rel_error_pct)

        print("\n" + "=" * 90)
        print("Summary")
        print("-" * 90)
        print(f"Images:           {self.processed}")
        print(f"Count Accuracy:   {count_accuracy:.2f}%")
        print(f"Count MAE:        {count_mae:.2f} coins/image")
        print(f"Recall:           {detection_recall:.2f}%")
        print(f"Precision:        {detection_precision:.2f}%")
        print(f"F1 Score:         {detection_f1:.2f}%")
        print(f"Labeled Coverage: {label_coverage:.2f}%")
        if self.processed_value > 0:
            print(f"Value Samples:    {self.processed_value}")
            print(f"Value MAE:        {value_mae:.3f} EUR/image")
            print(f"Value Rel Error:  {value_rel_error_pct:.2f}%")
            print(f"Value Accuracy:   {value_accuracy:.2f}%")
        else:
            print("Value Accuracy:   N/A")
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
        """Run the pipeline on all annotated images and print detailed diagnostics."""
        df = self._dataset_repo.to_dataframe()
        print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

        stats = RunStats()
        results_all: List[PipelineResult] = []

        self._print_header()

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            filename = row["image"]
            true_count = int(row["pieces"])
            true_value = row["value_eur"]
            group_annot = row["group"]
            true_value_num = float(true_value) if pd.notna(true_value) else None

            image_path = self._path_resolver.resolve(filename, group_annot)
            true_value_text = f"{true_value_num:.2f}" if true_value_num is not None else "-"
            file_col = self._truncate_filename(filename, width=28)
            if not image_path:
                print(
                    f"{idx:>3}  {file_col:<28} {group_annot:<4} "
                    f"{'-':>4} {true_count:>4} {'-':>4} {'-':>4} "
                    f"{'-':>7} {true_value_text:>7} {'-':>7} {'MISS':<3}"
                )
                self._print_image_value_trace(
                    image_name=filename,
                    expected_folder=group_annot,
                    src_path=None,
                    pred_value=None,
                    real_value=true_value_num,
                )
                continue

            resolved_path = Path(image_path)
            resolved_filename = resolved_path.name
            resolved_group = resolved_path.parent.name
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERR ] Unreadable: {filename}")
                file_col = self._truncate_filename(resolved_filename, width=28)
                print(
                    f"{idx:>3}  {file_col:<28} {resolved_group:<4} "
                    f"{'-':>4} {true_count:>4} {'-':>4} {'-':>4} "
                    f"{'-':>7} {true_value_text:>7} {'-':>7} {'IOE':<3}"
                )
                self._print_image_value_trace(
                    image_name=resolved_filename,
                    expected_folder=group_annot,
                    src_path=image_path,
                    pred_value=None,
                    real_value=true_value_num,
                )
                continue

            source_label = f"{resolved_group}/{resolved_filename}"
            result = self._processor.execute(img, source_label)
            if not result:
                continue

            # Compare prediction vs annotation at both count level and value level.
            pred = int(result.coin_count)
            diff = pred - true_count
            pred_value = float(result.estimated_value_eur)
            value_diff = None
            if true_value_num is not None:
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
            file_col = self._truncate_filename(resolved_filename, width=28)
            print(
                f"{idx:>3}  {file_col:<28} {resolved_group:<4} "
                f"{pred:>4} {true_count:>4} {diff:>+4} {result.labeled_coin_count:>4} "
                f"{pred_value:>7.2f} {true_value_text:>7} {value_diff_text:>7} {status:<3}"
            )
            self._print_image_value_trace(
                image_name=resolved_filename,
                expected_folder=group_annot,
                src_path=image_path,
                pred_value=pred_value,
                real_value=true_value_num,
            )

            if result.coin_labels and (diff != 0 or result.labeled_coin_count < result.coin_count):
                label_hist = self._format_label_hist(result.coin_labels, expected_count=result.coin_count)
                print(f"  labels: {label_hist}")

            results_all.append(result)

            if self._runtime.SAVE_STEPS:
                saved = self._visualizer.save_pipeline_steps(result, self._runtime.OUT_DIR)
                if saved:
                    print(f"[SAVED] {self._short_path(saved, width=56)}")

        stats.print_summary()

        if self._runtime.BROWSE_TUNE:
            HoughTuningBrowser(self._processor, results_all).show()

    def _print_header(self):
        """Prints the table header used by per-image result lines."""
        print("\n" + self._TABLE_RULE)
        print(f"{'#':>3}  {'FILE':<28} {'GRP':<4} {'PRED':>4} {'TRUE':>4} {'DIFF':>4} {'LAB':>4} {'P_EUR':>7} {'T_EUR':>7} {'D_EUR':>7} {'ST':<3}")
        print(self._TABLE_RULE)

    def _truncate_filename(self, filename: str, width: int = 28) -> str:
        if len(filename) <= width:
            return filename
        return f"...{filename[-(width - 3):]}"

    def _short_path(self, path_text: Optional[str], width: int = 44) -> str:
        if not path_text:
            return "-"

        path_obj = Path(path_text)
        short_text = path_obj.as_posix()
        base_dir = Path(self._runtime.IMAGE_DIRECTORY).resolve()
        try:
            short_text = path_obj.resolve().relative_to(base_dir).as_posix()
        except Exception:
            short_text = path_obj.as_posix()

        if len(short_text) <= width:
            return short_text
        return f"...{short_text[-(width - 3):]}"

    def _format_label_hist(self, labels, expected_count: Optional[int] = None):
        """Convert per-coin labels into a compact denomination histogram string."""
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

    def _print_image_value_trace(
        self,
        *,
        image_name: str,
        expected_folder: str,
        src_path: Optional[str],
        pred_value: Optional[float],
        real_value: Optional[float],
    ):
        pred_text = f"{float(pred_value):.2f}" if pred_value is not None else "-"
        real_text = f"{float(real_value):.2f}" if real_value is not None else "-"
        folder_text = expected_folder
        if src_path is not None:
            resolved_folder = Path(src_path).parent.name
            folder_text = resolved_folder if resolved_folder == expected_folder else f"{resolved_folder} (ann:{expected_folder})"
        src_text = self._short_path(src_path, width=44)
        print(
            f"      image={image_name} | folder={folder_text} | src={src_text} | "
            f"value={pred_text} | real_value={real_text}"
        )


class AppRunner:
    """Thin wrapper to keep a stable app entrypoint."""

    def main(self):
        PipelineApp(RuntimeConfig()).run()


if __name__ == "__main__":
    AppRunner().main()
