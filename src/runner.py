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

        general_accuracy = (detection_recall + value_accuracy) / 2.0 if self.processed_value > 0 else detection_recall

        print("=" * 120)
        print(f"Total Images:     {self.processed}")
        print("[Detection]")
        print(f"Count Accuracy:   {count_accuracy:.2f}% (exact match)")
        print(f"Recall:           {detection_recall:.2f}%")
        print(f"Precision:        {detection_precision:.2f}%")
        print(f"F1 Score:         {detection_f1:.2f}%")
        print(f"Count MAE:        {count_mae:.2f} coins/image")
        print("[Classification]")
        print(f"Labeled Coverage: {label_coverage:.2f}%")
        if self.processed_value > 0:
            print(f"Value Samples:    {self.processed_value}")
            print(f"Value MAE:        {value_mae:.3f} EUR/image")
            print(f"Value Rel Error:  {value_rel_error_pct:.2f}%")
            print(f"Value Accuracy:   {value_accuracy:.2f}% (from relative error)")
        else:
            print("Value Accuracy:   N/A (no ground truth values)")
        print(f"General Accuracy: {general_accuracy:.2f}%")
        print("=" * 120)


class PipelineApp:
    """Coordinates dataset loading, processing, reporting, and optional UI browsing."""

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

        for _, row in df.iterrows():
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

            # Compare prediction vs annotation at both count level and value level.
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

            status = "PERFECT" if diff == 0 else "ERROR"
            true_value_text = f"{float(true_value):.2f}" if pd.notna(true_value) else "N/A"
            value_diff_text = f"{value_diff:+.2f}" if value_diff is not None else "N/A"
            print(
                f"{filename:<25} | {group:<5} | {pred:<6} | {true_count:<6} | {diff:<6} | "
                f"{result.labeled_coin_count:<7} | {pred_value:<8.2f} | {true_value_text:<8} | {value_diff_text:<8} | "
                f"{status:<10}"
            )

            if result.coin_labels:
                label_hist = self._format_label_hist(result.coin_labels, expected_count=result.coin_count)
                print(f"  labels: {label_hist}")
            if result.coin_tags and result.coin_radii:
                self._print_coin_properties(
                    coin_tags=result.coin_tags,
                    coin_radii=result.coin_radii,
                    color_labels=result.coin_color_labels,
                    candidate_denoms=result.coin_candidate_denoms,
                    labels=result.coin_labels,
                    fit_errors=result.ratio_fit_errors,
                )
            if result.radius_ratio_matrix:
                self._print_ratio_matrix(
                    result.radius_ratio_matrix,
                    result.coin_tags,
                    result.coin_labels,
                    result.ratio_fit_errors,
                )

            results_all.append(result)

            if self._runtime.SAVE_STEPS:
                saved = self._visualizer.save_pipeline_steps(result, self._runtime.OUT_DIR, cols=4)
                if saved:
                    print(f"[SAVED] {saved}")

        stats.print_summary()

        if self._runtime.BROWSE_TUNE:
            HoughTuningBrowser(self._processor, results_all, cols=4).show()

    def _print_header(self):
        """Prints the table header used by per-image result lines."""
        print("\n" + "=" * 120)
        print(
            f"{'FILENAME':<25} | {'GRP':<5} | {'PRED':<6} | {'TRUE':<6} | {'DIFF':<6} | "
            f"{'LABELED':<7} | {'PRED_EUR':<8} | {'TRUE_EUR':<8} | {'VDIFF':<8} | {'STATUS':<10}"
        )
        print("=" * 120)

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

    def _print_ratio_matrix(
        self,
        ratio_matrix: List[List[float]],
        coin_tags: List[str],
        labels: List[Optional[int]],
        fit_errors: List[Optional[float]],
    ):
        """Pretty-print the inter-coin radius ratio matrix for debugging scale fitting."""
        n = len(ratio_matrix)
        header_tags = [coin_tags[j] if j < len(coin_tags) else f"C{j+1}" for j in range(n)]
        header = "      " + " ".join([f"{tag:>6}" for tag in header_tags])
        print("  radius_ratio_matrix (r_i / r_j):")
        print("   " + header)
        for i in range(n):
            row_txt = " ".join([f"{v:>6.3f}" for v in ratio_matrix[i]])
            label_txt = f"{labels[i]}c" if i < len(labels) and labels[i] is not None else "?"
            err = fit_errors[i] if i < len(fit_errors) else None
            err_txt = f"{err:.3f}" if err is not None else "N/A"
            row_tag = coin_tags[i] if i < len(coin_tags) else f"C{i+1}"
            print(f"   {row_tag:>4} {row_txt} | guess={label_txt:<4} ratio_err={err_txt}")

    def _print_coin_properties(
        self,
        coin_tags: List[str],
        coin_radii: List[float],
        color_labels: List[str],
        candidate_denoms: List[List[int]],
        labels: List[Optional[int]],
        fit_errors: List[Optional[float]],
    ):
        """Print per-coin attributes used by color + geometric classification."""
        if not coin_radii:
            return
        r_ref = float(sorted(coin_radii)[len(coin_radii) // 2])
        if r_ref <= 1e-6:
            r_ref = 1.0

        print("  coin_properties:")
        for i, (tag, radius) in enumerate(zip(coin_tags, coin_radii)):
            norm = radius / r_ref
            color_lbl = color_labels[i] if i < len(color_labels) else "unknown"
            candidates = candidate_denoms[i] if i < len(candidate_denoms) else []
            cand_txt = ",".join([f"{d}c" for d in candidates]) if candidates else "-"
            guess = f"{labels[i]}c" if i < len(labels) and labels[i] is not None else "?"
            err = fit_errors[i] if i < len(fit_errors) else None
            err_txt = f"{err:.3f}" if err is not None else "N/A"
            print(
                f"   {tag}: radius={radius:.2f}px norm={norm:.3f} color={color_lbl} "
                f"cands=[{cand_txt}] guess={guess} ratio_err={err_txt}"
            )


class AppRunner:
    """Thin wrapper to keep a stable app entrypoint."""

    def main(self):
        PipelineApp(RuntimeConfig()).run()


if __name__ == "__main__":
    AppRunner().main()
