from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

from src.io_utils import ImagePathResolver

from .analyzer import HybridCoinAnalyzer
from .config import RuntimeConfig
from .dataset import DatasetRepository
from .io_utils import ensure_parent_dir, list_image_paths, short_path
from .models import AnalysisResult


@dataclass
class EvalStats:
    """Aggregate metrics for dataset evaluation."""

    processed: int = 0
    missing_files: int = 0
    unreadable_files: int = 0
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
        *,
        pred_count: int,
        true_count: int,
        labeled_count: int,
        value_diff: Optional[float],
        true_value: Optional[float],
    ) -> None:
        """Accumulate one image result into dataset-level totals."""
        diff = int(pred_count) - int(true_count)
        self.processed += 1
        self.total_abs_error += abs(diff)
        self.total_true_count += max(0, int(true_count))
        self.total_pred_count += max(0, int(pred_count))
        # Coin-wise overlap between pred/true counts used for recall/precision proxy.
        self.matched_count_sum += min(max(0, int(true_count)), max(0, int(pred_count)))
        self.total_detected_coins += max(0, int(pred_count))
        self.total_labeled_coins += max(0, min(int(labeled_count), int(pred_count)))
        if diff == 0:
            self.exact_count_matches += 1

        if value_diff is not None and true_value is not None:
            self.processed_value += 1
            self.total_abs_value_error += abs(float(value_diff))
            self.total_true_value += abs(float(true_value))

    def register_missing(self) -> None:
        self.missing_files += 1

    def register_unreadable(self) -> None:
        self.unreadable_files += 1

    def print_summary(self) -> None:
        """Print terminal summary with count + value quality metrics."""
        if self.processed <= 0:
            print("\n" + "=" * 98)
            print("Summary")
            print("-" * 98)
            print("[WARN] No evaluable images processed.")
            print(f"Missing Files:    {self.missing_files}")
            print(f"Unreadable Files: {self.unreadable_files}")
            print("Count Accuracy:   N/A")
            print("Count MAE:        N/A")
            print("Recall:           0.00%")
            print("Precision:        0.00%")
            print("F1 Score:         0.00%")
            print("Labeled Coverage: N/A")
            print("Value Samples:    0")
            print("Value MAE:        N/A")
            print("Value Rel Error:  N/A")
            print("Value Accuracy:   N/A")
            print("=" * 98)
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

        print("\n" + "=" * 98)
        print("Summary")
        print("-" * 98)
        print(f"Images:           {self.processed}")
        print(f"Missing Files:    {self.missing_files}")
        print(f"Unreadable Files: {self.unreadable_files}")
        print(f"Count Accuracy:   {count_accuracy:.2f}%")
        print(f"Count MAE:        {count_mae:.2f} coins/image")
        print(f"Recall:           {recall:.2f}%")
        print(f"Precision:        {precision:.2f}%")
        print(f"F1 Score:         {f1:.2f}%")
        print(f"Labeled Coverage: {labeled_coverage:.2f}%")
        if self.processed_value > 0:
            print(f"Value Samples:    {self.processed_value}")
            print(f"Value MAE:        {value_mae:.3f} EUR/image")
            print(f"Value Rel Error:  {value_rel_error_pct:.2f}%")
            print(f"Value Accuracy:   {value_accuracy:.2f}%")
        else:
            print("Value Accuracy:   N/A")
        print("=" * 98)


class ExperimentRunner:
    """CLI pipeline for policy tracing and dataset-ground-truth evaluation."""

    _SCAN_TABLE_RULE = "-" * 170
    _DATASET_TABLE_RULE = "-" * 160

    def __init__(self, runtime: RuntimeConfig):
        self._runtime = runtime
        self._analyzer = HybridCoinAnalyzer()
        self._dataset_repo = DatasetRepository()
        self._path_resolver = ImagePathResolver(runtime.image_directory)
        self._dataset_truth_index = self._build_truth_index(self._dataset_repo.to_dataframe())

    def run_scan(
        self,
        *,
        path: str,
        short_root: str,
        mode: str,
        max_images: int,
        csv_path: str,
        visualize: bool,
    ) -> List[AnalysisResult]:
        """Scan arbitrary image path(s) and print per-image method selection + errors."""
        image_paths = list_image_paths(path, self._runtime.valid_extensions)
        if max_images > 0:
            image_paths = image_paths[:max_images]

        if not image_paths:
            print(f"[WARN] no images found under: {path}")
            return []

        short_root_path = Path(short_root)
        print(f"[INFO] mode={mode} | path={Path(path).as_posix()} | images={len(image_paths)}")
        if visualize:
            print("[INFO] visualizer will open after scan processing finishes.")
        self._print_scan_header()

        results: List[AnalysisResult] = []
        scan_rows: List[Dict[str, object]] = []
        for idx, img_path in enumerate(image_paths, start=1):
            group_name = img_path.parent.name
            file_name = img_path.name
            image = cv2.imread(str(img_path))
            if image is None:
                print(
                    f"{idx:>3}  {self._truncate(img_path.as_posix(), 34):<34} {'-':<10} {'-':<16} "
                    f"{'-':>4} {'-':>4} {'-':>4} {'-':>4} {'-':>7} {'-':>7} {'-':>7} {'IOERR':<8}"
                )
                scan_rows.append(
                    {
                        "source_path": img_path.as_posix(),
                        "short_path": img_path.as_posix(),
                        "group": group_name,
                        "image": file_name,
                        "background_label": "",
                        "selected_method": "",
                        "pred_count": "",
                        "true_count": "",
                        "count_diff": "",
                        "labeled_coin_count": "",
                        "pred_value_eur": "",
                        "true_value_eur": "",
                        "value_diff_eur": "",
                        "status": "IOERR",
                        "likely_overlap": "",
                    }
                )
                continue

            rel = short_path(img_path, short_root_path)
            result = self._analyzer.analyze(
                image,
                source_path=img_path.as_posix(),
                short_path=rel,
                mode=mode,
            )
            results.append(result)

            m = result.metrics
            pred_count = int(len(result.circles))
            true_count, true_value = self._lookup_truth(group_name, file_name)
            diff = pred_count - true_count if true_count is not None else None
            pred_value = float(result.estimated_value_eur)
            value_diff = (pred_value - true_value) if true_value is not None else None

            true_count_text = f"{true_count:>4}" if true_count is not None else f"{'-':>4}"
            diff_text = f"{diff:+4d}" if diff is not None else f"{'-':>4}"
            true_value_text = f"{true_value:>7.2f}" if true_value is not None else f"{'-':>7}"
            value_diff_text = f"{value_diff:+7.2f}" if value_diff is not None else f"{'-':>7}"
            status = "OK" if diff == 0 else ("ERR" if diff is not None else "NO_GT")
            print(
                f"{idx:>3}  {self._truncate(rel, 34):<34} "
                f"{m.background_label:<10} "
                f"{result.selected_method:<16} "
                f"{pred_count:>4} "
                f"{true_count_text} "
                f"{diff_text} "
                f"{result.labeled_coin_count:>5} "
                f"{pred_value:>7.2f} "
                f"{true_value_text} "
                f"{value_diff_text} "
                f"{status:<8}"
            )
            print(
                f"      method={result.selected_method} | overlap={'YES' if m.likely_overlap else 'NO'} "
                f"| bg={m.background_label}"
            )
            scan_rows.append(
                {
                    "source_path": result.source_path,
                    "short_path": result.short_path,
                    "group": group_name,
                    "image": file_name,
                    "background_label": result.metrics.background_label,
                    "selected_method": result.selected_method,
                    "pred_count": pred_count,
                    "true_count": true_count if true_count is not None else "",
                    "count_diff": diff if diff is not None else "",
                    "labeled_coin_count": result.labeled_coin_count,
                    "pred_value_eur": f"{pred_value:.6f}",
                    "true_value_eur": f"{true_value:.6f}" if true_value is not None else "",
                    "value_diff_eur": f"{value_diff:.6f}" if value_diff is not None else "",
                    "status": status,
                    "likely_overlap": int(m.likely_overlap),
                }
            )

        # Summary mirrors dataset metrics when ground truth can be matched by filename/group.
        self._print_scan_summary(results, scan_rows)

        if csv_path:
            ensure_parent_dir(csv_path)
            self._write_scan_csv(csv_path, scan_rows)
            print(f"[INFO] trace csv written to {Path(csv_path).as_posix()}")

        if visualize and results:
            print(f"[INFO] opening visualizer for {len(results)} scanned images...")
            try:
                from .visualizer import HybridVisualizer

                HybridVisualizer(
                    analyzer=self._analyzer,
                    image_paths=[Path(r.source_path) for r in results],
                    short_root=short_root_path,
                    start_mode=mode,
                ).show()
            except Exception as exc:  # pragma: no cover - backend/display dependent
                print(f"[WARN] visualizer failed to open: {exc}")
                print("[INFO] continue without UI using: --no-visualize")
        elif visualize and not results:
            print("[WARN] visualizer skipped: no readable scan results.")

        return results

    def run_dataset_eval(
        self,
        *,
        mode: str,
        dataset_limit: int,
        csv_path: str,
        visualize: bool,
        auto_calibrate_value: bool,
    ) -> List[AnalysisResult]:
        """Evaluate against dataset annotations (count/value ground truth)."""
        df = self._dataset_repo.to_dataframe()
        if dataset_limit > 0:
            df = df.head(dataset_limit)

        print(f"[INFO] Loaded {len(df)} annotation rows from dataset table.")
        print(f"[INFO] Dataset evaluation mode={mode} | image_root={self._runtime.image_directory}")
        if auto_calibrate_value:
            self._auto_fit_value_calibration(df=df, mode=mode)
        if visualize:
            print("[INFO] visualizer will open after dataset evaluation finishes.")
        self._print_dataset_header()

        stats = EvalStats()
        rows_for_csv: List[Dict[str, object]] = []
        results_all: List[AnalysisResult] = []

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            filename = str(row["image"])
            true_count = int(row["pieces"])
            true_value_raw = row["value_eur"]
            true_value_num = float(true_value_raw) if pd.notna(true_value_raw) else None
            group = str(row["group"])

            image_path = self._path_resolver.resolve(filename, group)
            true_value_text = f"{true_value_num:.2f}" if true_value_num is not None else "-"

            if not image_path:
                # Annotation exists but no matching image on disk.
                stats.register_missing()
                self._print_dataset_row(
                    idx=idx,
                    filename=filename,
                    group=group,
                    background_label="-",
                    method="-",
                    pred_count=None,
                    true_count=true_count,
                    diff=None,
                    labeled_count=None,
                    pred_value=None,
                    true_value_text=true_value_text,
                    value_diff=None,
                    status="MISS",
                )
                rows_for_csv.append(
                    {
                        "source_path": "",
                        "short_path": f"{group}/{filename}",
                        "image": filename,
                        "group": group,
                        "background_label": "",
                        "selected_method": "",
                        "pred_count": "",
                        "true_count": true_count,
                        "count_diff": "",
                        "labeled_coin_count": "",
                        "pred_value_eur": "",
                        "true_value_eur": true_value_num if true_value_num is not None else "",
                        "value_diff_eur": "",
                        "status": "MISS",
                        "border_cv": "",
                        "edge_density": "",
                        "texture_score": "",
                        "contour_merge_score": "",
                        "hough_overlap_pairs": "",
                        "likely_overlap": "",
                    }
                )
                continue

            image = cv2.imread(image_path)
            resolved_group = Path(image_path).parent.name
            short_resolved = short_path(Path(image_path), Path(self._runtime.image_directory))
            if image is None:
                # Path resolved but image decoding failed.
                stats.register_unreadable()
                self._print_dataset_row(
                    idx=idx,
                    filename=filename,
                    group=resolved_group,
                    background_label="-",
                    method="-",
                    pred_count=None,
                    true_count=true_count,
                    diff=None,
                    labeled_count=None,
                    pred_value=None,
                    true_value_text=true_value_text,
                    value_diff=None,
                    status="IOE",
                )
                rows_for_csv.append(
                    {
                        "source_path": image_path,
                        "short_path": short_resolved,
                        "image": filename,
                        "group": resolved_group,
                        "background_label": "",
                        "selected_method": "",
                        "pred_count": "",
                        "true_count": true_count,
                        "count_diff": "",
                        "labeled_coin_count": "",
                        "pred_value_eur": "",
                        "true_value_eur": true_value_num if true_value_num is not None else "",
                        "value_diff_eur": "",
                        "status": "IOE",
                        "border_cv": "",
                        "edge_density": "",
                        "texture_score": "",
                        "contour_merge_score": "",
                        "hough_overlap_pairs": "",
                        "likely_overlap": "",
                    }
                )
                continue

            result = self._analyzer.analyze(
                image,
                source_path=image_path,
                short_path=short_resolved,
                mode=mode,
            )
            results_all.append(result)

            pred_count = int(len(result.circles))
            diff = pred_count - true_count
            pred_value = float(result.estimated_value_eur)
            value_diff = (pred_value - true_value_num) if true_value_num is not None else None

            stats.update(
                pred_count=pred_count,
                true_count=true_count,
                labeled_count=result.labeled_coin_count,
                value_diff=value_diff,
                true_value=true_value_num,
            )

            status = "OK" if diff == 0 else "ERR"
            self._print_dataset_row(
                idx=idx,
                filename=filename,
                group=resolved_group,
                background_label=result.metrics.background_label,
                method=result.selected_method,
                pred_count=pred_count,
                true_count=true_count,
                diff=diff,
                labeled_count=result.labeled_coin_count,
                pred_value=pred_value,
                true_value_text=true_value_text,
                value_diff=value_diff,
                status=status,
            )

            if result.coin_labels and (diff != 0 or result.labeled_coin_count < pred_count):
                # Print denomination breakdown only when result needs inspection.
                print(f"      labels: {self._format_label_hist(result.coin_labels)}")

            rows_for_csv.append(
                {
                    "source_path": result.source_path,
                    "short_path": result.short_path,
                    "image": filename,
                    "group": resolved_group,
                    "background_label": result.metrics.background_label,
                    "selected_method": result.selected_method,
                    "pred_count": pred_count,
                    "true_count": true_count,
                    "count_diff": diff,
                    "labeled_coin_count": result.labeled_coin_count,
                    "pred_value_eur": f"{pred_value:.6f}",
                    "true_value_eur": f"{true_value_num:.6f}" if true_value_num is not None else "",
                    "value_diff_eur": f"{value_diff:.6f}" if value_diff is not None else "",
                    "status": status,
                    "border_cv": f"{result.metrics.border_cv:.6f}",
                    "edge_density": f"{result.metrics.edge_density:.6f}",
                    "texture_score": f"{result.metrics.texture_score:.6f}",
                    "contour_merge_score": f"{result.metrics.contour_merge_score:.6f}",
                    "hough_overlap_pairs": int(result.metrics.hough_overlap_pairs),
                    "likely_overlap": int(result.metrics.likely_overlap),
                }
            )

        stats.print_summary()

        if csv_path:
            ensure_parent_dir(csv_path)
            self._write_dataset_csv(csv_path, rows_for_csv)
            print(f"[INFO] dataset eval csv written to {Path(csv_path).as_posix()}")

        if visualize and results_all:
            print(f"[INFO] opening visualizer for {len(results_all)} evaluated images...")
            try:
                from .visualizer import HybridVisualizer

                HybridVisualizer(
                    analyzer=self._analyzer,
                    image_paths=[Path(r.source_path) for r in results_all],
                    short_root=Path(self._runtime.image_directory),
                    start_mode=mode,
                ).show()
            except Exception as exc:  # pragma: no cover - backend/display dependent
                print(f"[WARN] visualizer failed to open: {exc}")
                print("[INFO] continue without UI using: --no-visualize")
        elif visualize and not results_all:
            print("[WARN] visualizer skipped: no evaluable dataset images.")

        return results_all

    def _auto_fit_value_calibration(self, *, df: pd.DataFrame, mode: str) -> None:
        """Fit value calibration coefficients from dataset truth using least squares."""
        if df.empty:
            return

        print("[INFO] fitting value calibration coefficients from dataset...")
        base_policy = self._analyzer.policy
        self._analyzer.policy = replace(base_policy, value_calibration_enabled=False)

        samples: List[tuple[float, float, float, float]] = []
        considered = 0
        try:
            for _, row in df.iterrows():
                true_value_raw = row["value_eur"]
                if pd.isna(true_value_raw):
                    continue

                filename = str(row["image"])
                group = str(row["group"])
                image_path = self._path_resolver.resolve(filename, group)
                if not image_path:
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    continue

                short_resolved = short_path(Path(image_path), Path(self._runtime.image_directory))
                result = self._analyzer.analyze(
                    image,
                    source_path=image_path,
                    short_path=short_resolved,
                    mode=mode,
                )
                considered += 1
                raw_value = float(result.estimated_value_eur)
                pred_count_int = int(len(result.circles))
                pred_count = float(pred_count_int)
                labeled_ratio = float(result.labeled_coin_count / max(1, pred_count_int))
                true_value = float(true_value_raw)
                true_count = int(row["pieces"])

                # Fit calibration only on reliable scenes to avoid learning count/detection noise.
                if abs(pred_count_int - true_count) > 2:
                    continue
                if labeled_ratio < 0.65:
                    continue
                samples.append((raw_value, pred_count, labeled_ratio, true_value))
        finally:
            self._analyzer.policy = base_policy

        if len(samples) < 12:
            print(
                f"[WARN] calibration skipped: need >=12 reliable samples,"
                f" got {len(samples)} (considered={considered})"
            )
            return

        x_raw = np.asarray([s[0] for s in samples], dtype=np.float64)
        x_cnt = np.asarray([s[1] for s in samples], dtype=np.float64)
        x_bias = np.ones_like(x_raw, dtype=np.float64)
        y = np.asarray([s[3] for s in samples], dtype=np.float64)
        labeled_ratio = np.asarray([s[2] for s in samples], dtype=np.float64)

        x = np.column_stack([x_raw, x_cnt, x_bias])
        weights = np.clip(0.35 + 0.65 * labeled_ratio, 0.10, 1.00)
        sqrt_w = np.sqrt(weights)
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w

        coeff, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        alpha = float(np.clip(coeff[0], 0.0, 2.0))
        count_beta = float(np.clip(coeff[1], -0.35, 0.35))
        bias = float(np.clip(coeff[2], -3.0, 4.0))

        pred_before = x_raw
        pred_after = alpha * x_raw + count_beta * x_cnt + bias
        mae_before = float(np.mean(np.abs(pred_before - y)))
        mae_after = float(np.mean(np.abs(pred_after - y)))

        self._analyzer.policy = replace(
            base_policy,
            value_calibration_enabled=True,
            value_calibration_alpha=alpha,
            value_calibration_count_beta=count_beta,
            value_calibration_bias=bias,
        )
        print(
            "[INFO] calibration fitted:"
            f" alpha={alpha:.5f} beta={count_beta:.5f} bias={bias:.5f}"
            f" | mae_before={mae_before:.4f} mae_after={mae_after:.4f}"
            f" | samples={len(samples)} reliable/{considered} considered"
        )

    def _print_scan_header(self) -> None:
        print(self._SCAN_TABLE_RULE)
        print(
            f"{'#':>3}  {'IMAGE':<34} {'BG':<10} {'METHOD':<16} {'PRED':>4} {'TRUE':>4} {'DIFF':>4} "
            f"{'LAB':>4} {'P_EUR':>7} {'T_EUR':>7} {'D_EUR':>7} {'ST':<8}"
        )
        print(self._SCAN_TABLE_RULE)

    def _print_scan_summary(self, results: Sequence[AnalysisResult], scan_rows: Sequence[Dict[str, object]]) -> None:
        """Summarize scan results, including metrics when truth rows are available."""
        if not results:
            return

        total = len(results)
        easy = sum(1 for r in results if r.metrics.background_label == "easy")
        medium = sum(1 for r in results if r.metrics.background_label == "medium")
        difficult = sum(1 for r in results if r.metrics.background_label == "difficult")

        by_method = {
            "contours": sum(1 for r in results if r.selected_method == "contours"),
            "hough": sum(1 for r in results if r.selected_method == "hough"),
            "watershed": sum(1 for r in results if r.selected_method == "watershed"),
            "hough+watershed": sum(1 for r in results if r.selected_method == "hough+watershed"),
        }

        avg_value = float(sum(r.estimated_value_eur for r in results) / max(1, total))
        rows_with_gt = [row for row in scan_rows if row.get("true_count", "") != ""]

        print(self._SCAN_TABLE_RULE)
        print(f"[SUMMARY] total={total} | easy={easy} medium={medium} difficult={difficult} | avg_value={avg_value:.2f} EUR")
        print(
            "[SUMMARY] methods="
            f"contours:{by_method['contours']} "
            f"hough:{by_method['hough']} "
            f"watershed:{by_method['watershed']} "
            f"hough+watershed:{by_method['hough+watershed']}"
        )

        if not rows_with_gt:
            print("[SUMMARY] gt_matched=0 | no ground-truth rows matched this scan set")
            print("[SUMMARY] count_acc=N/A | count_mae=N/A | recall=0.00% | precision=0.00% | f1=0.00%")
            print("[SUMMARY] labeled_coverage=N/A | value_mae=N/A | value_rel_err=N/A")
            return

        processed = len(rows_with_gt)
        total_abs_error = 0
        exact_count_matches = 0
        total_true_count = 0
        total_pred_count = 0
        matched_count_sum = 0
        total_detected_coins = 0
        total_labeled_coins = 0
        total_abs_value_error = 0.0
        total_true_value = 0.0
        value_samples = 0

        for row in rows_with_gt:
            pred = int(row["pred_count"])
            true = int(row["true_count"])
            diff = pred - true
            total_abs_error += abs(diff)
            total_true_count += max(0, true)
            total_pred_count += max(0, pred)
            matched_count_sum += min(max(0, true), max(0, pred))
            total_detected_coins += max(0, pred)
            total_labeled_coins += max(0, min(int(row["labeled_coin_count"]), pred))
            if diff == 0:
                exact_count_matches += 1

            if str(row.get("value_diff_eur", "")) != "" and str(row.get("true_value_eur", "")) != "":
                total_abs_value_error += abs(float(row["value_diff_eur"]))
                total_true_value += abs(float(row["true_value_eur"]))
                value_samples += 1

        count_accuracy = (exact_count_matches / processed) * 100.0
        count_mae = total_abs_error / processed
        recall = (matched_count_sum / total_true_count) * 100.0 if total_true_count > 0 else 0.0
        precision = (matched_count_sum / total_pred_count) * 100.0 if total_pred_count > 0 else 0.0
        f1 = (2.0 * recall * precision) / (recall + precision) if (recall + precision) > 1e-9 else 0.0
        labeled_coverage = (total_labeled_coins / total_detected_coins) * 100.0 if total_detected_coins > 0 else 0.0

        value_mae = 0.0
        value_rel_error_pct = 0.0
        if value_samples > 0:
            value_mae = total_abs_value_error / value_samples
            if total_true_value > 1e-9:
                value_rel_error_pct = (total_abs_value_error / total_true_value) * 100.0

        print(
            f"[SUMMARY] gt_matched={processed} | count_acc={count_accuracy:.2f}% | count_mae={count_mae:.2f} "
            f"| recall={recall:.2f}% | precision={precision:.2f}% | f1={f1:.2f}%"
        )
        print(
            f"[SUMMARY] labeled_coverage={labeled_coverage:.2f}% | value_mae={value_mae:.3f} EUR | "
            f"value_rel_err={value_rel_error_pct:.2f}%"
        )

    def _print_dataset_header(self) -> None:
        print(self._DATASET_TABLE_RULE)
        print(
            f"{'#':>3}  {'FILE':<28} {'GRP':<4} {'BG':<10} {'METHOD':<16} {'PRED':>4} {'TRUE':>4} {'DIFF':>4} "
            f"{'LAB':>4} {'P_EUR':>7} {'T_EUR':>7} {'D_EUR':>7} {'ST':<4}"
        )
        print(self._DATASET_TABLE_RULE)

    def _print_dataset_row(
        self,
        *,
        idx: int,
        filename: str,
        group: str,
        background_label: str,
        method: str,
        pred_count: Optional[int],
        true_count: int,
        diff: Optional[int],
        labeled_count: Optional[int],
        pred_value: Optional[float],
        true_value_text: str,
        value_diff: Optional[float],
        status: str,
    ) -> None:
        pred_text = f"{pred_count:>4}" if pred_count is not None else f"{'-':>4}"
        diff_text = f"{diff:+4d}" if diff is not None else f"{'-':>4}"
        lab_text = f"{labeled_count:>4}" if labeled_count is not None else f"{'-':>4}"
        p_eur_text = f"{pred_value:>7.2f}" if pred_value is not None else f"{'-':>7}"
        d_eur_text = f"{value_diff:+7.2f}" if value_diff is not None else f"{'-':>7}"

        print(
            f"{idx:>3}  {self._truncate(filename, 28):<28} {group:<4} {background_label:<10} {method:<16} "
            f"{pred_text} {true_count:>4} {diff_text} {lab_text} {p_eur_text} {true_value_text:>7} {d_eur_text} {status:<4}"
        )

    def _write_scan_csv(self, csv_path: str, rows: Sequence[Dict[str, object]]) -> None:
        headers = [
            "source_path",
            "short_path",
            "group",
            "image",
            "background_label",
            "selected_method",
            "pred_count",
            "true_count",
            "count_diff",
            "labeled_coin_count",
            "pred_value_eur",
            "true_value_eur",
            "value_diff_eur",
            "status",
            "likely_overlap",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_dataset_csv(self, csv_path: str, rows: Sequence[Dict[str, object]]) -> None:
        headers = [
            "source_path",
            "short_path",
            "image",
            "group",
            "background_label",
            "selected_method",
            "pred_count",
            "true_count",
            "count_diff",
            "labeled_coin_count",
            "pred_value_eur",
            "true_value_eur",
            "value_diff_eur",
            "status",
            "border_cv",
            "edge_density",
            "texture_score",
            "contour_merge_score",
            "hough_overlap_pairs",
            "likely_overlap",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _format_label_hist(labels: Sequence[Optional[int]]) -> str:
        counts: Dict[int, int] = {}
        unknown_count = 0
        for label in labels:
            if label is None:
                unknown_count += 1
            else:
                den = int(label)
                counts[den] = counts.get(den, 0) + 1

        parts = [f"{den}c x{counts[den]}" for den in sorted(counts)]
        if unknown_count > 0:
            parts.append(f"unknown x{unknown_count}")
        return ", ".join(parts) if parts else "none"

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return f"...{text[-(width - 3):]}"

    @staticmethod
    def _normalize_group(group: str) -> str:
        text = str(group).strip().lower()
        if text.startswith("grp"):
            return "gp" + text[3:]
        return text

    def _build_truth_index(self, df: pd.DataFrame) -> Dict[tuple[str, str], tuple[int, Optional[float]]]:
        """Build (group, filename) -> (count, value) map for fast lookups during scan mode."""
        index: Dict[tuple[str, str], tuple[int, Optional[float]]] = {}
        for _, row in df.iterrows():
            filename = str(row["image"]).strip()
            group = self._normalize_group(str(row["group"]))
            true_count = int(row["pieces"])
            true_value = float(row["value_eur"]) if pd.notna(row["value_eur"]) else None
            index[(group, filename)] = (true_count, true_value)
        return index

    def _lookup_truth(self, group: str, filename: str) -> tuple[Optional[int], Optional[float]]:
        key = (self._normalize_group(group), str(filename).strip())
        return self._dataset_truth_index.get(key, (None, None))


class AppRunner:
    """Thin app entrypoint used by `main_v2.py`."""

    def main(self) -> None:
        if "MPLCONFIGDIR" not in os.environ:
            # Force matplotlib cache inside writable temp dir to avoid permission issues.
            mpl_cache = Path(tempfile.gettempdir()) / "image_projet_money_mplcache"
            mpl_cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = mpl_cache.as_posix()

        runtime = RuntimeConfig()
        args = _build_parser(runtime).parse_args()
        # Default behavior is dataset evaluation unless --scan is explicitly requested.
        run_dataset_eval = bool(args.evaluate_dataset or not args.scan)

        csv_path = args.csv
        if not csv_path:
            csv_path = runtime.dataset_eval_csv_path if run_dataset_eval else runtime.report_csv_path
        elif not Path(csv_path).is_absolute():
            # Resolve relative csv paths against project root for predictable output location.
            project_root = Path(runtime.image_directory).resolve().parents[1]
            csv_path = (project_root / csv_path).resolve().as_posix()

        runner = ExperimentRunner(runtime)
        if run_dataset_eval:
            runner.run_dataset_eval(
                mode=args.mode,
                dataset_limit=args.dataset_limit,
                csv_path=csv_path,
                visualize=args.visualize,
                auto_calibrate_value=args.auto_calibrate_value,
            )
        else:
            runner.run_scan(
                path=args.path,
                short_root=args.short_root,
                mode=args.mode,
                max_images=args.max_images,
                csv_path=csv_path,
                visualize=args.visualize,
            )


def _build_parser(runtime: RuntimeConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid coin analysis: background label (easy/medium/difficult), method routing "
            "(contours/hough/watershed/hough+watershed), and optional dataset evaluation metrics."
        )
    )
    parser.add_argument("--path", default=runtime.image_directory, help="image file or directory for scan mode")
    parser.add_argument(
        "--short-root",
        default=runtime.image_directory,
        help="root used to print short relative image paths",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "fast", "contours", "hough", "watershed", "hybrid", "hough+watershed"],
        help="force one method or keep auto policy",
    )
    parser.add_argument("--max-images", type=int, default=0, help="scan mode only: 0 means all images")
    parser.add_argument("--scan", action="store_true", help="run scan mode (default run is dataset evaluation)")
    parser.add_argument("--evaluate-dataset", action="store_true", help="force dataset evaluation mode")
    parser.add_argument("--dataset-limit", type=int, default=0, help="dataset eval only: 0 means all rows")
    parser.add_argument(
        "--auto-calibrate-value",
        dest="auto_calibrate_value",
        action="store_true",
        help="dataset eval: fit value calibration (alpha/beta/bias) from dataset before evaluation",
    )
    parser.add_argument(
        "--no-auto-calibrate-value",
        dest="auto_calibrate_value",
        action="store_false",
        help="dataset eval: keep fixed value calibration constants from config",
    )
    parser.add_argument("--csv", default="", help="output CSV path (auto-selected by mode when omitted)")
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="open interactive visualizer")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="disable interactive visualizer")
    parser.set_defaults(visualize=True, auto_calibrate_value=False)
    return parser


if __name__ == "__main__":
    AppRunner().main()
