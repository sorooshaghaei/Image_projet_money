import argparse
from dataclasses import replace
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import HOUGH_PRESETS, PipelineConfig
from .dataset import ImageDataset
from .evaluation import Evaluation
from .ground_truth import GroundTruthRepository, normalize_group_name
from .processor_circles import CirclePipelineProcessor, PipelineResult
from .value_estimator import ValueEstimator


class PipelineViewer:
    def __init__(self, results: list[PipelineResult], cols: int = 3):
        self._results = results
        self._cols = max(1, cols)
        self._idx = 0
        self._rows = max(1, ceil(max(len(result.steps) for result in results) / self._cols))
        self._fig = None
        self._axes = None

    def show(self) -> None:
        if not self._results:
            print("[WARN] No pipeline results to display.")
            return
        backend = str(plt.get_backend()).lower()
        non_interactive_tokens = ("agg", "inline", "pdf", "svg", "ps", "template")
        if any(token in backend for token in non_interactive_tokens):
            print(
                f"[WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                "viewer window cannot open."
            )
            print("[WARN] Use a GUI backend (e.g. `MPLBACKEND=tkagg`) or run with `--save-dir outputs/pipeline --no-view`.")
            return

        self._fig, axes = plt.subplots(
            self._rows,
            self._cols,
            figsize=(4.6 * self._cols, 3.8 * self._rows),
        )
        self._axes = self._normalize_axes(axes, self._rows, self._cols)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()
        plt.show()

    def _on_key(self, event) -> None:
        if event.key in ("right", "d", "n", " "):
            self._idx = (self._idx + 1) % len(self._results)
            self._render()
        elif event.key in ("left", "a", "p"):
            self._idx = (self._idx - 1) % len(self._results)
            self._render()
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _render(self) -> None:
        result = self._results[self._idx]
        params = result.hough_params
        total_cents = int(result.debug_info.get("total_cents", 0))
        euros = total_cents // 100
        cents = total_cents % 100
        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._results)}] {result.source_path} | "
            f"circles={result.circle_count} | "
            f"param1={params.get('param1')} minR={params.get('minRadius')} maxR={params.get('maxRadius')} | "
            f"value={euros}EUR{cents:02d}c | "
            "next: right/d/space, prev: left/a, quit: q/esc",
            fontsize=11,
        )

        for flat_idx in range(self._rows * self._cols):
            row, col = divmod(flat_idx, self._cols)
            ax = self._axes[row, col]
            ax.clear()
            ax.axis("off")

            if flat_idx >= len(result.steps):
                continue

            step = result.steps[flat_idx]
            if step.cmap == "gray":
                ax.imshow(step.image, cmap="gray")
            else:
                ax.imshow(step.image)
            ax.set_title(step.name, fontsize=10)

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()

    @staticmethod
    def _normalize_axes(axes, rows: int, cols: int) -> np.ndarray:
        if rows == 1 and cols == 1:
            return np.array([[axes]])
        if rows == 1:
            return np.array([axes])
        if cols == 1:
            return np.array([[ax] for ax in axes])
        return axes


class AppRunner:
    def main(self) -> None:
        args = self._build_parser().parse_args()
        cols = max(1, args.cols)
        eval_groups = _parse_eval_groups(args.eval_groups)
        print(f"[INFO] Matplotlib backend: {plt.get_backend()}")
        config = PipelineConfig()
        if args.dataset_dir is not None:
            config = replace(config, dataset_dir=Path(args.dataset_dir).expanduser().resolve())

        preset_name = args.preset or config.active_preset
        if preset_name not in HOUGH_PRESETS:
            available = ", ".join(sorted(HOUGH_PRESETS))
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

        dataset = ImageDataset(config.dataset_dir, config.valid_extensions)
        images = dataset.list_images(limit=args.limit)
        if not images:
            print(f"[WARN] No images found under: {config.dataset_dir}")
            return

        processor = CirclePipelineProcessor(config, preset_name=preset_name)
        ground_truth = GroundTruthRepository()
        evaluator = Evaluation()
        results: list[PipelineResult] = []

        print(f"[INFO] Processing {len(images)} image(s) from: {config.dataset_dir}")
        if eval_groups is None:
            print("[INFO] Evaluation groups: all")
        else:
            print(f"[INFO] Evaluation groups: {', '.join(sorted(eval_groups))}")
        print("=" * 182)
        print(
            f"{'FILE':<35} {'GROUP':<6} "
            f"{'C_PRED':<6} {'C_TRUE':<6} {'C_DIFF':<6} "
            f"{'V_PRED':<14} {'V_TRUE':<14} {'V_DIFF':<12} {'STATUS':<10}"
        )
        print("=" * 182)
        for index, item in enumerate(images, start=1):
            try:
                result = processor.process_path(item.path)
            except Exception as exc:
                print(f"[ERR ] {item.relative_path}: {exc}")
                continue

            results.append(result)
            group_name = _group_from_relative_path(item.relative_path)
            if eval_groups is not None and group_name not in eval_groups:
                evaluator.add_filtered_group()
                print(
                    f"{str(item.relative_path):<35} {group_name:<6} "
                    f"{'-':<6} {'-':<6} {'-':<6} "
                    f"{'-':<14} {'-':<14} {'-':<12} {'SKIP_GROUP':<10}"
                )
                if args.save_dir:
                    out_dir = Path(args.save_dir)
                    out_subdir = out_dir / item.relative_path.parent
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    out_file = out_subdir / f"{item.relative_path.stem}_pipeline.png"
                    save_pipeline_figure(result, out_file, cols=cols)
                continue

            gt_entry = ground_truth.find(item.relative_path.name, group_name)
            if gt_entry is None:
                evaluator.add_missing_ground_truth()
                print(
                    f"{str(item.relative_path):<35} {group_name:<6} "
                    f"{'-':<6} {'-':<6} {'-':<6} "
                    f"{'-':<14} {'-':<14} {'-':<12} {'SKIP_NO_GT':<10}"
                )
                self._print_value_summary_line(result)
            else:
                predicted_value_cents = int(result.debug_info.get("total_cents", 0))
                eval_item = evaluator.add_match(
                    relative_path=item.relative_path,
                    group=group_name,
                    predicted=result.circle_count,
                    ground_truth=gt_entry,
                    predicted_value_cents=predicted_value_cents,
                )
                status = "OK" if eval_item.is_correct else "ERR"
                pred_value_txt = _format_total_cents(int(eval_item.predicted_value_cents))
                if eval_item.expected_value_cents is None:
                    true_value_txt = "n/a"
                    value_diff_txt = "n/a"
                else:
                    true_value_txt = _format_total_cents(int(eval_item.expected_value_cents))
                    value_diff_txt = _format_signed_cents(int(eval_item.value_diff_cents or 0))
                print(
                    f"{str(item.relative_path):<35} {group_name:<6} "
                    f"{eval_item.predicted:<6} {eval_item.expected:<6} {eval_item.diff:<6} "
                    f"{pred_value_txt:<14} {true_value_txt:<14} {value_diff_txt:<12} {status:<10}"
                )
                self._print_value_summary_line(result)

            if args.save_dir:
                out_dir = Path(args.save_dir)
                out_subdir = out_dir / item.relative_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_file = out_subdir / f"{item.relative_path.stem}_pipeline.png"
                save_pipeline_figure(result, out_file, cols=cols)

        print("=" * 182)
        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        print(f"[INFO] Completed detections: {len(results)}/{len(images)}")
        print(f"[INFO] Evaluated (found in GT): {int(summary['evaluated'])}")
        print(f"[INFO] Skipped (filtered group): {evaluator.skipped_filtered_group}")
        print(f"[INFO] Skipped (missing GT): {evaluator.skipped_missing_ground_truth}")
        print("[INFO] Coin Metrics:")
        print(
            f"  accuracy={float(summary['coin_accuracy']):.2f}% | "
            f"mae={float(summary['coin_mae']):.2f} coins/image | "
            f"correct={int(summary['coin_correct'])}/{int(summary['evaluated'])}"
        )
        print("[INFO] Value Metrics:")
        print(
            f"  accuracy={float(summary['value_accuracy']):.2f}% | "
            f"mae={float(summary['value_mae_eur']):.2f} EUR/image "
            f"({float(summary['value_mae_cents']):.1f} cents) | "
            f"correct={int(summary['value_correct'])}/{int(summary['value_evaluated'])}"
        )
        print("[INFO] Combined Score:")
        print(
            f"  coin_score={float(summary['coin_score']):.2f} | "
            f"value_score={_fmt_optional_score(summary.get('value_score'))} | "
            f"general_score={float(summary['general_score']):.2f}"
        )
        if by_group:
            print("-" * 120)
            print(
                f"{'GROUP':<8} {'EVAL':<6} "
                f"{'COIN_ACC':<10} {'COIN_MAE':<10} "
                f"{'VAL_ACC':<10} {'VAL_MAE_EUR':<12} {'GENERAL':<10}"
            )
            print("-" * 120)
            for group in sorted(by_group):
                row = by_group[group]
                print(
                    f"{group:<8} {int(row['evaluated']):<6} "
                    f"{float(row['coin_accuracy']):>8.2f}%  {float(row['coin_mae']):>8.2f}   "
                    f"{float(row['value_accuracy']):>8.2f}%  {float(row['value_mae_eur']):>10.2f}   "
                    f"{float(row['general_score']):>8.2f}"
                )
            print("-" * 120)
        if int(summary["evaluated"]) > 0:
            print(f"[INFO] Perfect coin matches: {int(summary['coin_correct'])}")
            print(f"[INFO] Coin accuracy: {float(summary['coin_accuracy']):.2f}%")
            print(f"[INFO] Coin MAE: {float(summary['coin_mae']):.2f} coins/image")
            print(f"[INFO] FINAL GENERAL SCORE: {float(summary['general_score']):.2f}")
        else:
            print("[WARN] No evaluation rows matched the ground truth.")
        if args.no_view:
            return
        if not results:
            print("[WARN] No successful results to display.")
            return
        PipelineViewer(results, cols=cols).show()

    @staticmethod
    def _print_value_summary_line(result: PipelineResult) -> None:
        total_cents = int(result.debug_info.get("total_cents", 0))
        counts = result.debug_info.get("value_counts", {})
        if not isinstance(counts, dict):
            counts = {}

        non_zero = [
            f"{ValueEstimator.DENOM_TEXT[d]}:{int(counts.get(d, 0))}"
            for d in ValueEstimator.DENOM_PRINT_ORDER
            if int(counts.get(d, 0)) > 0
        ]
        detail = ", ".join(non_zero) if non_zero else "none"
        print(f"{'':<35} {'':<6} {'':<6} {'':<6} {'':<6} {'VALUE':<10} {_format_total_cents(total_cents)} | {detail}")

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run the notebook-based circle detection pipeline over a full dataset."
        )
        parser.add_argument(
            "--dataset-dir",
            type=str,
            default=None,
            help="Dataset root folder (default: data/images).",
        )
        parser.add_argument(
            "--preset",
            type=str,
            default=None,
            help=f"Hough preset name (default: {PipelineConfig().active_preset}).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of images for quick debugging.",
        )
        parser.add_argument(
            "--cols",
            type=int,
            default=3,
            help="Number of columns in the pipeline view.",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default=None,
            help="Optional output folder to save per-image pipeline grids.",
        )
        parser.add_argument(
            "--no-view",
            action="store_true",
            help="Process images without opening the interactive viewer.",
        )
        parser.add_argument(
            "--eval-groups",
            nargs="*",
            default=None,
            help="Evaluate only these groups (e.g. --eval-groups gp1 gp2 or --eval-groups gp1,gp2).",
        )
        return parser


def save_pipeline_figure(result: PipelineResult, output_path: Path, cols: int = 3) -> None:
    rows = ceil(len(result.steps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows))
    axes_matrix = PipelineViewer._normalize_axes(axes, rows, cols)
    fig.suptitle(f"{result.source_path.name} | circles={result.circle_count}", fontsize=11)

    for flat_idx in range(rows * cols):
        row, col = divmod(flat_idx, cols)
        ax = axes_matrix[row, col]
        ax.axis("off")
        if flat_idx >= len(result.steps):
            continue

        step = result.steps[flat_idx]
        if step.cmap == "gray":
            ax.imshow(step.image, cmap="gray")
        else:
            ax.imshow(step.image)
        ax.set_title(step.name, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _group_from_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) <= 1:
        return ""
    return normalize_group_name(parts[0])


def _parse_eval_groups(raw_groups: list[str] | None) -> set[str] | None:
    if raw_groups is None:
        return None
    groups: set[str] = set()
    for token in raw_groups:
        for chunk in token.split(","):
            group = normalize_group_name(chunk.strip())
            if group:
                groups.add(group)
    return groups if groups else None


def _format_total_cents(total_cents: int) -> str:
    euros = total_cents // 100
    cents = total_cents % 100
    return f"{euros} EUR {cents:02d} c"


def _format_signed_cents(diff_cents: int) -> str:
    sign = "+" if diff_cents >= 0 else "-"
    euros_abs = abs(int(diff_cents)) // 100
    cents_abs = abs(int(diff_cents)) % 100
    return f"{sign}{euros_abs} EUR {cents_abs:02d} c"


def _fmt_optional_score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"
