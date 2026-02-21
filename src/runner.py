import argparse
from dataclasses import replace
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import HOUGH_PRESETS, PipelineConfig
from .dataset import ImageDataset
from .evaluation import Evaluator
from .ground_truth import GroundTruthRepository, normalize_group_name
from .processor_circles import CirclePipelineProcessor, PipelineResult


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
        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._results)}] {result.source_path} | "
            f"circles={result.circle_count} | "
            f"param1={params.get('param1')} minR={params.get('minRadius')} maxR={params.get('maxRadius')} | "
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
        evaluator = Evaluator()
        results: list[PipelineResult] = []

        print(f"[INFO] Processing {len(images)} image(s) from: {config.dataset_dir}")
        if eval_groups is None:
            print("[INFO] Evaluation groups: all")
        else:
            print(f"[INFO] Evaluation groups: {', '.join(sorted(eval_groups))}")
        print("=" * 110)
        print(f"{'FILE':<35} {'GROUP':<6} {'PRED':<6} {'TRUE':<6} {'DIFF':<6} {'STATUS':<10}")
        print("=" * 110)
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
                print(f"{str(item.relative_path):<35} {group_name:<6} {'-':<6} {'-':<6} {'-':<6} {'SKIP_GROUP':<10}")
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
                print(f"{str(item.relative_path):<35} {group_name:<6} {'-':<6} {'-':<6} {'-':<6} {'SKIP_NO_GT':<10}")
            else:
                eval_item = evaluator.add_match(
                    relative_path=item.relative_path,
                    group=group_name,
                    predicted=result.circle_count,
                    ground_truth=gt_entry,
                )
                status = "OK" if eval_item.is_correct else "ERR"
                print(
                    f"{str(item.relative_path):<35} {group_name:<6} "
                    f"{eval_item.predicted:<6} {eval_item.expected:<6} {eval_item.diff:<6} {status:<10}"
                )

            if args.save_dir:
                out_dir = Path(args.save_dir)
                out_subdir = out_dir / item.relative_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_file = out_subdir / f"{item.relative_path.stem}_pipeline.png"
                save_pipeline_figure(result, out_file, cols=cols)

        print("=" * 110)
        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        print(f"[INFO] Completed detections: {len(results)}/{len(images)}")
        print(f"[INFO] Evaluated (found in GT): {int(summary['evaluated'])}")
        print(f"[INFO] Skipped (filtered group): {evaluator.skipped_filtered_group}")
        print(f"[INFO] Skipped (missing GT): {evaluator.skipped_missing_ground_truth}")
        if by_group:
            print("-" * 70)
            print(f"{'GROUP':<8} {'EVAL':<8} {'OK':<8} {'ACCURACY':<12} {'MAE':<10}")
            print("-" * 70)
            for group in sorted(by_group):
                row = by_group[group]
                acc_text = f"{float(row['accuracy']):.2f}%"
                print(
                    f"{group:<8} {int(row['evaluated']):<8} {int(row['correct']):<8} "
                    f"{acc_text:<12} {float(row['mae']):<10.2f}"
                )
            print("-" * 70)
        if int(summary["evaluated"]) > 0:
            print(f"[INFO] Perfect matches: {int(summary['correct'])}")
            print(f"[INFO] Accuracy: {float(summary['accuracy']):.2f}%")
            print(f"[INFO] MAE: {float(summary['mae']):.2f} coins/image")
        else:
            print("[WARN] No evaluation rows matched the ground truth.")
        if args.no_view:
            return
        if not results:
            print("[WARN] No successful results to display.")
            return
        PipelineViewer(results, cols=cols).show()

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
