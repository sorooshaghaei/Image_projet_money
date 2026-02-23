#!/usr/bin/env python3
"""Single-file runner for notebook-style pipeline with project dataset evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Rich Imports ---
from rich.console import Console
from rich.table import Table
from rich.text import Text
# --------------------

from src.config import HOUGH_PRESETS, PipelineConfig
from src.dataset import ImageDataset
from src.evaluation import Evaluation
from src.ground_truth import GroundTruthRepository, normalize_group_name
from src.processor_circles import CirclePipelineProcessor, PipelineResult
from src.value_estimator import ValueEstimator

# Initialize the rich console globally for the script
console = Console()


class OneFileViewer:
    def __init__(self, results: list[PipelineResult], cols: int = 3, final_only: bool = False):
        self._results = results
        self._cols = max(1, cols)
        self._final_only = bool(final_only)
        self._idx = 0
        max_steps = max((len(result.steps) for result in results), default=1)
        self._rows = 1 if self._final_only else max(1, ceil(max_steps / self._cols))
        self._fig = None
        self._axes = None

    def show(self) -> None:
        if not self._results:
            console.print("[yellow][WARN] No pipeline results to display.[/yellow]")
            return

        backend = str(plt.get_backend()).lower()
        non_interactive_tokens = ("agg", "inline", "pdf", "svg", "ps", "template")
        if any(token in backend for token in non_interactive_tokens):
            console.print(
                f"[yellow][WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                "viewer window cannot open.[/yellow]"
            )
            return

        self._fig, axes = plt.subplots(self._rows, self._cols, figsize=(4.4 * self._cols, 3.7 * self._rows))
        self._axes = _normalize_axes(axes, self._rows, self._cols)
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
        steps = result.steps[-1:] if self._final_only else result.steps
        total_cents = int(result.debug_info.get("total_cents", 0))
        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._results)}] {result.source_path.name} | "
            f"coins={result.circle_count} | value={_format_total_cents(total_cents)} | "
            "next: right/d/space, prev: left/a, quit: q/esc",
            fontsize=11,
        )

        for flat_idx in range(self._rows * self._cols):
            row, col = divmod(flat_idx, self._cols)
            ax = self._axes[row, col]
            ax.clear()
            ax.axis("off")
            if flat_idx >= len(steps):
                continue
            step = steps[flat_idx]
            if step.cmap == "gray":
                ax.imshow(step.image, cmap="gray")
            else:
                ax.imshow(step.image)
            ax.set_title(step.name if not self._final_only else f"Final: {step.name}", fontsize=10)

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()


class OneFileRunner:
    def run(self) -> None:
        args = self._build_parser().parse_args()
        cols = max(1, args.cols)
        eval_groups = _parse_eval_groups(args.eval_groups)

        console.print(f"[cyan][INFO] Matplotlib backend:[/cyan] {plt.get_backend()}")
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
            console.print(f"[yellow][WARN] No images found under: {config.dataset_dir}[/yellow]")
            return

        processor = CirclePipelineProcessor(config, preset_name=preset_name)
        evaluator = Evaluation()
        ground_truth = GroundTruthRepository()
        results: list[PipelineResult] = []

        console.print(f"[cyan][INFO] Processing {len(images)} image(s) from:[/cyan] {config.dataset_dir}")
        if eval_groups is None:
            console.print("[cyan][INFO] Evaluation groups:[/cyan] all")
        else:
            console.print(f"[cyan][INFO] Evaluation groups:[/cyan] {', '.join(sorted(eval_groups))}")
        print()

        # Set up the main results table
        main_table = Table(show_header=True, header_style="bold magenta", expand=True)
        main_table.add_column("FILE", style="dim", width=25)
        main_table.add_column("GROUP", justify="center")
        main_table.add_column("C_PRED", justify="right")
        main_table.add_column("C_TRUE", justify="right")
        main_table.add_column("C_DIFF", justify="right")
        main_table.add_column("V_PRED", justify="right", style="cyan")
        main_table.add_column("V_TRUE", justify="right", style="cyan")
        main_table.add_column("V_DIFF", justify="right")
        main_table.add_column("STATUS", justify="center", style="bold")

        # Process images with a spinner so the terminal doesn't look frozen
        with console.status("[bold green]Processing images and calculating metrics...") as status:
            for item in images:
                try:
                    result = processor.process_path(item.path)
                except Exception as exc:
                    console.print(f"[bold red][ERR ][/bold red] {item.relative_path}: {exc}")
                    continue

                results.append(result)
                group_name = _group_from_relative_path(item.relative_path)
                
                # Handling Filtered Groups
                if eval_groups is not None and group_name not in eval_groups:
                    evaluator.add_filtered_group()
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_GROUP[/yellow]"
                    )
                    if args.save_dir:
                        self._save_result(result, item.relative_path, Path(args.save_dir), cols, config.viewer_final_only)
                    continue

                gt_entry = ground_truth.find(item.relative_path.name, group_name)
                
                # Handling Missing Ground Truth
                if gt_entry is None:
                    evaluator.add_missing_ground_truth()
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_NO_GT[/yellow]"
                    )
                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")
                
                # Normal Evaluation Process
                else:
                    pred_value_cents = int(result.debug_info.get("total_cents", 0))
                    eval_item = evaluator.add_match(
                        relative_path=item.relative_path,
                        group=group_name,
                        predicted=result.circle_count,
                        ground_truth=gt_entry,
                        predicted_value_cents=pred_value_cents,
                    )
                    
                    status_col = "[green]OK[/green]" if eval_item.is_correct else "[red]ERR[/red]"
                    
                    c_diff_val = int(eval_item.diff)
                    c_diff_txt = str(c_diff_val) if c_diff_val == 0 else f"[red]{c_diff_val}[/red]"

                    pred_value_txt = _format_total_cents(int(eval_item.predicted_value_cents))
                    
                    if eval_item.expected_value_cents is None:
                        true_value_txt = "n/a"
                        diff_txt = "n/a"
                    else:
                        true_value_txt = _format_total_cents(int(eval_item.expected_value_cents))
                        v_diff_val = int(eval_item.value_diff_cents or 0)
                        raw_diff_str = _format_signed_cents(v_diff_val)
                        diff_txt = raw_diff_str if v_diff_val == 0 else f"[red]{raw_diff_str}[/red]"

                    # Add main row
                    main_table.add_row(
                        str(item.relative_path), group_name, str(eval_item.predicted), str(eval_item.expected), 
                        c_diff_txt, pred_value_txt, true_value_txt, diff_txt, status_col
                    )
                    
                    # Add breakdown underneath
                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")

                if args.save_dir:
                    self._save_result(result, item.relative_path, Path(args.save_dir), cols, config.viewer_final_only)

        # Print the fully assembled table
        console.print(main_table)

        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        
        # Print Summary Metrics
        console.print("\n[bold cyan]--- OVERALL METRICS ---[/bold cyan]")
        console.print(f"[INFO] Completed detections: [bold]{len(results)}/{len(images)}[/bold]")
        console.print(f"[INFO] Evaluated (found in GT): [bold]{int(summary['evaluated'])}[/bold]")
        console.print(f"[INFO] Skipped (filtered group): {evaluator.skipped_filtered_group}")
        console.print(f"[INFO] Skipped (missing GT): {evaluator.skipped_missing_ground_truth}")
        
        console.print("\n[bold]Coin Metrics:[/bold]")
        console.print(
            f"  accuracy=[green]{float(summary['coin_accuracy']):.2f}%[/green] | "
            f"mae={float(summary['coin_mae']):.2f} coins/image | "
            f"correct={int(summary['coin_correct'])}/{int(summary['evaluated'])}"
        )
        
        console.print("[bold]Value Metrics:[/bold]")
        console.print(
            f"  accuracy=[green]{float(summary['value_accuracy']):.2f}%[/green] | "
            f"mae={float(summary['value_mae_eur']):.2f} EUR/image "
            f"({float(summary['value_mae_cents']):.1f} cents) | "
            f"correct={int(summary['value_correct'])}/{int(summary['value_evaluated'])}"
        )
        
        console.print("[bold]Combined Score:[/bold]")
        console.print(
            f"  coin_score={float(summary['coin_score']):.2f} | "
            f"value_score={_fmt_optional_score(summary.get('value_score'))} | "
            f"general_score=[bold magenta]{float(summary['general_score']):.2f}[/bold magenta]\n"
        )

        # Print Group Summary Table
        if by_group:
            group_table = Table(show_header=True, header_style="bold cyan", title="Summary By Group")
            group_table.add_column("GROUP")
            group_table.add_column("EVAL", justify="right")
            group_table.add_column("COIN ACC", justify="right")
            group_table.add_column("COIN MAE", justify="right")
            group_table.add_column("VAL ACC", justify="right")
            group_table.add_column("VAL MAE (EUR)", justify="right")
            group_table.add_column("GENERAL", justify="right", style="bold magenta")

            for group in sorted(by_group):
                row = by_group[group]
                group_table.add_row(
                    group,
                    str(int(row['evaluated'])),
                    f"{float(row['coin_accuracy']):.2f}%",
                    f"{float(row['coin_mae']):.2f}",
                    f"{float(row['value_accuracy']):.2f}%",
                    f"{float(row['value_mae_eur']):.2f}",
                    f"{float(row['general_score']):.2f}"
                )
            console.print(group_table)
            print()

        if args.no_view:
            return
        OneFileViewer(results, cols=cols, final_only=config.viewer_final_only).show()

    @staticmethod
    def _save_result(
        result: PipelineResult,
        relative_path: Path,
        save_dir: Path,
        cols: int,
        final_only: bool,
    ) -> None:
        out_subdir = save_dir / relative_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / f"{relative_path.stem}_pipeline.png"
        save_pipeline_figure(result, out_file, cols=cols, final_only=final_only)

    @staticmethod
    def _get_value_breakdown_str(result: PipelineResult) -> str:
        """Helper to return the formatted breakdown string instead of printing directly."""
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
        return f"VALUE      {_format_total_cents(total_cents):<12} | {detail}"

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "One-file script version of the notebook pipeline with project-style dataset loading and evaluation."
            )
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
            help="Number of columns in pipeline grid view.",
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
            help="Process/evaluate without opening interactive viewer.",
        )
        parser.add_argument(
            "--eval-groups",
            nargs="*",
            default=None,
            help="Evaluate only these groups (e.g. --eval-groups gp1 gp2 or --eval-groups gp1,gp2).",
        )
        return parser


def save_pipeline_figure(
    result: PipelineResult,
    output_path: Path,
    cols: int = 3,
    final_only: bool = False,
) -> None:
    cols = max(1, int(cols))
    rows = 1 if final_only else max(1, ceil(max(1, len(result.steps)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows))
    axes_matrix = _normalize_axes(axes, rows, cols)
    steps = result.steps[-1:] if final_only else result.steps

    for flat_idx in range(rows * cols):
        row, col = divmod(flat_idx, cols)
        ax = axes_matrix[row, col]
        ax.axis("off")
        if flat_idx >= len(steps):
            continue
        step = steps[flat_idx]
        if step.cmap == "gray":
            ax.imshow(step.image, cmap="gray")
        else:
            ax.imshow(step.image)
        ax.set_title(step.name if not final_only else f"Final: {step.name}", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _normalize_axes(axes, rows: int, cols: int) -> np.ndarray:
    if rows == 1 and cols == 1:
        return np.array([[axes]])
    if rows == 1:
        return np.array([axes])
    if cols == 1:
        return np.array([[ax] for ax in axes])
    return axes


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


if __name__ == "__main__":
    OneFileRunner().run()