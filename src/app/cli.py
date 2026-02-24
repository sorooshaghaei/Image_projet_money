"""CLI runner with evaluation/report flow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from src.common.debug_export import export_result_debug
from src.common.formatters import (
    format_cents_compact,
    format_diff_cents_compact,
    group_from_relative_path,
    parse_eval_groups,
)
from src.common.plotting import plt, save_pipeline_figure
from src.data.dataset import ImageDataset
from src.data.ground_truth import GroundTruthRepository
from src.evaluation.metrics import Evaluation
from src.evaluation.reporting import write_evaluation_report
from src.pipeline.config import HOUGH_PRESETS, PipelineConfig
from src.pipeline.models import AnalysisResult
from src.pipeline.orchestrator import Analyzer
from src.ui.debug_viewer import DebugViewer

console = Console()


class AppRunner:
    """CLI application shell for batch processing + evaluation + viewer."""

    def run(self) -> None:
        """Run full CLI flow from argument parsing to optional debug viewer."""
        args = self.build_parser().parse_args()
        cols = max(1, args.cols)
        eval_groups = parse_eval_groups(args.eval_groups)
        debug_export_dir = None
        if args.debug_export_dir:
            debug_export_dir = Path(args.debug_export_dir).expanduser().resolve()

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

        images_to_process = images
        filtered_out_count = 0
        if eval_groups is not None:
            selected = []
            for item in images:
                group_name = group_from_relative_path(item.relative_path)
                if group_name in eval_groups:
                    selected.append(item)
                else:
                    filtered_out_count += 1
            images_to_process = selected

        if eval_groups is not None and not images_to_process:
            console.print("[yellow][WARN] No images match --eval-groups filter.[/yellow]")
            return

        processor = Analyzer(config, preset_name=preset_name)
        evaluator = Evaluation()
        ground_truth = GroundTruthRepository()
        results: list[AnalysisResult] = []
        evaluation_rows: list[dict[str, Any]] = []

        if filtered_out_count > 0:
            for _ in range(filtered_out_count):
                evaluator.add_filtered_group()

        console.print(f"[cyan][INFO] Processing {len(images_to_process)} image(s) from:[/cyan] {config.dataset_dir}")
        if eval_groups is None:
            console.print("[cyan][INFO] Evaluation groups:[/cyan] all")
        else:
            console.print(f"[cyan][INFO] Evaluation groups:[/cyan] {', '.join(sorted(eval_groups))}")
        print()

        main_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAVY, expand=False)
        main_table.add_column("FILE", style="dim", width=20, overflow="ellipsis", no_wrap=True)
        main_table.add_column("GROUP", justify="center", width=5, no_wrap=True)
        main_table.add_column("C_PRED", justify="right", no_wrap=True)
        main_table.add_column("C_TRUE", justify="right", no_wrap=True)
        main_table.add_column("C_DIFF", justify="right", no_wrap=True)
        main_table.add_column("V_PRED", justify="right", style="cyan", width=8, no_wrap=True)
        main_table.add_column("V_TRUE", justify="right", style="cyan", width=8, no_wrap=True)
        main_table.add_column("V_DIFF", justify="right", width=9, no_wrap=True)
        main_table.add_column("STATUS", justify="center", style="bold", width=10, no_wrap=True)

        with console.status("[bold green]Processing images and calculating metrics..."):
            for item in images_to_process:
                group_name = group_from_relative_path(item.relative_path)
                try:
                    result = processor.process_path(item.path)
                except Exception as exc:
                    console.print(f"[bold red][ERR ][/bold red] {item.relative_path}: {exc}")
                    continue

                results.append(result)
                pred_value_cents = int(result.debug_info.get("total_cents", 0))
                result.debug_info.update(
                    {
                        "relative_path": str(item.relative_path),
                        "group": group_name,
                        "predicted_coin_count": int(result.circle_count),
                        "predicted_value_cents": pred_value_cents,
                        "has_ground_truth": False,
                        "true_coin_count": None,
                        "true_value_cents": None,
                        "coin_diff": None,
                        "value_diff_cents": None,
                        "status": "PREDICTED_ONLY",
                    }
                )

                gt_entry = ground_truth.find(item.relative_path.name, group_name)

                if gt_entry is None:
                    evaluator.add_missing_ground_truth()
                    result.debug_info["status"] = "SKIP_NO_GT"
                    evaluation_rows.append(
                        {
                            "file": str(item.relative_path),
                            "group": group_name,
                            "status": "SKIP_NO_GT",
                            "coin_pred": int(result.circle_count),
                            "coin_true": None,
                            "coin_diff": None,
                            "coin_abs_diff": None,
                            "value_pred_cents": pred_value_cents,
                            "value_true_cents": None,
                            "value_diff_cents": None,
                            "value_abs_diff_cents": None,
                        }
                    )
                    main_table.add_row(
                        str(item.relative_path),
                        group_name,
                        str(int(result.circle_count)),
                        "-",
                        "-",
                        format_cents_compact(pred_value_cents),
                        "-",
                        "-",
                        "[yellow]SKIP_NO_GT[/yellow]",
                    )
                else:
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

                    pred_value_txt = format_cents_compact(int(eval_item.predicted_value_cents))

                    if eval_item.expected_value_cents is None:
                        true_value_txt = "-"
                        diff_txt = "-"
                    else:
                        true_value_txt = format_cents_compact(int(eval_item.expected_value_cents))
                        v_diff_val = int(eval_item.value_diff_cents or 0)
                        raw_diff_str = format_diff_cents_compact(v_diff_val)
                        diff_txt = raw_diff_str if v_diff_val == 0 else f"[red]{raw_diff_str}[/red]"

                    result.debug_info.update(
                        {
                            "has_ground_truth": True,
                            "true_coin_count": int(eval_item.expected),
                            "true_value_cents": None
                            if eval_item.expected_value_cents is None
                            else int(eval_item.expected_value_cents),
                            "coin_diff": int(eval_item.diff),
                            "value_diff_cents": None
                            if eval_item.value_diff_cents is None
                            else int(eval_item.value_diff_cents),
                            "status": "OK" if eval_item.is_correct else "ERR",
                        }
                    )
                    evaluation_rows.append(
                        {
                            "file": str(item.relative_path),
                            "group": group_name,
                            "status": "OK" if eval_item.is_correct else "ERR",
                            "coin_pred": int(eval_item.predicted),
                            "coin_true": int(eval_item.expected),
                            "coin_diff": int(eval_item.diff),
                            "coin_abs_diff": int(abs(eval_item.diff)),
                            "value_pred_cents": int(eval_item.predicted_value_cents),
                            "value_true_cents": None
                            if eval_item.expected_value_cents is None
                            else int(eval_item.expected_value_cents),
                            "value_diff_cents": None
                            if eval_item.value_diff_cents is None
                            else int(eval_item.value_diff_cents),
                            "value_abs_diff_cents": None
                            if eval_item.value_abs_diff_cents is None
                            else int(eval_item.value_abs_diff_cents),
                        }
                    )

                    main_table.add_row(
                        str(item.relative_path),
                        group_name,
                        str(eval_item.predicted),
                        str(eval_item.expected),
                        c_diff_txt,
                        pred_value_txt,
                        true_value_txt,
                        diff_txt,
                        status_col,
                    )

                if args.save_dir:
                    self._save_result(result, item.relative_path, Path(args.save_dir), cols, config.viewer_final_only)

                if debug_export_dir is not None:
                    try:
                        export_result_debug(
                            result=result,
                            export_root=debug_export_dir,
                            step_index=max(0, len(result.steps) - 1),
                            final_only=True,
                            panel_text=None,
                        )
                    except Exception as exc:
                        console.print(
                            f"[yellow][WARN] Could not auto-export debug for {item.relative_path}: {exc}[/yellow]"
                        )

        console.print(main_table)

        report_path = Path(args.evaluation_report).expanduser().resolve()
        write_evaluation_report(evaluation_rows, report_path)

        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        console.print("\n[bold cyan]EVALUATION SUMMARY[/bold cyan]")

        overview_table = Table(show_header=False, box=box.SIMPLE, expand=False)
        overview_table.add_column("Metric", style="bold cyan")
        overview_table.add_column("Value", justify="right")
        overview_table.add_row("Dataset images", str(len(images)))
        overview_table.add_row("Processed images", str(len(images_to_process)))
        overview_table.add_row("Completed detections", f"{len(results)}/{len(images_to_process)}")
        overview_table.add_row("Evaluated (found in GT)", str(int(summary["evaluated"])))
        overview_table.add_row("Skipped (filtered group)", str(evaluator.skipped_filtered_group))
        overview_table.add_row("Skipped (missing GT)", str(evaluator.skipped_missing_ground_truth))
        console.print(overview_table)

        coin_metrics_table = Table(title="Coin Number Estimation", show_header=False, box=box.ROUNDED, expand=False)
        coin_metrics_table.add_column("Metric", style="bold")
        coin_metrics_table.add_column("Value", justify="right")
        coin_metrics_table.add_row("Accuracy (exact match)", f"{float(summary['coin_accuracy']):.2f}%")
        coin_metrics_table.add_row("Correct", f"{int(summary['coin_correct'])}/{int(summary['evaluated'])}")
        console.print(coin_metrics_table)

        value_metrics_table = Table(title="Value Estimation", show_header=False, box=box.ROUNDED, expand=False)
        value_metrics_table.add_column("Metric", style="bold")
        value_metrics_table.add_column("Value", justify="right")
        value_metrics_table.add_row("Evaluated", str(int(summary["value_evaluated"])))
        value_metrics_table.add_row(
            "MAE",
            f"{float(summary['value_mae_eur']):.3f} EUR/image ({float(summary['value_mae_cents']):.1f} cents)",
        )
        value_metrics_table.add_row(
            "MSE",
            f"{float(summary.get('value_mse_eur2', 0.0)):.4f} EUR^2/image "
            f"({float(summary.get('value_mse_cents2', 0.0)):.1f} cents^2)",
        )
        console.print(value_metrics_table)

        console.print(f"[dim]Saved per-image evaluation report:[/dim] {report_path}")
        print()

        if by_group:
            group_table = Table(show_header=True, header_style="bold cyan", title="Summary By Group", box=box.SIMPLE_HEAVY)
            group_table.add_column("GROUP")
            group_table.add_column("EVAL", justify="right")
            group_table.add_column("COIN ACC", justify="right")
            group_table.add_column("VAL EVAL", justify="right")
            group_table.add_column("VAL MAE (EUR)", justify="right")
            group_table.add_column("VAL MSE (EUR^2)", justify="right")

            for group in sorted(by_group):
                row = by_group[group]
                group_table.add_row(
                    group,
                    str(int(row["evaluated"])),
                    f"{float(row['coin_accuracy']):.2f}%",
                    str(int(row["value_evaluated"])),
                    f"{float(row['value_mae_eur']):.2f}",
                    f"{float(row.get('value_mse_eur2', 0.0)):.4f}",
                )
            console.print(group_table)

            print()

        if args.no_view:
            return

        DebugViewer(
            results,
            cols=cols,
            final_only=config.viewer_final_only,
            debug_export_dir=debug_export_dir,
        ).show()

    def main(self) -> None:
        """Compatibility alias used by `main.py`."""
        self.run()

    @staticmethod
    def _save_result(
        result: AnalysisResult,
        relative_path: Path,
        save_dir: Path,
        cols: int,
        final_only: bool,
    ) -> None:
        """Persist pipeline figure for one processed image."""
        out_subdir = save_dir / relative_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / f"{relative_path.stem}_pipeline.png"
        save_pipeline_figure(result, out_file, cols=cols, final_only=final_only)

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build CLI parser with dataset, evaluation and export options."""
        parser = argparse.ArgumentParser(
            description="Modular pipeline runner with project-style dataset loading and evaluation."
        )
        parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root folder (default: data/images).")
        parser.add_argument(
            "--preset",
            type=str,
            default=None,
            help=f"Hough preset name (default: {PipelineConfig().active_preset}).",
        )
        parser.add_argument("--limit", type=int, default=None, help="Limit number of images for quick debugging.")
        parser.add_argument("--cols", type=int, default=3, help="Number of columns in pipeline grid view.")
        parser.add_argument("--save-dir", type=str, default=None, help="Optional output folder to save per-image pipeline grids.")
        parser.add_argument("--no-view", action="store_true", help="Process/evaluate without opening interactive viewer.")
        parser.add_argument(
            "--eval-groups",
            nargs="*",
            default=None,
            help="Evaluate only these groups (e.g. --eval-groups gp1 gp2 or --eval-groups gp1,gp2).",
        )
        parser.add_argument(
            "--debug-export-dir",
            type=str,
            default=None,
            help=(
                "Optional folder to auto-export per-image debug dumps (.json/.txt). "
                "Inside viewer, press c/x to export the currently displayed debug info."
            ),
        )
        parser.add_argument(
            "--evaluation-report",
            type=str,
            default="reports/evaluation_rows.csv",
            help="CSV path where per-image evaluation rows are saved.",
        )
        return parser


OneFileRunner = AppRunner


def main() -> None:
    AppRunner().run()
