"""CLI runner with evaluation/report flow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.analyzer import Analyzer
from src.config import HOUGH_PRESETS, PipelineConfig
from src.dataset import GroundTruthRepository, ImageDataset
from src.detectors import ValueEstimator
from src.io_utils import (
    export_result_debug,
    fmt_optional_score,
    format_signed_cents,
    format_total_cents,
    group_from_relative_path,
    parse_eval_groups,
    plt,
    save_pipeline_figure,
)
from src.models import AnalysisResult, Evaluation
from src.visualizer import DebugViewer

console = Console()


class AppRunner:
    def run(self) -> None:
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

        processor = Analyzer(config, preset_name=preset_name)
        evaluator = Evaluation()
        ground_truth = GroundTruthRepository()
        results: list[AnalysisResult] = []

        console.print(f"[cyan][INFO] Processing {len(images)} image(s) from:[/cyan] {config.dataset_dir}")
        if eval_groups is None:
            console.print("[cyan][INFO] Evaluation groups:[/cyan] all")
        else:
            console.print(f"[cyan][INFO] Evaluation groups:[/cyan] {', '.join(sorted(eval_groups))}")
        print()

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

        with console.status("[bold green]Processing images and calculating metrics..."):
            for item in images:
                try:
                    result = processor.process_path(item.path)
                except Exception as exc:
                    console.print(f"[bold red][ERR ][/bold red] {item.relative_path}: {exc}")
                    continue

                results.append(result)
                group_name = group_from_relative_path(item.relative_path)
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

                if eval_groups is not None and group_name not in eval_groups:
                    evaluator.add_filtered_group()
                    result.debug_info["status"] = "SKIP_GROUP"
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_GROUP[/yellow]"
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
                    continue

                gt_entry = ground_truth.find(item.relative_path.name, group_name)

                if gt_entry is None:
                    evaluator.add_missing_ground_truth()
                    result.debug_info["status"] = "SKIP_NO_GT"
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_NO_GT[/yellow]"
                    )
                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")
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

                    pred_value_txt = format_total_cents(int(eval_item.predicted_value_cents))

                    if eval_item.expected_value_cents is None:
                        true_value_txt = "n/a"
                        diff_txt = "n/a"
                    else:
                        true_value_txt = format_total_cents(int(eval_item.expected_value_cents))
                        v_diff_val = int(eval_item.value_diff_cents or 0)
                        raw_diff_str = format_signed_cents(v_diff_val)
                        diff_txt = raw_diff_str if v_diff_val == 0 else f"[red]{raw_diff_str}[/red]"

                    result.debug_info.update(
                        {
                            "has_ground_truth": True,
                            "true_coin_count": int(eval_item.expected),
                            "true_value_cents": None if eval_item.expected_value_cents is None else int(eval_item.expected_value_cents),
                            "coin_diff": int(eval_item.diff),
                            "value_diff_cents": None if eval_item.value_diff_cents is None else int(eval_item.value_diff_cents),
                            "status": "OK" if eval_item.is_correct else "ERR",
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

                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")

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

        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        value_tolerance_cents = int(summary.get("value_tolerance_cents", 100))
        value_tolerance_label = format_total_cents(value_tolerance_cents)

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
            f"  accuracy(<= {value_tolerance_label})=[green]{float(summary['value_accuracy']):.2f}%[/green] | "
            f"exact={float(summary.get('value_accuracy_exact', 0.0)):.2f}% | "
            f"mae={float(summary['value_mae_eur']):.2f} EUR/image "
            f"({float(summary['value_mae_cents']):.1f} cents) | "
            f"correct={int(summary['value_correct'])}/{int(summary['value_evaluated'])} | "
            f"quality={float(summary.get('value_error_score', 0.0)):.2f}"
        )

        console.print("[bold]Combined Score:[/bold]")
        console.print(
            f"  coin_score={float(summary['coin_score']):.2f} | "
            f"value_score={fmt_optional_score(summary.get('value_score'))} | "
            f"general_score=[bold magenta]{float(summary['general_score']):.2f}[/bold magenta]\n"
        )

        if by_group:
            group_table = Table(show_header=True, header_style="bold cyan", title="Summary By Group")
            group_table.add_column("GROUP")
            group_table.add_column("EVAL", justify="right")
            group_table.add_column("COIN ACC", justify="right")
            group_table.add_column("COIN MAE", justify="right")
            group_table.add_column(f"VAL ACC <= {value_tolerance_label}", justify="right")
            group_table.add_column("VAL MAE (EUR)", justify="right")
            group_table.add_column("GENERAL", justify="right", style="bold magenta")

            for group in sorted(by_group):
                row = by_group[group]
                group_table.add_row(
                    group,
                    str(int(row["evaluated"])),
                    f"{float(row['coin_accuracy']):.2f}%",
                    f"{float(row['coin_mae']):.2f}",
                    f"{float(row['value_accuracy']):.2f}%",
                    f"{float(row['value_mae_eur']):.2f}",
                    f"{float(row['general_score']):.2f}",
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
        self.run()

    @staticmethod
    def _save_result(
        result: AnalysisResult,
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
    def _get_value_breakdown_str(result: AnalysisResult) -> str:
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
        return f"VALUE      {format_total_cents(total_cents):<12} | {detail}"

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Modular pipeline runner with project-style dataset loading and evaluation."
        )
        parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root folder (default: data/images).")
        parser.add_argument("--preset", type=str, default=None, help=f"Hough preset name (default: {PipelineConfig().active_preset}).")
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
        return parser


OneFileRunner = AppRunner


def main() -> None:
    AppRunner().run()
