"""Interactive debug viewer for pipeline results."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.detectors import ValueEstimator, _coin_marker_token
from src.io_utils import (
    export_result_debug,
    format_signed_cents,
    format_total_cents,
    is_non_interactive_backend,
    plt,
)
from src.models import AnalysisResult, PipelineStep

console = Console()


class DebugViewer:
    def __init__(
        self,
        results: list[AnalysisResult],
        cols: int = 3,
        final_only: bool = False,
        debug_export_dir: Path | None = None,
    ):
        self._results = results
        self._cols = max(1, cols)
        self._final_only = bool(final_only)
        self._idx = 0
        self._step_idx = 0
        self._fig = None
        self._image_ax = None
        self._info_ax = None
        self._debug_export_dir = Path(debug_export_dir) if debug_export_dir is not None else Path.cwd() / "debug_exports"

    def show(self) -> None:
        if not self._results:
            console.print("[yellow][WARN] No pipeline results to display.[/yellow]")
            return

        backend = str(plt.get_backend())
        if is_non_interactive_backend(backend):
            console.print(
                f"[yellow][WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                "viewer window cannot open.[/yellow]"
            )
            return
        if backend.strip().lower() == "webagg":
            console.print(
                "[cyan][INFO] Using WebAgg backend. Open the local URL printed by Matplotlib "
                "to view the interactive window in your browser.[/cyan]"
            )

        try:
            self._fig = plt.figure(figsize=(16, 9))
            grid = self._fig.add_gridspec(
                1,
                2,
                width_ratios=(4.8, 1.9),
                left=0.025,
                right=0.985,
                top=0.92,
                bottom=0.03,
                wspace=0.04,
            )
            self._image_ax = self._fig.add_subplot(grid[0, 0])
            self._info_ax = self._fig.add_subplot(grid[0, 1])
        except Exception as exc:
            console.print(
                f"[yellow][WARN] Unable to open interactive Matplotlib window using '{plt.get_backend()}': {exc}[/yellow]"
            )
            return
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
        elif event.key in ("up", "w"):
            self._step_idx += 1
            self._render()
        elif event.key in ("down", "s"):
            self._step_idx -= 1
            self._render()
        elif event.key in ("f",):
            self._final_only = not self._final_only
            self._step_idx = 0
            self._render()
        elif event.key in ("c", "x"):
            self._export_current_debug()
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _render(self) -> None:
        result = self._results[self._idx]
        steps = result.steps[-1:] if self._final_only else result.steps
        if not steps:
            return

        self._step_idx = max(0, min(self._step_idx, len(steps) - 1))
        step = steps[self._step_idx]
        total_cents = int(result.debug_info.get("total_cents", 0))
        title_step = f"step {self._step_idx + 1}/{len(steps)}"
        mode_label = "FINAL ONLY" if self._final_only else "FULL PIPELINE"
        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._results)}] {result.source_path.name} | "
            f"coins={result.circle_count} | value={format_total_cents(total_cents)} | "
            f"{title_step} | {mode_label} | "
            "img: right/left | step: up/down | toggle-final: f | export: c/x | quit: q/esc",
            fontsize=11,
        )

        self._image_ax.clear()
        self._image_ax.axis("off")
        if step.cmap == "gray":
            self._image_ax.imshow(step.image, cmap="gray")
        else:
            self._image_ax.imshow(step.image)
        self._image_ax.set_title(step.name, fontsize=12, pad=8)

        panel_text = self._build_info_panel_text(result, step, len(steps))
        self._info_ax.clear()
        self._info_ax.axis("off")
        self._info_ax.text(
            0.03,
            0.98,
            panel_text,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            linespacing=1.35,
            color="white",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#111827",
                "edgecolor": "#374151",
                "linewidth": 1.2,
                "alpha": 0.97,
            },
            transform=self._info_ax.transAxes,
        )
        self._fig.canvas.draw_idle()

    def _export_current_debug(self) -> None:
        if not self._results:
            return
        result = self._results[self._idx]
        steps = result.steps[-1:] if self._final_only else result.steps
        if not steps:
            console.print("[yellow][WARN] No pipeline steps available to export debug info.[/yellow]")
            return
        self._step_idx = max(0, min(self._step_idx, len(steps) - 1))
        step = steps[self._step_idx]
        panel_text = self._build_info_panel_text(result, step, len(steps))
        try:
            json_path, text_path = export_result_debug(
                result=result,
                export_root=self._debug_export_dir,
                step_index=self._step_idx,
                final_only=self._final_only,
                panel_text=panel_text,
            )
        except Exception as exc:
            console.print(f"[yellow][WARN] Failed to export debug info: {exc}[/yellow]")
            return

        console.print(f"[green][INFO] Debug exported:[/green] {json_path}")
        console.print(f"[green][INFO] Text snapshot:[/green] {text_path}")

    def _build_info_panel_text(self, result: AnalysisResult, step: PipelineStep, step_count: int) -> str:
        info = result.debug_info

        relative_path = str(info.get("relative_path", result.source_path.name))
        group_name = str(info.get("group", "n/a"))
        status = str(info.get("status", "n/a"))

        pred_coins = int(info.get("predicted_coin_count", result.circle_count))
        true_coins = info.get("true_coin_count")
        coin_diff = info.get("coin_diff")

        pred_value = int(info.get("predicted_value_cents", info.get("total_cents", 0)))
        true_value = info.get("true_value_cents")
        value_diff = info.get("value_diff_cents")

        has_gt = bool(info.get("has_ground_truth", False))
        if has_gt and true_coins is not None and coin_diff is not None:
            coins_line = f"{pred_coins} vs {int(true_coins)} ({int(coin_diff):+d})"
        else:
            coins_line = f"{pred_coins} vs n/a"

        if has_gt and true_value is not None and value_diff is not None:
            values_line = (
                f"{format_total_cents(pred_value)} vs "
                f"{format_total_cents(int(true_value))} ({format_signed_cents(int(value_diff))})"
            )
        else:
            values_line = f"{format_total_cents(pred_value)} vs n/a"

        counts = info.get("value_counts", {})
        if isinstance(counts, dict):
            breakdown = ", ".join(
                f"{ValueEstimator.DENOM_TEXT[d]}:{int(counts.get(d, 0))}"
                for d in ValueEstimator.DENOM_PRINT_ORDER
                if int(counts.get(d, 0)) > 0
            )
        else:
            breakdown = ""
        breakdown = breakdown if breakdown else "none"

        split_stats = info.get("split_stats", [])
        split_by_id: dict[int, dict] = {}
        if isinstance(split_stats, list):
            for row in split_stats:
                if isinstance(row, dict) and "id" in row:
                    split_by_id[int(row["id"])] = row

        prediction_lines: list[str] = []
        predictions = info.get("value_predictions", {})
        if isinstance(predictions, dict):
            coin_ids: list[int] = []
            for raw_key in predictions.keys():
                try:
                    coin_ids.append(int(raw_key))
                except (TypeError, ValueError):
                    continue
            sorted_ids = sorted(coin_ids)
            for coin_id in sorted_ids[:14]:
                pred = predictions.get(coin_id, {})
                if pred == {} and coin_id in predictions:
                    pred = predictions[coin_id]
                elif pred == {} and str(coin_id) in predictions:
                    pred = predictions[str(coin_id)]
                if not isinstance(pred, dict):
                    continue
                marker = _coin_marker_token(coin_id)
                best_label = str(pred.get("best_label", "?"))
                best_prob = int(round(100.0 * float(pred.get("best_prob", 0.0))))
                family = str(pred.get("family", "unknown"))
                split_row = split_by_id.get(coin_id, {})
                visual_type = str(split_row.get("short_label", "?"))
                prediction_lines.append(
                    f"  {marker:>2} -> {best_label:<4} {best_prob:>3}%  fam={family:<7} vis={visual_type}"
                )
            if len(sorted_ids) > 14:
                prediction_lines.append(f"  ... ({len(sorted_ids) - 14} more)")

        coin_map_block = "\n".join(prediction_lines) if prediction_lines else "  none"

        hough = result.hough_params
        hough_line = (
            f"dp={hough.get('dp', 'n/a')}  minDist={hough.get('minDist', 'n/a')}\n"
            f"param1={hough.get('param1', 'n/a')}  param2={hough.get('param2', 'n/a')}\n"
            f"minR={hough.get('minRadius', 'n/a')}  maxR={hough.get('maxRadius', 'n/a')}"
        )

        text = (
            "DEBUG PANEL\n"
            "================================\n"
            f"file        : {relative_path}\n"
            f"group       : {group_name}\n"
            f"status      : {status}\n"
            f"backend     : {plt.get_backend()}\n"
            "\n"
            f"step        : {self._step_idx + 1}/{step_count}\n"
            f"step name   : {step.name}\n"
            f"image size  : {step.image.shape[1]} x {step.image.shape[0]}\n"
            "\n"
            f"coins (P/T) : {coins_line}\n"
            f"value (P/T) : {values_line}\n"
            "\n"
            f"value split : {breakdown}\n"
            "\n"
            "coin map (matches in-image markers)\n"
            f"{coin_map_block}\n"
            "\n"
            "hough params\n"
            f"{hough_line}\n"
            "\n"
            "normalization\n"
            "  hist_norm  : removed\n"
            "\n"
            "keys\n"
            "  left/right : previous/next image\n"
            "  up/down    : previous/next step\n"
            "  f          : toggle final-only/full\n"
            "  c or x     : export full debug to files\n"
            "  q or esc   : quit\n"
        )
        return text


OneFileViewer = DebugViewer
