from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from .analyzer import HybridCoinAnalyzer
from .dataset import DatasetRepository
from .io_utils import short_path
from .models import AnalysisResult


class HybridVisualizer:
    """Interactive viewer for comparing algorithm modes and parameter settings."""

    def __init__(
        self,
        *,
        analyzer: HybridCoinAnalyzer,
        image_paths: Sequence[Path],
        short_root: Path,
        start_mode: str = "auto",
    ):
        self._analyzer = analyzer
        self._paths: List[Path] = list(image_paths)
        self._short_root = Path(short_root)
        self._mode = _canonical_mode(start_mode)
        self._idx = 0

        self._fig = None
        self._axes = None
        self._value_ax = None
        self._status_text = None
        self._image_slider = None
        self._hough_param2_slider = None
        self._hough_min_dist_slider = None
        self._contour_circ_slider = None
        self._watershed_fg_slider = None
        self._mode_radio = None
        self._value_text = None

        self._cache: Dict[Tuple[int, str, int, int, float, float], AnalysisResult] = {}
        self._truth_index = self._build_truth_index()

    def show(self) -> None:
        """Open the matplotlib UI."""
        if not self._paths:
            print("[WARN] no images available for visualizer")
            return

        self._setup_figure()
        self._render_current()
        plt.show()

    def _setup_figure(self) -> None:
        """Build dashboard layout (2x2 debug views + right analysis panel)."""
        self._fig = plt.figure(figsize=(14.2, 8.4), facecolor="#eef2f7")
        grid = self._fig.add_gridspec(
            nrows=2,
            ncols=3,
            left=0.035,
            right=0.985,
            top=0.90,
            bottom=0.25,
            width_ratios=[1.0, 1.0, 0.92],
            hspace=0.13,
            wspace=0.08,
        )
        self._axes = np.array(
            [
                [self._fig.add_subplot(grid[0, 0]), self._fig.add_subplot(grid[0, 1])],
                [self._fig.add_subplot(grid[1, 0]), self._fig.add_subplot(grid[1, 1])],
            ]
        )
        self._value_ax = self._fig.add_subplot(grid[:, 2])

        for ax in self._axes.flat:
            self._style_image_axis(ax, title="")
        self._style_value_axis()

        image_slider_ax = self._fig.add_axes([0.06, 0.17, 0.60, 0.026], facecolor="#dbe3ef")
        self._image_slider = Slider(
            image_slider_ax,
            "image",
            valmin=1,
            valmax=len(self._paths),
            valinit=1,
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider_widget(self._image_slider)

        # Keep a larger center gap so left value text never touches right labels on 13-inch screens.
        hough_param2_ax = self._fig.add_axes([0.06, 0.125, 0.25, 0.024], facecolor="#dbe3ef")
        self._hough_param2_slider = Slider(
            hough_param2_ax,
            "hough p2",
            valmin=8,
            valmax=80,
            valinit=float(self._analyzer.hough.param2),
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider_widget(self._hough_param2_slider)

        hough_min_dist_ax = self._fig.add_axes([0.41, 0.125, 0.25, 0.024], facecolor="#dbe3ef")
        self._hough_min_dist_slider = Slider(
            hough_min_dist_ax,
            "hough minDist",
            valmin=8,
            valmax=180,
            valinit=float(self._analyzer.hough.min_dist),
            valfmt="%0.0f",
            valstep=1,
        )
        self._style_slider_widget(self._hough_min_dist_slider)

        contour_circ_ax = self._fig.add_axes([0.06, 0.082, 0.25, 0.024], facecolor="#dbe3ef")
        self._contour_circ_slider = Slider(
            contour_circ_ax,
            "contour circ",
            valmin=0.45,
            valmax=0.95,
            valinit=float(self._analyzer.contour.min_circularity),
            valfmt="%0.2f",
            valstep=0.01,
        )
        self._style_slider_widget(self._contour_circ_slider)

        watershed_fg_ax = self._fig.add_axes([0.41, 0.082, 0.25, 0.024], facecolor="#dbe3ef")
        self._watershed_fg_slider = Slider(
            watershed_fg_ax,
            "watershed fg",
            valmin=0.20,
            valmax=0.85,
            valinit=float(self._analyzer.watershed.fg_ratio),
            valfmt="%0.2f",
            valstep=0.01,
        )
        self._style_slider_widget(self._watershed_fg_slider)

        mode_ax = self._fig.add_axes([0.72, 0.055, 0.24, 0.165], facecolor="#f8fafc")
        mode_items = ("auto", "fast", "contours", "hough", "watershed", "hybrid")
        self._mode_radio = RadioButtons(mode_ax, mode_items)
        self._mode_radio.set_active(mode_items.index(_radio_value(self._mode)))
        mode_ax.set_title("Mode", fontsize=10, loc="left", pad=8)

        self._status_text = self._fig.text(
            0.04,
            0.012,
            "",
            family="monospace",
            fontsize=9,
            color="#0f172a",
        )
        self._value_text = self._value_ax.text(
            0.03,
            0.965,
            "",
            transform=self._value_ax.transAxes,
            family="monospace",
            fontsize=8.8,
            ha="left",
            va="top",
            color="#0f172a",
        )

        self._image_slider.on_changed(self._on_controls_changed)
        self._hough_param2_slider.on_changed(self._on_controls_changed)
        self._hough_min_dist_slider.on_changed(self._on_controls_changed)
        self._contour_circ_slider.on_changed(self._on_controls_changed)
        self._watershed_fg_slider.on_changed(self._on_controls_changed)
        self._mode_radio.on_clicked(self._on_mode_changed)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_controls_changed(self, _value) -> None:
        # Slider value is 1-based for readability; convert to 0-based list index.
        self._idx = int(round(self._image_slider.val)) - 1
        self._render_current()

    def _on_mode_changed(self, label: str) -> None:
        self._mode = _canonical_mode(label)
        self._render_current()

    def _on_key(self, event) -> None:
        # Keyboard shortcuts for fast browsing while tuning parameters.
        if event.key in ("left", "a"):
            self._idx = (self._idx - 1) % len(self._paths)
            self._image_slider.set_val(self._idx + 1)
        elif event.key in ("right", "d"):
            self._idx = (self._idx + 1) % len(self._paths)
            self._image_slider.set_val(self._idx + 1)

    def _render_current(self) -> None:
        """Render images + analysis text for current controls."""
        result = self._current_result()
        true_count, true_value = self._lookup_truth(result.short_path)
        pred_count = int(len(result.circles))
        count_diff = (pred_count - true_count) if true_count is not None else None
        pred_value = float(result.estimated_value_eur)
        value_diff = (pred_value - true_value) if true_value is not None else None

        self._axes[0, 0].clear()
        self._style_image_axis(self._axes[0, 0], title="Detections")
        self._axes[0, 0].imshow(cv2.cvtColor(result.frames.overlay, cv2.COLOR_BGR2RGB))

        self._axes[0, 1].clear()
        self._style_image_axis(self._axes[0, 1], title="Normalized Gray")
        self._axes[0, 1].imshow(result.frames.gray, cmap="gray")

        self._axes[1, 0].clear()
        self._style_image_axis(self._axes[1, 0], title="Binary Mask")
        self._axes[1, 0].imshow(result.frames.binary_mask, cmap="gray")

        self._axes[1, 1].clear()
        self._style_image_axis(self._axes[1, 1], title="Watershed Regions")
        self._axes[1, 1].imshow(cv2.cvtColor(result.frames.watershed_markers, cv2.COLOR_BGR2RGB))

        m = result.metrics
        true_count_text = str(true_count) if true_count is not None else "-"
        count_diff_text = f"{count_diff:+d}" if count_diff is not None else "-"
        value_diff_text = f"{value_diff:+.2f}" if value_diff is not None else "-"

        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._paths)}] {result.short_path}",
            x=0.038,
            y=0.965,
            ha="left",
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
        )

        self._status_text.set_text(
            (
                f"border_cv={m.border_cv:.3f} | edge_density={m.edge_density:.3f} | "
                f"merge_score={m.contour_merge_score:.3f} | hough_overlap_pairs={m.hough_overlap_pairs} | "
                f"likely_overlap={'YES' if m.likely_overlap else 'NO'} | value={result.estimated_value_eur:.2f} EUR | "
                f"delta_coins={count_diff_text} | delta_value={value_diff_text} | "
                f"pred/true={pred_count}/{true_count_text}\n"
                "controls: left/right or a/d = image navigation"
            )
        )
        if self._value_text is not None:
            self._value_text.set_text(
                _format_coin_values_panel(
                    labels=result.coin_labels,
                    pred_count=pred_count,
                    true_count=true_count,
                    count_diff=count_diff,
                    pred_value=pred_value,
                    true_value=true_value,
                    value_diff=value_diff,
                    method=result.selected_method,
                    viewer_mode=self._mode,
                    background_label=m.background_label,
                    likely_overlap=m.likely_overlap,
                    ratio_fit_errors=result.ratio_fit_errors,
                )
            )

        self._fig.canvas.draw_idle()

    def _style_image_axis(self, ax, *, title: str) -> None:
        """Consistent styling for image panes (preserve aspect to avoid distortion)."""
        ax.set_title(title, fontsize=12, color="#0f172a", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#0b1220")
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.0)

    def _style_value_axis(self) -> None:
        """Styling for right-side text panel."""
        if self._value_ax is None:
            return
        self._value_ax.set_xticks([])
        self._value_ax.set_yticks([])
        self._value_ax.set_facecolor("#f8fafc")
        self._value_ax.set_title("Analysis Panel", fontsize=12, loc="left", pad=10, color="#0f172a")
        for spine in self._value_ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(1.0)

    @staticmethod
    def _style_slider_widget(slider: Slider) -> None:
        """Keep slider labels/value text inside their own lane to avoid overlap."""
        slider.label.set_color("#0f172a")
        slider.label.set_fontsize(11)
        slider.label.set_horizontalalignment("left")
        slider.label.set_position((0.0, 0.5))
        slider.valtext.set_color("#0f172a")
        slider.valtext.set_fontsize(11)
        slider.valtext.set_horizontalalignment("right")
        slider.valtext.set_position((0.995, 0.5))

    def _current_result(self) -> AnalysisResult:
        """Compute or fetch cached analysis for current control state."""
        key = (
            self._idx,
            self._mode,
            int(round(self._hough_param2_slider.val)),
            int(round(self._hough_min_dist_slider.val)),
            round(float(self._contour_circ_slider.val), 2),
            round(float(self._watershed_fg_slider.val), 2),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        image_path = self._paths[self._idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read image: {image_path}")

        result = self._analyzer.analyze(
            image,
            source_path=image_path.as_posix(),
            short_path=short_path(image_path, self._short_root),
            mode=self._mode,
            overrides={
                "hough_param2": int(round(self._hough_param2_slider.val)),
                "hough_min_dist": int(round(self._hough_min_dist_slider.val)),
                "contour_min_circularity": float(self._contour_circ_slider.val),
                "watershed_fg_ratio": float(self._watershed_fg_slider.val),
            },
        )
        self._cache[key] = result
        if len(self._cache) > 180:
            # Keep cache bounded to avoid memory growth during long sessions.
            self._cache.pop(next(iter(self._cache)))
        return result

    @staticmethod
    def _normalize_group(group: str) -> str:
        text = str(group).strip().lower()
        if text.startswith("grp"):
            return "gp" + text[3:]
        return text

    @staticmethod
    def _normalize_filename(name: str) -> str:
        return str(name).strip()

    def _build_truth_index(self) -> Dict[Tuple[str, str], Tuple[int, float]]:
        """Dataset ground-truth index for per-image pred/true comparison in UI."""
        index: Dict[Tuple[str, str], Tuple[int, float]] = {}
        for row in DatasetRepository.DATA_ROWS:
            filename = self._normalize_filename(str(row[0]))
            true_count = int(row[1])
            true_value = float(row[2]) if row[2] is not None else None
            group = self._normalize_group(str(row[3]))
            index[(group, filename)] = (true_count, true_value)
        return index

    def _lookup_truth(self, short_path_text: str) -> Tuple[int | None, float | None]:
        path = Path(short_path_text)
        if len(path.parts) < 2:
            return None, None
        group = self._normalize_group(path.parts[0])
        filename = self._normalize_filename(path.parts[-1])
        return self._truth_index.get((group, filename), (None, None))


def _canonical_mode(mode: str) -> str:
    mode_text = str(mode or "auto").strip().lower()
    if mode_text in {"auto", "fast", "contours", "hough", "watershed", "hough+watershed"}:
        return mode_text
    if mode_text == "hybrid":
        return "hough+watershed"
    return "auto"


def _radio_value(mode: str) -> str:
    return "hybrid" if mode == "hough+watershed" else mode


def _format_coin_values_panel(
    *,
    labels: Sequence[Optional[int]],
    pred_count: int,
    true_count: Optional[int],
    count_diff: Optional[int],
    pred_value: float,
    true_value: Optional[float],
    value_diff: Optional[float],
    method: str,
    viewer_mode: str,
    background_label: str,
    likely_overlap: bool,
    ratio_fit_errors: Sequence[Optional[float]],
) -> str:
    """Format right-side panel text with count/value differences and denomination details."""
    true_count_text = str(true_count) if true_count is not None else "-"
    count_diff_text = f"{count_diff:+d}" if count_diff is not None else "-"
    true_value_text = f"{true_value:.2f}" if true_value is not None else "-"
    value_diff_text = f"{value_diff:+.2f}" if value_diff is not None else "-"
    if not labels:
        return (
            "Scene\n"
            f"  background: {background_label}\n"
            f"  mode: {viewer_mode}\n"
            f"  method: {method}\n"
            f"  overlap: {'YES' if likely_overlap else 'NO'}\n\n"
            "Counts\n"
            f"  pred/true: {pred_count}/{true_count_text}\n"
            f"  diff: {count_diff_text}\n\n"
            "Value (EUR)\n"
            f"  predicted(calibrated): {pred_value:.2f}\n"
            f"  ground_truth: {true_value_text}\n"
            f"  diff(pred-true): {value_diff_text}\n\n"
            "Coin Values\n"
            "  (no detections)"
        )

    counts: Dict[int, int] = {}
    unknown_count = 0
    values: List[str] = []
    known_sum_cents = 0

    for idx, label in enumerate(labels, start=1):
        if label is None:
            unknown_count += 1
            values.append(f"{idx}:?")
            continue
        den = int(label)
        known_sum_cents += den
        counts[den] = counts.get(den, 0) + 1
        values.append(f"{idx}:{den}c")

    known_count = len(labels) - unknown_count
    hist_parts = [f"{den}c x{counts[den]}" for den in sorted(counts)]
    if unknown_count > 0:
        hist_parts.append(f"unknown x{unknown_count}")
    hist_text = ", ".join(hist_parts) if hist_parts else "unknown only"
    hist_wrapped = fill(hist_text, width=44, subsequent_indent="    ")

    preview_rows: List[str] = []
    per_row = 6
    max_tokens = 18
    for i in range(0, min(len(values), max_tokens), per_row):
        preview_rows.append(" ".join(values[i : i + per_row]))
    if len(values) > max_tokens:
        preview_rows.append(f"... (+{len(values) - max_tokens} more)")
    preview = "\n  ".join(preview_rows)

    fit_values = [float(err) for err in ratio_fit_errors if err is not None]
    mean_fit_err = float(np.mean(fit_values)) if fit_values else None
    mean_fit_text = f"{mean_fit_err:.3f}" if mean_fit_err is not None else "-"
    raw_labeled_sum = known_sum_cents / 100.0
    calibration_gap = pred_value - raw_labeled_sum

    return (
        "Scene\n"
        f"  background: {background_label}\n"
        f"  mode: {viewer_mode}\n"
        f"  method: {method}\n"
        f"  overlap: {'YES' if likely_overlap else 'NO'}\n\n"
        "Counts\n"
        f"  pred/true: {pred_count}/{true_count_text}\n"
        f"  diff: {count_diff_text}\n\n"
        "Value (EUR)\n"
        f"  predicted(calibrated): {pred_value:.2f}\n"
        f"  ground_truth: {true_value_text}\n"
        f"  diff(pred-true): {value_diff_text}\n"
        f"  raw_labeled_sum: {raw_labeled_sum:.2f}\n"
        f"  calibration_gap: {calibration_gap:+.2f}\n\n"
        "Coin Values\n"
        f"  labeled/unknown: {known_count}/{unknown_count}\n"
        f"  hist: {hist_wrapped}\n"
        f"  fit_err_mean: {mean_fit_text}\n"
        f"  per coin:\n  {preview}"
    )
