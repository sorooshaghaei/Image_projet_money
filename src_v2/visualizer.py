from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from .analyzer import HybridCoinAnalyzer
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
        self._status_text = None
        self._image_slider = None
        self._hough_param2_slider = None
        self._hough_min_dist_slider = None
        self._contour_circ_slider = None
        self._watershed_fg_slider = None
        self._mode_radio = None

        self._cache: Dict[Tuple[int, str, int, int, float, float], AnalysisResult] = {}

    def show(self) -> None:
        if not self._paths:
            print("[WARN] no images available for visualizer")
            return

        self._setup_figure()
        self._render_current()
        plt.show()

    def _setup_figure(self) -> None:
        self._fig, self._axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.28, wspace=0.08, hspace=0.16)

        for ax in self._axes.flat:
            ax.axis("off")

        image_slider_ax = self._fig.add_axes([0.10, 0.18, 0.58, 0.03], facecolor="#ecf0f4")
        self._image_slider = Slider(
            image_slider_ax,
            "image",
            valmin=1,
            valmax=len(self._paths),
            valinit=1,
            valstep=1,
        )

        hough_param2_ax = self._fig.add_axes([0.10, 0.13, 0.28, 0.03], facecolor="#ecf0f4")
        self._hough_param2_slider = Slider(
            hough_param2_ax,
            "hough p2",
            valmin=8,
            valmax=80,
            valinit=float(self._analyzer.hough.param2),
            valstep=1,
        )

        hough_min_dist_ax = self._fig.add_axes([0.40, 0.13, 0.28, 0.03], facecolor="#ecf0f4")
        self._hough_min_dist_slider = Slider(
            hough_min_dist_ax,
            "hough minDist",
            valmin=8,
            valmax=180,
            valinit=float(self._analyzer.hough.min_dist),
            valstep=1,
        )

        contour_circ_ax = self._fig.add_axes([0.10, 0.08, 0.28, 0.03], facecolor="#ecf0f4")
        self._contour_circ_slider = Slider(
            contour_circ_ax,
            "contour circ",
            valmin=0.45,
            valmax=0.95,
            valinit=float(self._analyzer.contour.min_circularity),
            valstep=0.01,
        )

        watershed_fg_ax = self._fig.add_axes([0.40, 0.08, 0.28, 0.03], facecolor="#ecf0f4")
        self._watershed_fg_slider = Slider(
            watershed_fg_ax,
            "watershed fg",
            valmin=0.20,
            valmax=0.85,
            valinit=float(self._analyzer.watershed.fg_ratio),
            valstep=0.01,
        )

        mode_ax = self._fig.add_axes([0.73, 0.06, 0.24, 0.18], facecolor="#f4f6f8")
        self._mode_radio = RadioButtons(mode_ax, ("auto", "contours", "hough", "watershed", "hybrid"))
        self._mode_radio.set_active(("auto", "contours", "hough", "watershed", "hybrid").index(_radio_value(self._mode)))

        self._status_text = self._fig.text(
            0.05,
            0.01,
            "",
            family="monospace",
            fontsize=9,
        )

        self._image_slider.on_changed(self._on_controls_changed)
        self._hough_param2_slider.on_changed(self._on_controls_changed)
        self._hough_min_dist_slider.on_changed(self._on_controls_changed)
        self._contour_circ_slider.on_changed(self._on_controls_changed)
        self._watershed_fg_slider.on_changed(self._on_controls_changed)
        self._mode_radio.on_clicked(self._on_mode_changed)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_controls_changed(self, _value) -> None:
        self._idx = int(round(self._image_slider.val)) - 1
        self._render_current()

    def _on_mode_changed(self, label: str) -> None:
        self._mode = _canonical_mode(label)
        self._render_current()

    def _on_key(self, event) -> None:
        if event.key in ("left", "a"):
            self._idx = (self._idx - 1) % len(self._paths)
            self._image_slider.set_val(self._idx + 1)
        elif event.key in ("right", "d"):
            self._idx = (self._idx + 1) % len(self._paths)
            self._image_slider.set_val(self._idx + 1)

    def _render_current(self) -> None:
        result = self._current_result()

        self._axes[0, 0].clear()
        self._axes[0, 0].imshow(cv2.cvtColor(result.frames.overlay, cv2.COLOR_BGR2RGB))
        self._axes[0, 0].set_title("Detections")
        self._axes[0, 0].axis("off")

        self._axes[0, 1].clear()
        self._axes[0, 1].imshow(result.frames.gray, cmap="gray")
        self._axes[0, 1].set_title("Normalized Gray")
        self._axes[0, 1].axis("off")

        self._axes[1, 0].clear()
        self._axes[1, 0].imshow(result.frames.binary_mask, cmap="gray")
        self._axes[1, 0].set_title("Binary Mask (Contours/Watershed Input)")
        self._axes[1, 0].axis("off")

        self._axes[1, 1].clear()
        self._axes[1, 1].imshow(cv2.cvtColor(result.frames.watershed_markers, cv2.COLOR_BGR2RGB))
        self._axes[1, 1].set_title("Watershed Regions")
        self._axes[1, 1].axis("off")

        m = result.metrics
        self._fig.suptitle(
            (
                f"[{self._idx + 1}/{len(self._paths)}] {result.short_path} | "
                f"background={m.background_label} (score={m.texture_score:.3f}) | "
                f"method={result.selected_method} | coins={len(result.circles)}"
            ),
            x=0.01,
            ha="left",
            fontsize=10,
        )

        self._status_text.set_text(
            (
                f"border_cv={m.border_cv:.3f} | edge_density={m.edge_density:.3f} | "
                f"merge_score={m.contour_merge_score:.3f} | hough_overlap_pairs={m.hough_overlap_pairs} | "
                f"likely_overlap={'YES' if m.likely_overlap else 'NO'}\n"
                "controls: left/right or a/d = image navigation"
            )
        )

        self._fig.canvas.draw_idle()

    def _current_result(self) -> AnalysisResult:
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
            self._cache.pop(next(iter(self._cache)))
        return result


def _canonical_mode(mode: str) -> str:
    mode_text = str(mode or "auto").strip().lower()
    if mode_text in {"auto", "contours", "hough", "watershed", "hough+watershed"}:
        return mode_text
    if mode_text == "hybrid":
        return "hough+watershed"
    return "auto"


def _radio_value(mode: str) -> str:
    return "hybrid" if mode == "hough+watershed" else mode
