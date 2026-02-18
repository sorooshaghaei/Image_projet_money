from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .models import PipelineResult
from .processor import CoinProcessor


class PipelineVisualizer:
    """Saves static montage images of pipeline stages for one processed input."""

    def save_pipeline_steps(self, result: PipelineResult, out_dir: str, cols: int = 4) -> Optional[str]:
        """Render and persist a grid of debug steps for offline inspection."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        steps = result.steps
        n = len(steps)
        if n == 0:
            return None

        rows = ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
        fig.suptitle(
            (
                f"{result.source_filename} | pred={result.coin_count} | "
                f"labeled={result.labeled_coin_count} | value={result.estimated_value_eur:.2f} EUR | "
                f"inverted={result.is_inverted}"
            ),
            fontsize=14,
        )

        axes_matrix = self._normalize_axes_shape(axes, rows, cols)

        for i, step in enumerate(steps):
            r, c = divmod(i, cols)
            ax = axes_matrix[r, c]
            img = step.image

            if step.cmap == "rgb":
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")

            ax.set_title(step.name, fontsize=11)
            ax.axis("off")

        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes_matrix[r, c].axis("off")

        plt.tight_layout()
        out_path = Path(out_dir) / f"{Path(result.source_filename).stem}_pipeline.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    def _normalize_axes_shape(self, axes, rows: int, cols: int) -> np.ndarray:
        """Normalize matplotlib axis return type to a 2D matrix for uniform indexing."""
        if rows == 1 and cols == 1:
            return np.array([[axes]])
        if rows == 1:
            return np.array([axes])
        if cols == 1:
            return np.array([[ax] for ax in axes])
        return axes


class HoughTuningBrowser:
    """Interactive UI for tuning Hough parameters and inspecting pipeline behavior."""

    def __init__(self, processor: CoinProcessor, results: List[PipelineResult], cols: int = 4):
        self._processor = processor
        self._results = results
        self._cols = cols

        self._idx = 0
        self._locked = False
        self._pending = {"timer": None}
        self._suppress_updates = False

        self._originals: List[np.ndarray] = [r.steps[0].image.copy() for r in results]
        self._filenames: List[str] = [r.source_filename for r in results]

        self._cfg = processor._cfg

        self._rows = ceil(5 / cols)
        self._fig = None
        self._axes = None

        self._s_dp = None
        self._s_minDist = None
        self._s_p1 = None
        self._s_p2 = None
        self._s_minR = None
        self._s_maxR = None
        self._s_image = None
        self._btn_lock = None
        self._btn_prev = None
        self._btn_next = None
        self._btn_fast = None
        self._status_text = None
        self._syncing_index_slider = False
        self._axis_to_index: Dict[object, int] = {}
        self._zoom_state: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        self._cache: Dict[Tuple[int, int, float, int, int, int, int, int], PipelineResult] = {}
        self._cache_limit = 256
        self._debounce_ms = 80
        self._last_compute_ms = 0.0
        self._fast_mode = False

    def show(self):
        """Open the interactive tuning window."""
        if not self._results:
            print("[WARN] No results to display.")
            return

        self._setup_figure()
        self._bind_events()
        self._compute_live()
        plt.show()

    def _setup_figure(self):
        """Build UI layout: preview panels, sliders, and control buttons."""
        fig_w = min(14.0, max(10.5, 3.4 * self._cols))
        fig_h = min(8.6, max(6.8, 2.35 * self._rows + 2.8))
        self._fig = plt.figure(figsize=(fig_w, fig_h))
        self._fig.set_facecolor("#f3f4f6")
        gs = self._fig.add_gridspec(
            self._rows,
            self._cols,
            top=0.86,
            bottom=0.32,
            left=0.03,
            right=0.99,
            wspace=0.05,
            hspace=0.15,
        )

        self._axes = np.empty((self._rows, self._cols), dtype=object)
        for r in range(self._rows):
            for c in range(self._cols):
                self._axes[r, c] = self._fig.add_subplot(gs[r, c])
                self._axes[r, c].set_facecolor("#ffffff")
                self._axis_to_index[self._axes[r, c]] = r * self._cols + c

        ax_dp = self._fig.add_axes([0.06, 0.27, 0.40, 0.03], facecolor="#e8edf5")
        ax_minDist = self._fig.add_axes([0.06, 0.23, 0.40, 0.03], facecolor="#e8edf5")
        ax_p1 = self._fig.add_axes([0.06, 0.19, 0.40, 0.03], facecolor="#e8edf5")
        ax_p2 = self._fig.add_axes([0.06, 0.15, 0.40, 0.03], facecolor="#e8edf5")
        ax_minR = self._fig.add_axes([0.54, 0.27, 0.40, 0.03], facecolor="#e8edf5")
        ax_maxR = self._fig.add_axes([0.54, 0.23, 0.40, 0.03], facecolor="#e8edf5")

        self._s_dp = Slider(ax_dp, "dp", 0.5, 3.0, valinit=float(self._cfg.HOUGH_DP), valstep=0.1)
        self._s_minDist = Slider(ax_minDist, "minDist", 1, 300, valinit=int(self._cfg.HOUGH_MIN_DIST), valstep=1)
        self._s_p1 = Slider(ax_p1, "param1", 1, 300, valinit=int(self._cfg.HOUGH_PARAM1), valstep=1)
        self._s_p2 = Slider(ax_p2, "param2", 1, 300, valinit=int(self._cfg.HOUGH_PARAM2), valstep=1)
        self._s_minR = Slider(ax_minR, "minR", 0, 300, valinit=int(self._cfg.HOUGH_MIN_RADIUS), valstep=1)
        self._s_maxR = Slider(ax_maxR, "maxR", 1, 500, valinit=int(self._cfg.HOUGH_MAX_RADIUS), valstep=1)
        for slider in (self._s_dp, self._s_minDist, self._s_p1, self._s_p2, self._s_minR, self._s_maxR):
            self._style_slider(slider)

        if len(self._originals) > 1:
            ax_image = self._fig.add_axes([0.54, 0.19, 0.40, 0.03], facecolor="#e8edf5")
            self._s_image = Slider(
                ax_image,
                "image",
                1,
                len(self._originals),
                valinit=1,
                valstep=1,
            )
            self._style_slider(self._s_image)

        ax_prev = self._fig.add_axes([0.06, 0.08, 0.12, 0.055], facecolor="#e8edf5")
        ax_next = self._fig.add_axes([0.19, 0.08, 0.12, 0.055], facecolor="#e8edf5")
        ax_reset = self._fig.add_axes([0.54, 0.08, 0.12, 0.055], facecolor="#e8edf5")
        ax_lock = self._fig.add_axes([0.68, 0.08, 0.12, 0.055], facecolor="#e8edf5")
        ax_fast = self._fig.add_axes([0.82, 0.08, 0.12, 0.055], facecolor="#e8edf5")

        self._btn_prev = Button(ax_prev, "Prev")
        self._btn_next = Button(ax_next, "Next")
        btn_reset = Button(ax_reset, "Reset")
        self._btn_lock = Button(ax_lock, "Lock: OFF")
        self._btn_fast = Button(ax_fast, "Fast: OFF")
        for btn in (self._btn_prev, self._btn_next, btn_reset, self._btn_lock, self._btn_fast):
            btn.label.set_fontsize(10)

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_next.on_clicked(self._on_next)
        btn_reset.on_clicked(self._on_reset)
        self._btn_lock.on_clicked(self._on_lock)
        self._btn_fast.on_clicked(self._on_fast_mode)

        self._status_text = self._fig.text(
            0.06,
            0.02,
            "",
            fontsize=10,
            family="monospace",
            color="#1f2937",
            va="bottom",
        )
        self._try_fit_to_screen()

    def _bind_events(self):
        """Bind keyboard/mouse/slider events to update callbacks."""
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._s_dp.on_changed(self._schedule_update)
        self._s_minDist.on_changed(self._schedule_update)
        self._s_p1.on_changed(self._schedule_update)
        self._s_p2.on_changed(self._schedule_update)
        self._s_minR.on_changed(self._schedule_update)
        self._s_maxR.on_changed(self._schedule_update)
        if self._s_image is not None:
            self._s_image.on_changed(self._on_image_slider)

    def _render(self, res: PipelineResult):
        """Redraw all panels using the current image and current parameter set."""
        dp = self._s_dp.val
        md = int(self._s_minDist.val)
        p1 = int(self._s_p1.val)
        p2 = int(self._s_p2.val)
        mn = int(self._s_minR.val)
        mx = int(self._s_maxR.val)
        fname = self._filenames[self._idx]

        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._originals)}] {fname} | pred={res.coin_count} | labeled={res.labeled_coin_count} "
            f"| value={res.estimated_value_eur:.2f} EUR | inv={res.is_inverted} | "
            f"dp={dp:.1f} minDist={md} p1={p1} p2={p2} minR={mn} maxR={mx} | "
            f"mode={'fast' if self._fast_mode else 'full'} | t={self._last_compute_ms:.0f}ms | "
            f"(a/d or arrows nav, f fast, l lock, r reset, q/esc quit)",
            fontsize=12,
        )

        for j in range(self._rows * self._cols):
            rr, cc = divmod(j, self._cols)
            ax = self._axes[rr, cc]
            ax.clear()
            ax.axis("off")

            if j >= len(res.steps):
                continue

            step = res.steps[j]
            if step.cmap == "rgb":
                ax.imshow(cv2.cvtColor(step.image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(step.image, cmap="gray")
            ax.set_title(step.name, fontsize=11, color="#111827")
            self._apply_zoom_state(ax=ax, axis_index=j, image=step.image)

        if self._status_text is not None:
            self._status_text.set_text(
                (
                    f"Image {self._idx + 1}/{len(self._originals)}   "
                    f"Detected {res.coin_count}   Labeled {res.labeled_coin_count}   "
                    f"Value {res.estimated_value_eur:.2f} EUR   "
                    f"Compute {self._last_compute_ms:.0f} ms\n"
                    f"Mode: {'Fast preview (detection only)' if self._fast_mode else 'Full pipeline (detection + value)'}   "
                    f"Zoom: mouse wheel | reset zoom: right click"
                )
            )
        self._fig.canvas.draw_idle()

    def _clamp_slider_pair(self):
        """Ensure radius slider bounds stay valid (maxR must stay > minR)."""
        mn = int(self._s_minR.val)
        mx = int(self._s_maxR.val)
        if mx <= mn:
            self._suppress_updates = True
            self._s_maxR.set_val(mn + 1)
            self._suppress_updates = False

    def _current_params_key(self) -> Tuple[int, int, float, int, int, int, int, int]:
        """Cache key for avoiding redundant recomputation."""
        return (
            self._idx,
            1 if self._fast_mode else 0,
            round(float(self._s_dp.val), 1),
            int(self._s_minDist.val),
            int(self._s_p1.val),
            int(self._s_p2.val),
            int(self._s_minR.val),
            int(self._s_maxR.val),
        )

    def _sync_image_slider(self):
        """Keep image slider and current index synchronized."""
        if self._s_image is None:
            return
        self._syncing_index_slider = True
        self._s_image.set_val(self._idx + 1)
        self._syncing_index_slider = False

    def _compute_live(self):
        """Run live detection/classification for current controls and cache result."""
        self._clamp_slider_pair()

        img = self._originals[self._idx]
        fname = self._filenames[self._idx]
        key = self._current_params_key()

        start = perf_counter()
        res = self._cache.get(key)
        if res is None:
            res = self._processor.detect_with_params(
                img_bgr_resized=img,
                dp=float(self._s_dp.val),
                minDist=int(self._s_minDist.val),
                param1=int(self._s_p1.val),
                param2=int(self._s_p2.val),
                minRadius=int(self._s_minR.val),
                maxRadius=int(self._s_maxR.val),
                filename=fname,
                classify=not self._fast_mode,
            )
            # Small bounded cache keeps UI responsive when toggling nearby settings.
            self._cache[key] = res
            if len(self._cache) > self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
        self._last_compute_ms = (perf_counter() - start) * 1000.0
        self._render(res)

    def _schedule_update(self, _=None):
        """Debounce expensive recomputation while sliders are moving."""
        if self._locked or self._suppress_updates:
            return

        timer = self._pending["timer"]
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

        timer = self._fig.canvas.new_timer(interval=self._debounce_ms)
        timer.add_callback(self._compute_live)
        self._pending["timer"] = timer
        timer.start()

    def _on_key(self, event):
        """Keyboard shortcuts for fast navigation and mode toggles."""
        if event.key in ("d", "right"):
            self._idx = (self._idx + 1) % len(self._originals)
            self._sync_image_slider()
            self._compute_live()
        elif event.key in ("a", "left"):
            self._idx = (self._idx - 1) % len(self._originals)
            self._sync_image_slider()
            self._compute_live()
        elif event.key == "f":
            self._on_fast_mode(None)
        elif event.key == "l":
            self._on_lock(None)
        elif event.key == "r":
            self._on_reset(None)
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _on_reset(self, _event):
        """Restore default Hough parameters and clear zoom state."""
        self._suppress_updates = True
        self._s_dp.set_val(float(self._cfg.HOUGH_DP))
        self._s_minDist.set_val(int(self._cfg.HOUGH_MIN_DIST))
        self._s_p1.set_val(int(self._cfg.HOUGH_PARAM1))
        self._s_p2.set_val(int(self._cfg.HOUGH_PARAM2))
        self._s_minR.set_val(int(self._cfg.HOUGH_MIN_RADIUS))
        self._s_maxR.set_val(int(self._cfg.HOUGH_MAX_RADIUS))
        self._suppress_updates = False
        self._zoom_state.clear()
        self._compute_live()

    def _on_lock(self, _event):
        """Freeze/unfreeze auto-updates while user edits controls."""
        self._locked = not self._locked
        self._btn_lock.label.set_text("Lock: ON" if self._locked else "Lock: OFF")
        self._fig.canvas.draw_idle()
        if not self._locked:
            self._compute_live()

    def _on_fast_mode(self, _event):
        """Toggle fast mode (detection only) vs full mode (detection + classification)."""
        self._fast_mode = not self._fast_mode
        self._btn_fast.label.set_text("Fast: ON" if self._fast_mode else "Fast: OFF")
        self._fig.canvas.draw_idle()
        self._compute_live()

    def _on_prev(self, _event):
        """Move to previous image in browser."""
        self._idx = (self._idx - 1) % len(self._originals)
        self._sync_image_slider()
        self._compute_live()

    def _on_next(self, _event):
        """Move to next image in browser."""
        self._idx = (self._idx + 1) % len(self._originals)
        self._sync_image_slider()
        self._compute_live()

    def _on_image_slider(self, value):
        """Jump to the image index selected via slider."""
        if self._syncing_index_slider:
            return
        new_idx = int(round(value)) - 1
        self._idx = int(np.clip(new_idx, 0, len(self._originals) - 1))
        self._compute_live()

    def _apply_zoom_state(self, ax, axis_index: int, image: np.ndarray):
        """Reapply persisted zoom box when panels rerender."""
        if axis_index not in self._zoom_state:
            return
        xlim, ylim = self._zoom_state[axis_index]
        h, w = image.shape[:2]
        x0 = float(np.clip(min(xlim), 0, max(0, w - 1)))
        x1 = float(np.clip(max(xlim), 1, max(1, w)))
        y0 = float(np.clip(min(ylim), 0, max(0, h - 1)))
        y1 = float(np.clip(max(ylim), 1, max(1, h)))
        ax.set_xlim((x0, x1))
        ax.set_ylim((y1, y0))

    def _on_scroll_zoom(self, event):
        """Mouse wheel zoom centered at cursor position."""
        if event.inaxes is None:
            return
        axis_index = self._axis_to_index.get(event.inaxes)
        if axis_index is None:
            return
        ax = event.inaxes
        if not ax.images:
            return
        if event.xdata is None or event.ydata is None:
            return

        img = ax.images[0].get_array()
        h, w = img.shape[:2]
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xmin, cur_xmax = min(cur_xlim), max(cur_xlim)
        cur_ymin, cur_ymax = min(cur_ylim), max(cur_ylim)

        zoom_factor = 1.2
        scale = 1.0 / zoom_factor if event.button == "up" else zoom_factor
        new_w = max(8.0, (cur_xmax - cur_xmin) * scale)
        new_h = max(8.0, (cur_ymax - cur_ymin) * scale)

        cx = float(np.clip(event.xdata, 0, w))
        cy = float(np.clip(event.ydata, 0, h))
        x0 = float(np.clip(cx - new_w / 2.0, 0, max(0, w - new_w)))
        y0 = float(np.clip(cy - new_h / 2.0, 0, max(0, h - new_h)))
        x1 = x0 + new_w
        y1 = y0 + new_h

        ax.set_xlim((x0, x1))
        ax.set_ylim((y1, y0))
        self._zoom_state[axis_index] = ((x0, x1), (y1, y0))
        self._fig.canvas.draw_idle()

    def _on_click(self, event):
        """Right-click resets zoom on the clicked panel."""
        if event.inaxes is None:
            return
        axis_index = self._axis_to_index.get(event.inaxes)
        if axis_index is None:
            return
        if event.button != 3:
            return

        ax = event.inaxes
        if not ax.images:
            return
        img = ax.images[0].get_array()
        h, w = img.shape[:2]
        ax.set_xlim((0, w))
        ax.set_ylim((h, 0))
        self._zoom_state.pop(axis_index, None)
        self._fig.canvas.draw_idle()

    def _try_fit_to_screen(self):
        """Shrink figure when needed so controls remain visible on smaller displays."""
        try:
            manager = self._fig.canvas.manager
            if not hasattr(manager, "window"):
                return
            window = manager.window
            if not hasattr(window, "winfo_screenwidth"):
                return
            sw = int(window.winfo_screenwidth())
            sh = int(window.winfo_screenheight())
            dpi = float(self._fig.dpi)
            max_w = max(8.0, (sw - 120) / dpi)
            max_h = max(5.5, (sh - 180) / dpi)
            cur_w, cur_h = self._fig.get_size_inches()
            scale = min(1.0, max_w / cur_w, max_h / cur_h)
            if scale < 0.999:
                self._fig.set_size_inches(cur_w * scale, cur_h * scale, forward=True)
        except Exception:
            return

    def _style_slider(self, slider: Slider):
        """Apply a consistent visual style to sliders."""
        slider.poly.set_facecolor("#2a6fdb")
        if hasattr(slider, "vline"):
            slider.vline.set_color("#1e3a8a")
        if hasattr(slider, "track"):
            slider.track.set_color("#d7dce5")
        slider.label.set_fontsize(10)
