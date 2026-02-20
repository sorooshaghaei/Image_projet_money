from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider

from .models import PipelineResult
from .processor import CoinProcessor


def _source_path_text(source: str) -> str:
    return Path(source).as_posix()


def _truncate_middle(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    keep_left = max(10, (width // 2) - 2)
    keep_right = max(10, width - keep_left - 3)
    return f"{text[:keep_left]}...{text[-keep_right:]}"


class PipelineVisualizer:
    """Saves static montage images of pipeline stages for one processed input."""

    def save_pipeline_steps(self, result: PipelineResult, out_dir: str) -> Optional[str]:
        """Render and persist a grid of debug steps for offline inspection."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        steps = result.steps
        if not steps:
            return None

        cols = 4
        rows = ceil(len(steps) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
        source_path = _source_path_text(result.source_filename)
        short_path = _truncate_middle(source_path, 120)
        fig.suptitle(
            (
                f"image: {short_path}\n"
                f"label={result.labeled_coin_count}  pred={result.coin_count}  "
                f"value={result.estimated_value_eur:.2f} EUR  inverted={result.is_inverted}"
            ),
            fontsize=11,
            x=0.01,
            ha="left",
            y=0.98,
        )

        axes_matrix = np.array(axes, dtype=object).reshape(rows, cols)
        for idx, ax in enumerate(axes_matrix.flat):
            ax.axis("off")
            if idx >= len(steps):
                continue
            step = steps[idx]
            if step.cmap == "rgb":
                ax.imshow(cv2.cvtColor(step.image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(step.image, cmap="gray")
            ax.set_title(step.name, fontsize=11)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        out_path = Path(out_dir) / f"{Path(result.source_filename).stem}_pipeline.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)


class HoughTuningBrowser:
    """Interactive UI for tuning Hough parameters and inspecting pipeline behavior."""

    _GRID_COLS = 3
    _GRID_ROWS = 2
    _SLIDER_FACE = "#e8edf5"
    _PARAM_SPECS = (
        ("dp", "dp", (0.06, 0.29, 0.40, 0.03), 0.5, 3.0, "HOUGH_DP", 0.1, float),
        ("minDist", "minDist", (0.06, 0.25, 0.40, 0.03), 1, 300, "HOUGH_MIN_DIST", 1, int),
        ("param1", "param1", (0.06, 0.21, 0.40, 0.03), 1, 300, "HOUGH_PARAM1", 1, int),
        ("param2", "param2", (0.06, 0.17, 0.40, 0.03), 1, 300, "HOUGH_PARAM2", 1, int),
        ("maskOffset", "maskOffset", (0.06, 0.13, 0.40, 0.03), -30, 30, "MASK_OTSU_OFFSET", 1, int),
        ("minComp", "minComp", (0.06, 0.09, 0.40, 0.03), 50, 1800, "MASK_MIN_COMPONENT_AREA", 10, int),
        ("minR", "minR", (0.54, 0.29, 0.40, 0.03), 0, 300, "HOUGH_MIN_RADIUS", 1, int),
        ("maxR", "maxR", (0.54, 0.25, 0.40, 0.03), 1, 500, "HOUGH_MAX_RADIUS", 1, int),
        ("fgRatio", "fgRatio", (0.54, 0.21, 0.40, 0.03), 0.20, 0.85, "WATERSHED_DT_FG_RATIO", 0.01, float),
        ("minCirc", "minCirc", (0.54, 0.17, 0.40, 0.03), 0.40, 0.95, "GEOM_MIN_CIRCULARITY", 0.01, float),
        ("hybGap", "hybGap", (0.54, 0.13, 0.40, 0.03), 5, 40, "HYBRID_MASK_OVER_HOUGH_MIN_GAP", 1, int),
        ("hybMinH", "hybMinH", (0.54, 0.09, 0.40, 0.03), 5, 80, "HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT", 1, int),
    )
    _NAV_KEYS = {"d": 1, "right": 1, "a": -1, "left": -1}

    def __init__(self, processor: CoinProcessor, results: List[PipelineResult]):
        self._processor = processor
        self._results = results
        self._cfg = processor._cfg

        self._idx = 0
        self._locked = False
        self._fast_mode = False
        self._suppress_updates = False
        self._syncing_index_slider = False

        self._originals: List[np.ndarray] = [r.steps[0].image.copy() for r in results]
        self._filenames: List[str] = [r.source_filename for r in results]

        self._fig = None
        self._axes = None
        self._status_text = None
        self._sliders: Dict[str, Slider] = {}
        self._image_slider: Optional[Slider] = None
        self._buttons: Dict[str, Button] = {}

        self._axis_to_index: Dict[object, int] = {}
        # Per-panel zoom box as normalized coordinates (x0, x1, y0, y1) in [0, 1].
        self._zoom_state: Dict[int, Tuple[float, float, float, float]] = {}

        self._cache: Dict[Tuple[object, ...], PipelineResult] = {}
        self._cache_limit = 256
        self._debounce_ms = 80
        self._pending_timer = None
        self._last_compute_ms = 0.0

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
        fig_w = min(15.0, max(12.0, 4.2 * self._GRID_COLS))
        fig_h = min(9.2, max(7.4, 2.55 * self._GRID_ROWS + 2.9))
        self._fig = plt.figure(figsize=(fig_w, fig_h))
        self._fig.set_facecolor("#f3f4f6")

        gs = self._fig.add_gridspec(
            self._GRID_ROWS,
            self._GRID_COLS,
            top=0.81,
            bottom=0.33,
            left=0.03,
            right=0.99,
            wspace=0.03,
            hspace=0.12,
        )

        self._axes = np.empty((self._GRID_ROWS, self._GRID_COLS), dtype=object)
        for idx in range(self._GRID_ROWS * self._GRID_COLS):
            r, c = divmod(idx, self._GRID_COLS)
            ax = self._fig.add_subplot(gs[r, c])
            ax.set_facecolor("#ffffff")
            self._axes[r, c] = ax
            self._axis_to_index[ax] = idx

        self._build_sliders()
        self._build_buttons()
        self._status_text = self._fig.text(
            0.06,
            0.008,
            "",
            fontsize=10,
            family="monospace",
            color="#1f2937",
            va="bottom",
        )
        self._try_fit_to_screen()

    def _build_sliders(self):
        for key, label, rect, valmin, valmax, cfg_attr, valstep, caster in self._PARAM_SPECS:
            init = caster(getattr(self._cfg, cfg_attr))
            self._sliders[key] = self._create_slider(
                label=label,
                rect=rect,
                valmin=valmin,
                valmax=valmax,
                valinit=init,
                valstep=valstep,
            )

        if len(self._originals) > 1:
            self._image_slider = self._create_slider(
                label="image",
                rect=(0.54, 0.05, 0.40, 0.03),
                valmin=1,
                valmax=len(self._originals),
                valinit=1,
                valstep=1,
            )

    def _build_buttons(self):
        button_specs = (
            ("prev", "Prev", (0.06, 0.02, 0.12, 0.045), lambda _e: self._shift_image(-1)),
            ("next", "Next", (0.19, 0.02, 0.12, 0.045), lambda _e: self._shift_image(1)),
            ("reset", "Reset", (0.54, 0.02, 0.12, 0.045), self._on_reset),
            ("lock", "Lock: OFF", (0.68, 0.02, 0.12, 0.045), lambda _e: self._toggle("lock")),
            ("fast", "Fast: OFF", (0.82, 0.02, 0.12, 0.045), lambda _e: self._toggle("fast")),
        )
        for key, label, rect, callback in button_specs:
            ax = self._fig.add_axes(rect, facecolor=self._SLIDER_FACE)
            btn = Button(ax, label)
            btn.label.set_fontsize(10)
            btn.on_clicked(callback)
            self._buttons[key] = btn

    def _create_slider(self, label: str, rect, valmin: float, valmax: float, valinit: float, valstep: float) -> Slider:
        ax = self._fig.add_axes(rect, facecolor=self._SLIDER_FACE)
        slider = Slider(ax, label, valmin, valmax, valinit=valinit, valstep=valstep)
        self._style_slider(slider)
        return slider

    def _bind_events(self):
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        for slider in self._sliders.values():
            slider.on_changed(self._schedule_update)
        if self._image_slider is not None:
            self._image_slider.on_changed(self._on_image_slider)

    def _params(self) -> Dict[str, float]:
        sliders = self._sliders
        return {
            "dp": float(sliders["dp"].val),
            "minDist": int(sliders["minDist"].val),
            "param1": int(sliders["param1"].val),
            "param2": int(sliders["param2"].val),
            "minRadius": int(sliders["minR"].val),
            "maxRadius": int(sliders["maxR"].val),
        }

    def _config_overrides(self) -> Dict[str, float]:
        sliders = self._sliders
        return {
            "MASK_OTSU_OFFSET": int(sliders["maskOffset"].val),
            "MASK_MIN_COMPONENT_AREA": int(sliders["minComp"].val),
            "WATERSHED_DT_FG_RATIO": float(sliders["fgRatio"].val),
            "GEOM_MIN_CIRCULARITY": float(sliders["minCirc"].val),
            "HYBRID_MASK_OVER_HOUGH_MIN_GAP": int(sliders["hybGap"].val),
            "HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT": int(sliders["hybMinH"].val),
        }

    def _render(self, res: PipelineResult):
        p = self._params()
        c = self._config_overrides()
        fname = self._filenames[self._idx]
        source_path = _source_path_text(fname)
        short_path = _truncate_middle(source_path, 136)

        self._fig.suptitle(
            (
                f"[{self._idx + 1}/{len(self._originals)}] image: {short_path}\n"
                f"label={res.labeled_coin_count}  pred={res.coin_count}  value={res.estimated_value_eur:.2f} EUR  "
                f"inverted={res.is_inverted}  mode={'fast' if self._fast_mode else 'full'}  "
                f"compute={self._last_compute_ms:.0f}ms\n"
                f"hough: dp={p['dp']:.1f}  minDist={p['minDist']}  p1={p['param1']}  p2={p['param2']}  "
                f"minR={p['minRadius']}  maxR={p['maxRadius']}\n"
                f"mask/hybrid: off={c['MASK_OTSU_OFFSET']}  minArea={c['MASK_MIN_COMPONENT_AREA']}  "
                f"fg={c['WATERSHED_DT_FG_RATIO']:.2f}  minCirc={c['GEOM_MIN_CIRCULARITY']:.2f}  "
                f"gap={c['HYBRID_MASK_OVER_HOUGH_MIN_GAP']}  minH={c['HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT']}"
            ),
            fontsize=9.5,
            x=0.01,
            ha="left",
            y=0.99,
        )

        for idx, ax in enumerate(self._axes.flat):
            ax.clear()
            ax.axis("off")
            if idx >= len(res.steps):
                continue

            step = res.steps[idx]
            if step.cmap == "rgb":
                ax.imshow(cv2.cvtColor(step.image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(step.image, cmap="gray")
            ax.set_title(step.name, fontsize=11, color="#111827")
            self._apply_zoom_state(ax=ax, axis_index=idx, image=step.image)

        if self._status_text is not None:
            self._status_text.set_text(
                (
                    f"path: {source_path}\n"
                    f"controls: a/d or arrows=nav  f=fast  l=lock  r=reset  q/esc=quit  "
                    f"zoom=wheel  reset-zoom=right-click"
                )
            )

        self._fig.canvas.draw_idle()

    def _compute_live(self):
        params = self._params()
        cfg_overrides = self._config_overrides()
        if params["maxRadius"] <= params["minRadius"]:
            self._suppress_updates = True
            self._sliders["maxR"].set_val(params["minRadius"] + 1)
            self._suppress_updates = False
            params = self._params()

        key = (
            self._idx,
            int(self._fast_mode),
            round(params["dp"], 1),
            params["minDist"],
            params["param1"],
            params["param2"],
            params["minRadius"],
            params["maxRadius"],
            cfg_overrides["MASK_OTSU_OFFSET"],
            cfg_overrides["MASK_MIN_COMPONENT_AREA"],
            round(cfg_overrides["WATERSHED_DT_FG_RATIO"], 2),
            round(cfg_overrides["GEOM_MIN_CIRCULARITY"], 2),
            cfg_overrides["HYBRID_MASK_OVER_HOUGH_MIN_GAP"],
            cfg_overrides["HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT"],
        )
        img = self._originals[self._idx]
        fname = self._filenames[self._idx]
        start = perf_counter()
        result = self._cache.get(key)
        if result is None:
            result = self._processor.detect_with_params(
                img_bgr_resized=img,
                filename=fname,
                classify=not self._fast_mode,
                config_overrides=cfg_overrides,
                **params,
            )
            self._cache[key] = result
            if len(self._cache) > self._cache_limit:
                self._cache.pop(next(iter(self._cache)))

        self._last_compute_ms = (perf_counter() - start) * 1000.0
        self._render(result)

    def _schedule_update(self, _=None):
        if self._locked or self._suppress_updates:
            return

        if self._pending_timer is not None:
            try:
                self._pending_timer.stop()
            except Exception:
                pass

        self._pending_timer = self._fig.canvas.new_timer(interval=self._debounce_ms)
        self._pending_timer.add_callback(self._compute_live)
        self._pending_timer.start()

    def _shift_image(self, delta: int):
        self._idx = (self._idx + delta) % len(self._originals)
        if self._image_slider is not None:
            self._syncing_index_slider = True
            self._image_slider.set_val(self._idx + 1)
            self._syncing_index_slider = False
        self._compute_live()

    def _on_key(self, event):
        if event.key in self._NAV_KEYS:
            self._shift_image(self._NAV_KEYS[event.key])
            return
        if event.key == "f":
            self._toggle("fast")
        elif event.key == "l":
            self._toggle("lock")
        elif event.key == "r":
            self._on_reset(None)
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _on_reset(self, _event):
        self._suppress_updates = True
        for key, _label, _rect, _vmin, _vmax, cfg_attr, _step, caster in self._PARAM_SPECS:
            self._sliders[key].set_val(caster(getattr(self._cfg, cfg_attr)))
        self._suppress_updates = False
        self._zoom_state.clear()
        self._compute_live()

    def _toggle(self, mode: str):
        if mode == "lock":
            self._locked = not self._locked
            self._buttons["lock"].label.set_text("Lock: ON" if self._locked else "Lock: OFF")
            self._fig.canvas.draw_idle()
            if not self._locked:
                self._compute_live()
            return

        if mode == "fast":
            self._fast_mode = not self._fast_mode
            self._buttons["fast"].label.set_text("Fast: ON" if self._fast_mode else "Fast: OFF")
            self._fig.canvas.draw_idle()
            self._compute_live()

    def _on_image_slider(self, value):
        if self._syncing_index_slider:
            return
        new_idx = int(round(value)) - 1
        self._idx = int(np.clip(new_idx, 0, len(self._originals) - 1))
        self._compute_live()

    def _apply_zoom_state(self, ax, axis_index: int, image: np.ndarray):
        if axis_index not in self._zoom_state:
            return
        h, w = image.shape[:2]
        x0n, x1n, y0n, y1n = self._zoom_state[axis_index]

        x0 = float(np.clip(min(x0n, x1n) * w, 0, max(0, w - 1)))
        x1 = float(np.clip(max(x0n, x1n) * w, 1, max(1, w)))
        y0 = float(np.clip(min(y0n, y1n) * h, 0, max(0, h - 1)))
        y1 = float(np.clip(max(y0n, y1n) * h, 1, max(1, h)))
        ax.set_xlim((x0, x1))
        ax.set_ylim((y1, y0))

    def _on_scroll_zoom(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        axis_index = self._axis_to_index.get(event.inaxes)
        if axis_index is None:
            return
        ax = event.inaxes
        if not ax.images:
            return
        img = ax.images[0].get_array()

        h, w = img.shape[:2]
        cur_xmin, cur_xmax = sorted(ax.get_xlim())
        cur_ymin, cur_ymax = sorted(ax.get_ylim())

        zoom_in = bool(getattr(event, "step", 0) > 0 or event.button == "up")
        scale = (1.0 / 1.2) if zoom_in else 1.2
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
        self._zoom_state[axis_index] = (
            float(np.clip(x0 / max(w, 1), 0.0, 1.0)),
            float(np.clip(x1 / max(w, 1), 0.0, 1.0)),
            float(np.clip(y0 / max(h, 1), 0.0, 1.0)),
            float(np.clip(y1 / max(h, 1), 0.0, 1.0)),
        )
        self._fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.button not in (3, MouseButton.RIGHT):
            return
        axis_index = self._axis_to_index.get(event.inaxes)
        if axis_index is None:
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
        slider.poly.set_facecolor("#2a6fdb")
        if hasattr(slider, "vline"):
            slider.vline.set_color("#1e3a8a")
        if hasattr(slider, "track"):
            slider.track.set_color("#d7dce5")
        slider.label.set_fontsize(10)
