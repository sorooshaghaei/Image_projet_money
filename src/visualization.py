from math import ceil
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .models import PipelineResult
from .processor import CoinProcessor


class PipelineVisualizer:
    def save_pipeline_steps(self, result: PipelineResult, out_dir: str, cols: int = 4) -> Optional[str]:
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
        if rows == 1 and cols == 1:
            return np.array([[axes]])
        if rows == 1:
            return np.array([axes])
        if cols == 1:
            return np.array([[ax] for ax in axes])
        return axes


class HoughTuningBrowser:
    def __init__(self, processor: CoinProcessor, results: List[PipelineResult], cols: int = 4):
        self._processor = processor
        self._results = results
        self._cols = cols

        self._idx = 0
        self._locked = False
        self._pending = {"timer": None}

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
        self._btn_lock = None

    def show(self):
        if not self._results:
            print("[WARN] No results to display.")
            return

        self._setup_figure()
        self._bind_events()
        self._compute_live()
        plt.show()

    def _setup_figure(self):
        self._fig = plt.figure(figsize=(4.5 * self._cols, 4.0 * self._rows + 3.2))
        gs = self._fig.add_gridspec(
            self._rows,
            self._cols,
            top=0.88,
            bottom=0.27,
            left=0.03,
            right=0.99,
            wspace=0.05,
            hspace=0.15,
        )

        self._axes = np.empty((self._rows, self._cols), dtype=object)
        for r in range(self._rows):
            for c in range(self._cols):
                self._axes[r, c] = self._fig.add_subplot(gs[r, c])

        ax_dp = self._fig.add_axes([0.06, 0.20, 0.40, 0.03])
        ax_minDist = self._fig.add_axes([0.06, 0.16, 0.40, 0.03])
        ax_p1 = self._fig.add_axes([0.06, 0.12, 0.40, 0.03])
        ax_p2 = self._fig.add_axes([0.06, 0.08, 0.40, 0.03])
        ax_minR = self._fig.add_axes([0.56, 0.20, 0.38, 0.03])
        ax_maxR = self._fig.add_axes([0.56, 0.16, 0.38, 0.03])

        self._s_dp = Slider(ax_dp, "dp", 0.5, 3.0, valinit=float(self._cfg.HOUGH_DP), valstep=0.1)
        self._s_minDist = Slider(ax_minDist, "minDist", 1, 300, valinit=int(self._cfg.HOUGH_MIN_DIST), valstep=1)
        self._s_p1 = Slider(ax_p1, "param1", 1, 300, valinit=int(self._cfg.HOUGH_PARAM1), valstep=1)
        self._s_p2 = Slider(ax_p2, "param2", 1, 300, valinit=int(self._cfg.HOUGH_PARAM2), valstep=1)
        self._s_minR = Slider(ax_minR, "minR", 0, 300, valinit=int(self._cfg.HOUGH_MIN_RADIUS), valstep=1)
        self._s_maxR = Slider(ax_maxR, "maxR", 1, 500, valinit=int(self._cfg.HOUGH_MAX_RADIUS), valstep=1)

        ax_reset = self._fig.add_axes([0.56, 0.08, 0.18, 0.06])
        btn_reset = Button(ax_reset, "Reset")

        ax_lock = self._fig.add_axes([0.76, 0.08, 0.18, 0.06])
        self._btn_lock = Button(ax_lock, "Lock: OFF")

        btn_reset.on_clicked(self._on_reset)
        self._btn_lock.on_clicked(self._on_lock)

    def _bind_events(self):
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._s_dp.on_changed(self._schedule_update)
        self._s_minDist.on_changed(self._schedule_update)
        self._s_p1.on_changed(self._schedule_update)
        self._s_p2.on_changed(self._schedule_update)
        self._s_minR.on_changed(self._schedule_update)
        self._s_maxR.on_changed(self._schedule_update)

    def _render(self, res: PipelineResult):
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
            f"(a/d nav, q/esc quit)",
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
            ax.set_title(step.name, fontsize=11)

        self._fig.canvas.draw_idle()

    def _clamp_slider_pair(self):
        mn = int(self._s_minR.val)
        mx = int(self._s_maxR.val)
        if mx <= mn:
            self._s_maxR.set_val(mn + 1)

    def _compute_live(self):
        self._clamp_slider_pair()

        img = self._originals[self._idx]
        fname = self._filenames[self._idx]

        res = self._processor.detect_with_params(
            img_bgr_resized=img,
            dp=float(self._s_dp.val),
            minDist=int(self._s_minDist.val),
            param1=int(self._s_p1.val),
            param2=int(self._s_p2.val),
            minRadius=int(self._s_minR.val),
            maxRadius=int(self._s_maxR.val),
            filename=fname,
        )
        self._render(res)

    def _schedule_update(self, _=None):
        if self._locked:
            return

        timer = self._pending["timer"]
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

        timer = self._fig.canvas.new_timer(interval=120)
        timer.add_callback(self._compute_live)
        self._pending["timer"] = timer
        timer.start()

    def _on_key(self, event):
        if event.key == "d":
            self._idx = (self._idx + 1) % len(self._originals)
            self._compute_live()
        elif event.key == "a":
            self._idx = (self._idx - 1) % len(self._originals)
            self._compute_live()
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _on_reset(self, _event):
        self._s_dp.set_val(float(self._cfg.HOUGH_DP))
        self._s_minDist.set_val(int(self._cfg.HOUGH_MIN_DIST))
        self._s_p1.set_val(int(self._cfg.HOUGH_PARAM1))
        self._s_p2.set_val(int(self._cfg.HOUGH_PARAM2))
        self._s_minR.set_val(int(self._cfg.HOUGH_MIN_RADIUS))
        self._s_maxR.set_val(int(self._cfg.HOUGH_MAX_RADIUS))
        self._compute_live()

    def _on_lock(self, _event):
        self._locked = not self._locked
        self._btn_lock.label.set_text("Lock: ON" if self._locked else "Lock: OFF")
        self._fig.canvas.draw_idle()
        if not self._locked:
            self._compute_live()
