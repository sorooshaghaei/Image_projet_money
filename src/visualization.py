import os
from math import ceil
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .models import PipelineResult
from .processor import CoinProcessor


def save_pipeline_steps(result: PipelineResult, out_dir: str, cols: int = 4) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    steps = result.steps
    n = len(steps)
    if n == 0:
        return None

    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    fig.suptitle(
        f"{result.source_filename} | pred={result.coin_count} | inverted={result.is_inverted}",
        fontsize=14,
    )

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, step in enumerate(steps):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = step.image

        if step.cmap == "rgb":
            img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_disp)
        else:
            ax.imshow(img, cmap="gray")

        ax.set_title(step.name, fontsize=11)
        ax.axis("off")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{os.path.splitext(result.source_filename)[0]}_pipeline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def browse_and_tune(processor: CoinProcessor, results: List[PipelineResult], cols: int = 4):
    if not results:
        print("[WARN] No results to display.")
        return

    originals: List[np.ndarray] = [r.steps[0].image.copy() for r in results]
    filenames: List[str] = [r.source_filename for r in results]

    cfg = processor._cfg
    idx = 0
    locked = False

    max_steps = 5
    rows = ceil(max_steps / cols)

    fig = plt.figure(figsize=(4.5 * cols, 4.0 * rows + 3.2))
    gs = fig.add_gridspec(rows, cols, top=0.88, bottom=0.27, left=0.03, right=0.99, wspace=0.05, hspace=0.15)

    axes = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    ax_dp = fig.add_axes([0.06, 0.20, 0.40, 0.03])
    ax_minDist = fig.add_axes([0.06, 0.16, 0.40, 0.03])
    ax_p1 = fig.add_axes([0.06, 0.12, 0.40, 0.03])
    ax_p2 = fig.add_axes([0.06, 0.08, 0.40, 0.03])
    ax_minR = fig.add_axes([0.56, 0.20, 0.38, 0.03])
    ax_maxR = fig.add_axes([0.56, 0.16, 0.38, 0.03])

    s_dp = Slider(ax_dp, "dp", 0.5, 3.0, valinit=float(cfg.HOUGH_DP), valstep=0.1)
    s_minDist = Slider(ax_minDist, "minDist", 1, 300, valinit=int(cfg.HOUGH_MIN_DIST), valstep=1)
    s_p1 = Slider(ax_p1, "param1", 1, 300, valinit=int(cfg.HOUGH_PARAM1), valstep=1)
    s_p2 = Slider(ax_p2, "param2", 1, 300, valinit=int(cfg.HOUGH_PARAM2), valstep=1)
    s_minR = Slider(ax_minR, "minR", 0, 300, valinit=int(cfg.HOUGH_MIN_RADIUS), valstep=1)
    s_maxR = Slider(ax_maxR, "maxR", 1, 500, valinit=int(cfg.HOUGH_MAX_RADIUS), valstep=1)

    ax_reset = fig.add_axes([0.56, 0.08, 0.18, 0.06])
    btn_reset = Button(ax_reset, "Reset")

    ax_lock = fig.add_axes([0.76, 0.08, 0.18, 0.06])
    btn_lock = Button(ax_lock, "Lock: OFF")

    pending = {"timer": None}

    def render(res: PipelineResult, fname: str, k: int):
        dp = s_dp.val
        md = int(s_minDist.val)
        p1 = int(s_p1.val)
        p2 = int(s_p2.val)
        mn = int(s_minR.val)
        mx = int(s_maxR.val)

        fig.suptitle(
            f"[{k + 1}/{len(originals)}] {fname} | pred={res.coin_count} | inv={res.is_inverted} | "
            f"dp={dp:.1f} minDist={md} p1={p1} p2={p2} minR={mn} maxR={mx} | "
            f"(a/d nav, q/esc quit)",
            fontsize=12,
        )

        for j in range(rows * cols):
            rr, cc = divmod(j, cols)
            ax = axes[rr, cc]
            ax.clear()
            ax.axis("off")

            if j >= len(res.steps):
                continue

            step = res.steps[j]
            img = step.image
            if step.cmap == "rgb":
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(step.name, fontsize=11)

        fig.canvas.draw_idle()

    def clamp_slider_pair():
        mn = int(s_minR.val)
        mx = int(s_maxR.val)
        if mx <= mn:
            s_maxR.set_val(mn + 1)

    def compute_live():
        clamp_slider_pair()

        img = originals[idx]
        fname = filenames[idx]

        res = processor.detect_with_params(
            img_bgr_resized=img,
            dp=float(s_dp.val),
            minDist=int(s_minDist.val),
            param1=int(s_p1.val),
            param2=int(s_p2.val),
            minRadius=int(s_minR.val),
            maxRadius=int(s_maxR.val),
            filename=fname,
        )
        render(res, fname, idx)

    def schedule_update(_=None):
        nonlocal locked
        if locked:
            return
        if pending["timer"] is not None:
            try:
                pending["timer"].stop()
            except Exception:
                pass

        t = fig.canvas.new_timer(interval=120)
        t.add_callback(compute_live)
        pending["timer"] = t
        t.start()

    def on_key(event):
        nonlocal idx
        if event.key == "d":
            idx = (idx + 1) % len(originals)
            compute_live()
        elif event.key == "a":
            idx = (idx - 1) % len(originals)
            compute_live()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    def on_reset(_event):
        s_dp.set_val(float(cfg.HOUGH_DP))
        s_minDist.set_val(int(cfg.HOUGH_MIN_DIST))
        s_p1.set_val(int(cfg.HOUGH_PARAM1))
        s_p2.set_val(int(cfg.HOUGH_PARAM2))
        s_minR.set_val(int(cfg.HOUGH_MIN_RADIUS))
        s_maxR.set_val(int(cfg.HOUGH_MAX_RADIUS))
        compute_live()

    def on_lock(_event):
        nonlocal locked
        locked = not locked
        btn_lock.label.set_text("Lock: ON" if locked else "Lock: OFF")
        fig.canvas.draw_idle()
        if not locked:
            compute_live()

    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_reset.on_clicked(on_reset)
    btn_lock.on_clicked(on_lock)

    s_dp.on_changed(schedule_update)
    s_minDist.on_changed(schedule_update)
    s_p1.on_changed(schedule_update)
    s_p2.on_changed(schedule_update)
    s_minR.on_changed(schedule_update)
    s_maxR.on_changed(schedule_update)

    compute_live()
    plt.show()
