import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple
from dataclasses import dataclass
from math import ceil
from matplotlib.widgets import Slider, Button

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class DetectionConfig:
    TARGET_WIDTH: int = 800
    BLUR_KERNEL_SIZE: int = 15

    # Hough Circle Parameters (defaults)
    HOUGH_DP: float = 1.2
    HOUGH_MIN_DIST: int = 70
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 45
    HOUGH_MIN_RADIUS: int = 10
    HOUGH_MAX_RADIUS: int = 150

    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class PipelineStep:
    name: str
    image: np.ndarray
    cmap: str


@dataclass
class PipelineResult:
    steps: List[PipelineStep]
    coin_count: int
    is_inverted: bool
    source_filename: str


# ==========================================
# 2. COIN PROCESSOR
# ==========================================
class CoinProcessor:
    """
    Resize, Grayscale, Normalize, Optional invert, Median blur, HoughCircles, Draw.
    """

    def __init__(self, config: DetectionConfig):
        self._cfg = config

    def execute(self, img: np.ndarray, filename: str = "Unknown") -> Optional[PipelineResult]:
        if img is None or img.size == 0:
            return None

        img_resized = self._resize(img)
        return self.detect_with_params(
            img_bgr_resized=img_resized,
            dp=self._cfg.HOUGH_DP,
            minDist=self._cfg.HOUGH_MIN_DIST,
            param1=self._cfg.HOUGH_PARAM1,
            param2=self._cfg.HOUGH_PARAM2,
            minRadius=self._cfg.HOUGH_MIN_RADIUS,
            maxRadius=self._cfg.HOUGH_MAX_RADIUS,
            filename=filename,
        )

    def detect_with_params(
        self,
        img_bgr_resized: np.ndarray,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
        filename: str = "LIVE_TUNE",
    ) -> PipelineResult:
        steps: List[PipelineStep] = []

        display_img = img_bgr_resized.copy()
        steps.append(PipelineStep("1. Original", img_bgr_resized, "rgb"))

        gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        mean_brightness = float(np.mean(gray))
        inverted = False
        if mean_brightness < 110:
            gray = cv2.bitwise_not(gray)
            inverted = True
            steps.append(PipelineStep("2a. Inverted (Low Brightness)", gray, "gray"))
        else:
            steps.append(PipelineStep("2. Grayscale", gray, "gray"))

        blurred = cv2.medianBlur(gray, self._cfg.BLUR_KERNEL_SIZE)
        steps.append(PipelineStep("3. Median Blur", blurred, "gray"))

        # safety clamp
        minRadius = int(max(0, minRadius))
        maxRadius = int(max(minRadius + 1, maxRadius))
        minDist = int(max(1, minDist))
        param1 = int(max(1, param1))
        param2 = int(max(1, param2))
        dp = float(max(0.1, dp))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        mask = np.zeros_like(gray)
        coin_count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            coin_count = circles.shape[1]

            for x, y, r in circles[0, :]:
                cv2.circle(display_img, (int(x), int(y)), int(r), (0, 255, 0), 3)
                cv2.circle(display_img, (int(x), int(y)), 2, (0, 0, 255), 3)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        steps.append(PipelineStep("4. Detected Circles", display_img, "rgb"))
        steps.append(PipelineStep("5. Mask (Debug)", mask, "gray"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=inverted,
            source_filename=filename,
        )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == 0:
            return img
        scale = self._cfg.TARGET_WIDTH / w
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(h * scale)))


# ==========================================
# 3. UTILITIES
# ==========================================
def get_image_path(base_dir: str, filename: str, group: str) -> Optional[str]:
    path_grouped = os.path.join(base_dir, group, filename)
    if os.path.exists(path_grouped):
        return path_grouped

    if group.startswith("grp"):
        alt_group = group.replace("grp", "gp")
        path_alt = os.path.join(base_dir, alt_group, filename)
        if os.path.exists(path_alt):
            return path_alt

    path_flat = os.path.join(base_dir, filename)
    if os.path.exists(path_flat):
        return path_flat

    return None


def save_pipeline_steps(result: PipelineResult, out_dir: str, cols: int = 4) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    steps = result.steps
    n = len(steps)
    if n == 0:
        return None

    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    fig.suptitle(f"{result.source_filename} | pred={result.coin_count} | inverted={result.is_inverted}", fontsize=14)

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


# ==========================================
# 4. BROWSE + REALTIME HOUGH TUNING
# ==========================================
def browse_and_tune(processor: CoinProcessor, results: List[PipelineResult], cols: int = 4):
    """
    Keys:
      a = prev, d = next, q/Esc = quit
    Sliders:
      dp, minDist, param1, param2, minR, maxR (recompute live)
    Buttons:
      Reset (defaults), Lock (pause live recompute while dragging)
    """
    if not results:
        print("[WARN] No results to display.")
        return

    # We assume step[0] = resized original BGR
    originals: List[np.ndarray] = [r.steps[0].image.copy() for r in results]
    filenames: List[str] = [r.source_filename for r in results]

    cfg = processor._cfg
    idx = 0
    locked = False

    # Fixed: 5 steps in detect_with_params
    max_steps = 5
    rows = ceil(max_steps / cols)

    # Figure layout: grid on top, controls bottom
    fig = plt.figure(figsize=(4.5 * cols, 4.0 * rows + 3.2))
    gs = fig.add_gridspec(rows, cols, top=0.88, bottom=0.27, left=0.03, right=0.99, wspace=0.05, hspace=0.15)

    axes = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Sliders
    ax_dp      = fig.add_axes([0.06, 0.20, 0.40, 0.03])
    ax_minDist = fig.add_axes([0.06, 0.16, 0.40, 0.03])
    ax_p1      = fig.add_axes([0.06, 0.12, 0.40, 0.03])
    ax_p2      = fig.add_axes([0.06, 0.08, 0.40, 0.03])

    ax_minR    = fig.add_axes([0.56, 0.20, 0.38, 0.03])
    ax_maxR    = fig.add_axes([0.56, 0.16, 0.38, 0.03])

    s_dp = Slider(ax_dp, "dp", 0.5, 3.0, valinit=float(cfg.HOUGH_DP), valstep=0.1)
    s_minDist = Slider(ax_minDist, "minDist", 1, 300, valinit=int(cfg.HOUGH_MIN_DIST), valstep=1)
    s_p1 = Slider(ax_p1, "param1", 1, 300, valinit=int(cfg.HOUGH_PARAM1), valstep=1)
    s_p2 = Slider(ax_p2, "param2", 1, 300, valinit=int(cfg.HOUGH_PARAM2), valstep=1)
    s_minR = Slider(ax_minR, "minR", 0, 300, valinit=int(cfg.HOUGH_MIN_RADIUS), valstep=1)
    s_maxR = Slider(ax_maxR, "maxR", 1, 500, valinit=int(cfg.HOUGH_MAX_RADIUS), valstep=1)

    # Buttons
    ax_reset = fig.add_axes([0.56, 0.08, 0.18, 0.06])
    btn_reset = Button(ax_reset, "Reset")

    ax_lock = fig.add_axes([0.76, 0.08, 0.18, 0.06])
    btn_lock = Button(ax_lock, "Lock: OFF")

    # Debounce timer (avoid recompute on every pixel while dragging)
    pending = {"timer": None}

    def render(res: PipelineResult, fname: str, k: int):
        # read current slider values for title
        dp = s_dp.val
        md = int(s_minDist.val)
        p1 = int(s_p1.val)
        p2 = int(s_p2.val)
        mn = int(s_minR.val)
        mx = int(s_maxR.val)

        fig.suptitle(
            f"[{k+1}/{len(originals)}] {fname} | pred={res.coin_count} | inv={res.is_inverted} | "
            f"dp={dp:.1f} minDist={md} p1={p1} p2={p2} minR={mn} maxR={mx} | "
            f"(a/d nav, q/esc quit)",
            fontsize=12
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
        # enforce maxR >= minR+1
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

        t = fig.canvas.new_timer(interval=120)  # ms
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

    # Bind events
    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_reset.on_clicked(on_reset)
    btn_lock.on_clicked(on_lock)

    s_dp.on_changed(schedule_update)
    s_minDist.on_changed(schedule_update)
    s_p1.on_changed(schedule_update)
    s_p2.on_changed(schedule_update)
    s_minR.on_changed(schedule_update)
    s_maxR.on_changed(schedule_update)

    # First render
    compute_live()
    plt.show()


# ==========================================
# 5. DATA
# ==========================================
DATA_ROWS = [
    ["exemple1.png", 4, 7.25, "gp1"],
    ["10.jpg", 9, 3.13, "gp5"],
    ["11.jpg", 12, 6.18, "gp5"],
    ["12.jpg", 16, 8.83, "gp5"],
    ["13.jpg", 19, 12.33, "gp5"],
    ["14.jpg", 28, 15.69, "gp5"],
    ["15.jpg", 35, 17.32, "gp5"],
    ["16.jpg", 48, 18.69, "gp5"],
    ["17.jpg", 48, 18.20, "gp5"],
    ["0.jpeg", 2, 2.20, "gp5"],
    ["1.jpeg", 4, 4.22, "gp5"],
    ["2.jpeg", 3, 3.20, "gp5"],
    ["3.jpeg", 4, 0.80, "gp5"],
    ["4.jpeg", 3, 3.00, "gp5"],
    ["5.jpeg", 2, 1.20, "gp5"],
    ["6.jpeg", 11, 10.26, "gp5"],
    ["7.jpeg", 3, 1.70, "gp5"],
    ["8.jpg", 6, None, "gp5"],
    ["9.jpg", 8, 3.88, "gp5"],

    ["18.png", 7, 4.31, "gp1"],
    ["19.png", 4, 1.60, "gp1"],
    ["20.png", 8, 4.81, "gp1"],
    ["21.png", 6, 3.76, "gp1"],
    ["22.png", 5, 2.25, "gp1"],
    ["23.png", 8, 4.34, "gp1"],
    ["24.png", 3, 2.55, "gp1"],
    ["25.png", 10, 4.40, "gp1"],
    ["26.jpg", 8, 3.51, "gp1"],
    ["27.jpg", 9, 0.88, "gp1"],
    ["28.jpg", 3, 0.21, "gp1"],
    ["29.jpg", 5, 0.36, "gp1"],
    ["30.jpg", 7, 3.72, "gp1"],
    ["31.jpg", 4, 1.70, "gp1"],

    ["3_1.jpg", 8, 5.00, "grp3"],
    ["3_2.jpg", 16, 4.80, "grp3"],
    ["3_3.jpg", 8, 5.00, "grp3"],
    ["3_4.jpg", 10, 4.03, "grp3"],
    ["3_5.jpg", 25, 12.50, "grp3"],
    ["3_6.jpg", 8, 16.00, "grp3"],
    ["3_7.jpg", 8, 16.00, "grp3"],
    ["3_8.jpg", 50, 5.00, "grp3"],
    ["3_9.jpg", 24, 24.00, "grp3"],
    ["3_10.jpg", 35, 3.50, "grp3"],

    ["2e01.jpg", 8, 2.01, "grp5"],
    ["3e19.jpg", 10, 3.19, "grp5"],
    ["4.17.jpg", 12, 4.17, "grp5"],
    ["4e22.jpg", 8, 4.22, "grp5"],
    ["6e19.jpg", 12, 6.19, "grp5"],
    ["8e88.jpg", 20, 8.88, "grp5"],
    ["10e05.jpg", 26, 10.05, "grp5"],

    ["1.jpg", 2, 1.50, "grp4"],
    ["2.jpg", 4, 2.27, "grp4"],
    ["3.jpg", 5, 3.27, "grp4"],
    ["4.jpg", 7, 1.88, "grp4"],
    ["5.jpg", 8, 4.38, "grp4"],
    ["6.jpg", 7, 2.37, "grp4"],
    ["7.jpg", 8, 3.88, "grp4"],
    ["8.jpg", 8, 3.88, "grp4"],
    ["9.jpg", 4, 2.65, "grp4"],
    ["10.jpg", 7, 5.12, "grp4"],

    ["60.jpg", 13, 6.33, "gp6"],
    ["61.jpg", 11, 5.53, "gp6"],
    ["62.jpg", 9, 6.86, "gp6"],
    ["63.jpg", 9, 5.34, "gp6"],
    ["64.jpg", 12, 7.07, "gp6"],
    ["65.jpg", 13, 2.63, "gp6"],
    ["66.jpg", 7, 0.77, "gp6"],
    ["67.jpg", 10, 3.31, "gp6"],
    ["68.jpg", 11, 5.41, "gp6"],
    ["69.jpg", 9, 7.40, "gp6"],

    ["gp7_01.webp", 7, 3.79, "gp7"],
    ["gp7_02.webp", 12, 1.85, "gp7"],
    ["gp7_03.webp", 12, 4.60, "gp7"],
    ["gp7_04.webp", 13, 4.65, "gp7"],
    ["gp7_05.webp", 12, 4.15, "gp7"],
    ["gp7_06.webp", 12, 4.74, "gp7"],
    ["gp7_07.webp", 11, 3.74, "gp7"],
    ["gp7_08.webp", 10, 4.19, "gp7"],
    ["gp7_09.webp", 11, 2.55, "gp7"],
    ["gp7_10.webp", 9, 4.46, "gp7"],
    ["gp7_11.webp", 10, 4.03, "gp7"],
    ["gp7_12.webp", 14, 4.95, "gp7"],

    ["IMG_1136.png", 5, 0.83, "gp8"],
    ["IMG_1137.png", 10, 2.16, "gp8"],
    ["IMG_1138.png", 9, 2.17, "gp8"],
    ["IMG_1139.png", 4, 1.21, "gp8"],
    ["IMG_1140.png", 11, 2.47, "gp8"],
    ["IMG_1141.png", 7, 1.36, "gp8"],
    ["IMG_1142.png", 4, 1.52, "gp8"],
    ["IMG_1143.png", 17, 1.40, "gp8"],
    ["IMG_1144.png", 16, 0.43, "gp8"],
    ["IMG_1145.png", 5, 2.12, "gp8"],

    ["1.jpeg", 8, 3.86, "gp2"],
    ["2.jpeg", 2, 3.00, "gp2"],
    ["3.jpeg", 3, 2.70, "gp2"],
    ["4.jpeg", 8, 3.86, "gp2"],
    ["5.jpeg", 3, 0.24, "gp2"],
    ["6.jpeg", 9, 3.98, "gp2"],
    ["7.jpeg", 9, 3.98, "gp2"],
    ["8.jpeg", 3, 3.50, "gp2"],
    ["9.jpeg", 6, 0.96, "gp2"],
    ["10.jpeg", 6, 0.96, "gp2"],
    ["11.jpeg", 9, 3.37, "gp2"],
    ["12.jpeg", 2, 3.00, "gp2"],
    ["13.jpeg", 9, 3.87, "gp2"],
    ["14.jpeg", 4, 2.45, "gp2"],
    ["15.jpeg", 5, 3.90, "gp2"],
]


# ==========================================
# 6. MAIN
# ==========================================
def main():
    IMAGE_DIRECTORY = "/Users/sigmoid/Desktop/Coding/Git/Image_projet_money/data/images"

    # Toggles
    BROWSE_TUNE = True        # interactive browse + realtime Hough tuning
    SAVE_STEPS = False        # optional: save default-pipeline grids
    OUT_DIR = "./pipeline_viz"

    config = DetectionConfig()
    processor = CoinProcessor(config)

    df = pd.DataFrame(DATA_ROWS, columns=["image", "pieces", "value_eur", "group"])
    print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

    correct = 0
    total_processed = 0
    total_abs_error = 0

    results_all: List[PipelineResult] = []

    print("\n" + "=" * 85)
    print(f"{'FILENAME':<25} | {'GRP':<5} | {'PRED':<6} | {'TRUE':<6} | {'DIFF':<6} | {'STATUS':<10}")
    print("=" * 85)

    for _, row in df.iterrows():
        filename = row["image"]
        true_count = int(row["pieces"])
        group = row["group"]

        image_path = get_image_path(IMAGE_DIRECTORY, filename, group)
        if not image_path:
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERR ] Unreadable: {filename}")
            continue

        result = processor.execute(img, filename)
        if not result:
            continue

        pred = int(result.coin_count)
        diff = pred - true_count
        total_abs_error += abs(diff)
        total_processed += 1

        status = "PERFECT" if diff == 0 else "ERROR"
        if diff == 0:
            correct += 1

        print(f"{filename:<25} | {group:<5} | {pred:<6} | {true_count:<6} | {diff:<6} | {status:<10}")

        results_all.append(result)

        if SAVE_STEPS:
            saved = save_pipeline_steps(result, OUT_DIR, cols=4)
            if saved:
                print(f"[SAVED] {saved}")

    if total_processed > 0:
        acc = (correct / total_processed) * 100.0
        mae = total_abs_error / total_processed
        print("=" * 85)
        print(f"Total Images:     {total_processed}")
        print(f"Perfect Matches:  {correct}")
        print(f"Accuracy:         {acc:.2f}%")
        print(f"Mean Abs Error:   {mae:.2f} coins/image")
        print("=" * 85)
    else:
        print("[WARN] No images processed. Check IMAGE_DIRECTORY path.")

    if BROWSE_TUNE:
        # You can change cols to 3 if you want bigger subplots.
        browse_and_tune(processor, results_all, cols=4)


if __name__ == "__main__":
    main()
