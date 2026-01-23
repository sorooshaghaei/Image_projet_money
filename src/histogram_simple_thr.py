from pathlib import Path
import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"

WIN_VIEW = "viewer"
WIN_HIST = "hist"
WIN_CTRL = "controls"
WIN_MASK = "removed mask"
WIN_OUT  = "output"


def iter_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def draw_header(img, text: str):
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)


def _noop(_=None):
    pass


# ---------------- Histogram peak removal (simple) ----------------

def smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) | 1
    ker = np.ones(k, np.float32) / k
    return np.convolve(x.astype(np.float32), ker, mode="same")


def find_peaks_1d(y: np.ndarray) -> list[int]:
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
    return peaks


def peak_interval(y_norm: np.ndarray, p: int, drop: float) -> tuple[int, int]:
    """
    y_norm is 0..1.
    Interval = walk left/right until below y[p]*drop.
    """
    thr = float(y_norm[p]) * float(drop)
    lo = p
    hi = p
    while lo > 0 and y_norm[lo] > thr:
        lo -= 1
    while hi < len(y_norm) - 1 and y_norm[hi] > thr:
        hi += 1
    return int(lo), int(hi)


def remove_hist_peaks_mask(gray_u8: np.ndarray, peaks_n: int, smooth_k: int, drop: float, min_peak: float):
    """
    Returns:
      remove_mask (uint8 0/255), hs_norm (0..1), intervals list[(peak, lo, hi)]
    """
    hist = cv.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel().astype(np.float32)
    hs = smooth1d(hist, smooth_k)
    hs_norm = hs / (hs.max() + 1e-12)

    peaks = find_peaks_1d(hs_norm)
    peaks = [p for p in peaks if hs_norm[p] >= float(min_peak)]

    if not peaks:
        return np.zeros_like(gray_u8, np.uint8), hs_norm, []

    peaks = sorted(peaks, key=lambda p: hs_norm[p], reverse=True)
    peaks = peaks[: max(1, int(peaks_n))]
    peaks = sorted(peaks)

    remove = np.zeros_like(gray_u8, np.uint8)
    intervals = []
    for p in peaks:
        lo, hi = peak_interval(hs_norm, p, drop=drop)
        intervals.append((int(p), int(lo), int(hi)))
        remove |= cv.inRange(gray_u8, int(lo), int(hi))

    return remove, hs_norm, intervals


# ---------------- Histogram rendering (curve + removed ranges) ----------------

def render_histogram(h_norm_0_1: np.ndarray, intervals, w=1024, h=320):
    """
    Draw smooth histogram curve (0..1) + green ranges for removed peak intervals.
    """
    img = np.full((h, w, 3), 18, np.uint8)

    left, top, right, bottom = 40, 10, w - 10, h - 30
    cv.line(img, (left, bottom), (right, bottom), (80, 80, 80), 1, cv.LINE_AA)
    cv.line(img, (left, top), (left, bottom), (80, 80, 80), 1, cv.LINE_AA)

    plot_w = right - left
    plot_h = bottom - top

    def x_of(b): return left + int(b * plot_w / 255)
    def y_of(v): return top + int((1.0 - float(v)) * plot_h)

    # removed ranges as background
    for (p, lo, hi) in intervals:
        x0, x1 = x_of(lo), x_of(hi)
        cv.rectangle(img, (x0, top), (x1, bottom), (40, 70, 40), -1)

    # curve
    pts = [(x_of(b), y_of(h_norm_0_1[b])) for b in range(256)]
    for i in range(255):
        cv.line(img, pts[i], pts[i + 1], (230, 230, 230), 1, cv.LINE_AA)

    # peak markers (optional but useful)
    for (p, lo, hi) in intervals:
        cv.circle(img, (x_of(p), y_of(h_norm_0_1[p])), 4, (0, 255, 255), -1, cv.LINE_AA)
        cv.putText(img, f"{p}", (x_of(p) - 10, y_of(h_norm_0_1[p]) - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)

    # ticks
    for b in (0, 64, 128, 192, 255):
        x = x_of(b)
        cv.line(img, (x, bottom), (x, bottom + 4), (120, 120, 120), 1, cv.LINE_AA)
        cv.putText(img, str(b), (x - 10, h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv.LINE_AA)

    return img


# ---------------- Main UI ----------------

def main():
    if not IN_DIR.exists():
        print(f"Input dir not found: {IN_DIR}")
        return

    paths = iter_images_recursive(IN_DIR)
    if not paths:
        print(f"No images found under: {IN_DIR}")
        return

    cv.namedWindow(WIN_VIEW, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_HIST, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_MASK, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_OUT,  cv.WINDOW_NORMAL)

    # Controls:
    # - PEAKS: how many dominant peaks to remove
    # - SMOOTH: smoothing window (odd internally)
    # - DROP%: interval width (lower => wider removal)
    # - MIN%: ignore small peaks
    cv.createTrackbar("PEAKS",  WIN_CTRL, 1, 6, _noop)         # 1..6
    cv.createTrackbar("SMOOTH", WIN_CTRL, 21, 101, _noop)      # 1..101 (odd enforced)
    cv.createTrackbar("DROP%",  WIN_CTRL, 25, 90, _noop)       # 5..90 typical
    cv.createTrackbar("MIN%",   WIN_CTRL, 2, 50, _noop)        # 0..50 (% of max)

    idx = 0
    cached_img = None
    cached_path = None

    while True:
        p = paths[idx]

        if cached_img is None or cached_path != p:
            img = cv.imread(str(p), cv.IMREAD_COLOR)
            cached_img = img
            cached_path = p
        else:
            img = cached_img

        if img is None:
            print(f"[skip] failed to read: {p}")
            idx = (idx + 1) % len(paths)
            cached_img = None
            cached_path = None
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        peaks_n = max(1, cv.getTrackbarPos("PEAKS", WIN_CTRL))
        smooth_k = max(3, cv.getTrackbarPos("SMOOTH", WIN_CTRL))
        smooth_k = int(smooth_k) | 1
        drop = max(1, cv.getTrackbarPos("DROP%", WIN_CTRL)) / 100.0
        min_peak = max(0, cv.getTrackbarPos("MIN%", WIN_CTRL)) / 100.0

        remove_mask, hs_norm, intervals = remove_hist_peaks_mask(
            gray, peaks_n=peaks_n, smooth_k=smooth_k, drop=drop, min_peak=min_peak
        )

        # output: remove selected peak ranges (set to black) OR show mask overlay
        out = img.copy()
        out[remove_mask > 0] = (0, 0, 0)

        # histogram window
        hist_img = render_histogram(hs_norm, intervals, w=1024, h=320)

        # viewer header
        vis = img.copy()
        interval_str = " ".join([f"[{lo},{hi}]" for (_, lo, hi) in intervals]) if intervals else "none"
        header = (
            f"[{idx+1}/{len(paths)}] {p.relative_to(PROJECT_ROOT)} | "
            f"PEAKS={peaks_n} SMOOTH={smooth_k} DROP={drop:.2f} MIN={min_peak:.2f} | "
            f"removed: {interval_str} | keys: n/p, r reset, q"
        )
        draw_header(vis, header)

        cv.imshow(WIN_VIEW, vis)
        cv.imshow(WIN_HIST, hist_img)
        cv.imshow(WIN_MASK, remove_mask)
        cv.imshow(WIN_OUT, out)

        key = cv.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        elif key in (ord("n"), ord(" ")):
            idx = (idx + 1) % len(paths)
            cached_img = None
            cached_path = None
        elif key == ord("p"):
            idx = (idx - 1) % len(paths)
            cached_img = None
            cached_path = None
        elif key == ord("r"):
            cv.setTrackbarPos("PEAKS",  WIN_CTRL, 1)
            cv.setTrackbarPos("SMOOTH", WIN_CTRL, 21)
            cv.setTrackbarPos("DROP%",  WIN_CTRL, 25)
            cv.setTrackbarPos("MIN%",   WIN_CTRL, 2)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()