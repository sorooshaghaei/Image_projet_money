from pathlib import Path
import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"

WIN_VIEW = "viewer"
WIN_HIST = "hist"


def iter_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def draw_header(img, text: str):
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)


def render_histogram_image(h_counts, w=1024, h=320):
    """
    h_counts: length 256, raw counts or probabilities
    Draws only the histogram curve + axes + tick labels.
    """
    # normalize to 0..1 for plotting
    h_counts = h_counts.astype(np.float32).ravel()
    if h_counts.sum() > 0:
        h_norm = h_counts / (h_counts.max() + 1e-12)
    else:
        h_norm = h_counts

    img = np.full((h, w, 3), 18, np.uint8)

    left = 40
    top = 10
    bottom = h - 30
    right = w - 10

    # axes
    cv.line(img, (left, bottom), (right, bottom), (80, 80, 80), 1, cv.LINE_AA)
    cv.line(img, (left, top), (left, bottom), (80, 80, 80), 1, cv.LINE_AA)

    plot_w = right - left
    plot_h = bottom - top

    def x_of(bin_idx):
        return left + int(bin_idx * plot_w / 255)

    def y_of(val01):
        return top + int((1.0 - float(val01)) * plot_h)

    # polyline
    pts = [(x_of(b), y_of(h_norm[b])) for b in range(256)]
    for i in range(255):
        cv.line(img, pts[i], pts[i + 1], (230, 230, 230), 1, cv.LINE_AA)

    # x ticks
    for b in (0, 64, 128, 192, 255):
        x = x_of(b)
        cv.line(img, (x, bottom), (x, bottom + 4), (120, 120, 120), 1, cv.LINE_AA)
        cv.putText(img, str(b), (x - 10, h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv.LINE_AA)

    # y label: max count
    maxc = float(h_counts.max()) if h_counts.size else 0.0
    cv.putText(img, f"max={maxc:.0f}", (left + 8, top + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv.LINE_AA)

    return img


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

    idx = 0
    cached = None
    cached_path = None

    while True:
        p = paths[idx]

        if cached is None or cached_path != p:
            img = cv.imread(str(p), cv.IMREAD_COLOR)
            cached = img
            cached_path = p
        else:
            img = cached

        if img is None:
            print(f"[skip] failed to read: {p}")
            idx = (idx + 1) % len(paths)
            cached = None
            cached_path = None
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # raw histogram counts (256 bins)
        h_counts = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()

        hist_img = render_histogram_image(h_counts, w=1024, h=320)

        vis = img.copy()
        header = f"[{idx+1}/{len(paths)}] {p.relative_to(PROJECT_ROOT)} | keys: n/p next/prev, q quit"
        draw_header(vis, header)

        cv.imshow(WIN_VIEW, vis)
        cv.imshow(WIN_HIST, hist_img)

        key = cv.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        elif key in (ord("n"), ord(" ")):
            idx = (idx + 1) % len(paths)
            cached = None
            cached_path = None
        elif key == ord("p"):
            idx = (idx - 1) % len(paths)
            cached = None
            cached_path = None

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()