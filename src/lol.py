#!/usr/bin/env python3
"""
Minimal OpenCV image viewer + color quantization demos (BGR/HSV uniform, k-means, bit-depth).

Keys:
  n / Right Arrow / Space : next image
  p / Left Arrow          : previous image
  r                       : rescan folder
  q / ESC                 : quit

Quantization:
  1 : toggle Uniform BGR (levels)
  2 : toggle Uniform HSV (levels)
  3 : toggle KMeans (K colors)
  4 : toggle Bit-depth (bits)
  [ / ] : decrease / increase main quant parameter (levels or K or bits)
  0 : disable quantization (show original)

Notes:
- HSV quantization uses OpenCV HSV: H in [0..179], S,V in [0..255]
- KMeans can be slow on large images; script auto-samples pixels for kmeans fit.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

import cv2 as cv
import numpy as np


# ---------- quantization methods ----------
def quantize_uniform_bgr(img_bgr: np.ndarray, levels: int = 8) -> np.ndarray:
    levels = int(levels)
    levels = max(2, min(256, levels))
    step = max(1, 256 // levels)
    q = (img_bgr // step) * step
    return q.astype(np.uint8)


def quantize_uniform_hsv(img_bgr: np.ndarray, levels: int = 8) -> np.ndarray:
    levels = int(levels)
    levels = max(2, min(256, levels))

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    h_step = max(1, 180 // levels)   # H: 0..179
    sv_step = max(1, 256 // levels)  # S,V: 0..255

    hsv_q = hsv.copy()
    hsv_q[..., 0] = (hsv_q[..., 0] // h_step) * h_step
    hsv_q[..., 1] = (hsv_q[..., 1] // sv_step) * sv_step
    hsv_q[..., 2] = (hsv_q[..., 2] // sv_step) * sv_step

    return cv.cvtColor(hsv_q, cv.COLOR_HSV2BGR)


def quantize_bits(img_bgr: np.ndarray, bits: int = 4) -> np.ndarray:
    bits = int(bits)
    bits = max(1, min(8, bits))
    shift = 8 - bits
    q = (img_bgr >> shift) << shift
    return q.astype(np.uint8)


def quantize_kmeans_bgr(
    img_bgr: np.ndarray,
    K: int = 8,
    max_iter: int = 25,
    attempts: int = 2,
    sample_n: int = 60000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (quantized_img, centers_bgr[K,3]).
    Uses sampling for fitting; then assigns all pixels to nearest center.
    """
    K = int(K)
    K = max(2, min(64, K))

    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    n = Z.shape[0]

    rng = np.random.default_rng(seed)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        Zfit = Z[idx]
    else:
        Zfit = Z

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, 1e-3)
    _compactness, _labels, centers = cv.kmeans(
        Zfit, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)  # Kx3

    # assign all pixels to nearest center (vectorized)
    # distances: (n,K)
    d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels_all = d2.argmin(axis=1).astype(np.int32)

    out = centers[labels_all].reshape(img_bgr.shape).astype(np.uint8)
    return out, centers.astype(np.uint8)


def palette_and_counts(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = img_bgr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    return colors[order], counts[order]


# ---------- viewer helpers ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"
WIN = "viewer"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(p: Path):
    m = re.search(r"\d+", p.stem)
    if not m:
        return (10**12, p.name.lower())
    return (int(m.group()), p.name.lower())


def iter_images_recursive(root: Path):
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(paths, key=natural_key)


def annotate(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv.putText(
        out,
        text,
        (12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        out,
        text,
        (12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return out

def show_histogram(gray: np.ndarray, win_name: str = "Histogram"):
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv.normalize(hist, hist).flatten()

    h, w = 200, 256
    hist_img = np.zeros((h, w), dtype=np.uint8)

    for x in range(256):
        cv.line(
            hist_img,
            (x, h),
            (x, h - int(hist[x] * h)),
            255,
            1,
        )

    cv.imshow(win_name, hist_img)

def show(img_path: Path, q_mode: str, param: int):
    img = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read: {img_path}")
        return False
    
    #img = quantize_uniform_hsv(img, 8)

    # --- your existing pipeline (mask preview) ---
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    show_histogram(gray)
    blur = cv.bilateralFilter(gray, 7, 75, 100)

    contour = cv.Canny(blur, 50, 100)

    kernel_size = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #dilation = cv.dilate(contour,kernel,iterations = 3)
    opening = cv.morphologyEx(contour, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        opening,
        cv.RETR_EXTERNAL,      # only outer contours
        cv.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(opening)

    # Fill all contours
    cv.drawContours(
        filled,
        contours,
        contourIdx=-1,         # all contours
        color=255,
        thickness=cv.FILLED
    )




    # windows
    title = f"{WIN} - {img_path.name}"
    cv.setWindowTitle(WIN, title)

    cv.imshow(WIN, annotate(img, "Original"))
    # cv.imshow("Grayscale", gray)
    # cv.imshow("Blur", blur)
    # cv.imshow("Canny", contour)
    cv.imshow("Morphology", opening)
    cv.imshow("filled", filled)
    
    return True


def main():
    root = IN_DIR
    print(f"[INFO] Root: {root}")

    paths = iter_images_recursive(root)
    if not paths:
        print("[ERROR] No images found.")
        return

    # quantization state
    q_mode = "u_bgr"     # none | u_bgr | u_hsv | kmeans | bits
    levels = 8          # for uniform
    K = 8               # for kmeans
    bits = 4            # for bit-depth

    i = 0
    cv.namedWindow(WIN, cv.WINDOW_NORMAL)

    while True:
        # choose current param depending on mode
        if q_mode in ("u_bgr", "u_hsv"):
            param = levels
        elif q_mode == "kmeans":
            param = K
        elif q_mode == "bits":
            param = bits
        else:
            param = 0

        if not show(paths[i], q_mode, param):
            i = (i + 1) % len(paths)
            continue

        print(f"[INFO] ({i+1}/{len(paths)}) {paths[i]}")
        print(f"[INFO] mode={q_mode} | levels={levels} | K={K} | bits={bits}")
        key = cv.waitKeyEx(0)

        # quit
        if key in (ord("q"), 27):
            break

        # next / prev
        if key in (ord("n"), ord(" "), 2555904):  # Right
            i = (i + 1) % len(paths)
            continue
        if key in (ord("p"), 2424832):            # Left
            i = (i - 1) % len(paths)
            continue

        # rescan
        if key == ord("r"):
            paths = iter_images_recursive(root)
            if not paths:
                print("[ERROR] No images found after rescan.")
                break
            i %= len(paths)
            print(f"[INFO] Rescanned. {len(paths)} images.")
            continue

        # quantization mode toggles
        if key == ord("0"):
            q_mode = "none"
            continue
        if key == ord("1"):
            q_mode = "u_bgr"
            continue
        if key == ord("2"):
            q_mode = "u_hsv"
            continue
        if key == ord("3"):
            q_mode = "kmeans"
            continue
        if key == ord("4"):
            q_mode = "bits"
            continue

        # adjust parameter with [ and ]
        if key in (ord("["),):
            if q_mode in ("u_bgr", "u_hsv"):
                levels = max(2, levels - 1)
            elif q_mode == "kmeans":
                K = max(2, K - 1)
            elif q_mode == "bits":
                bits = max(1, bits - 1)
            continue

        if key in (ord("]"),):
            if q_mode in ("u_bgr", "u_hsv"):
                levels = min(64, levels + 1)
            elif q_mode == "kmeans":
                K = min(32, K + 1)
            elif q_mode == "bits":
                bits = min(8, bits + 1)
            continue

        print(f"[INFO] Key code: {key} (unmapped)")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
