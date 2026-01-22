"""
Image Processing Playground (OpenCV)
-----------------------------------
Purpose:
  - Quick, interactive "first test file" to iterate on preprocessing choices.
  - Lets you visually compare: blur → edges → morphology (erode/dilate/open/close).
  - Useful before implementing full detection (circles/coins/etc.).

How to use:
  - Put images under: data/images/ (can be nested; we scan recursively)
  - Run: python src/main.py
  - Press any key to go to next image, ESC to quit.

Notes:
  - This file is intentionally verbose with comments to serve as a reference.
  - Keep this as a sandbox; production code should be cleaner / parameterized.
"""

import re
from pathlib import Path

import cv2 as cv
import numpy as np


# -----------------------------------------------------------------------------
# 0) Input discovery: recursively load images and sort them naturally by number
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"

def numeric_key(p: Path) -> int:
    """
    Sort key to get numeric ordering:
      1.jpg, 2.jpg, 10.jpg  (instead of lexicographic 1,10,2)
    If filename has no digits, put it at the end.
    """
    match = re.search(r"\d+", p.stem)
    if not match:
        raise ValueError("Expected numeric match, got None")
    return int(match.group())


# Recursively find JPGs in all subfolders.
IMAGE_PATHS = sorted(list(IMAGE_DIR.rglob("*.jpg")) + list(IMAGE_DIR.rglob("*.JPG")), key=numeric_key)


# -----------------------------------------------------------------------------
# 1) Utility: auto invert if scene is dark (polarity normalization)
# -----------------------------------------------------------------------------
def auto_invert_if_dark(img_bgr: np.ndarray, threshold: float = 110.0):
    """
    Why invert?
      Many classical CV steps behave more consistently when the "foreground"
      is bright and the background is darker. Real photos can be "polarity flipped"
      (dark scene + bright reflections), making thresholds/edges unstable.

    What we do:
      - Compute mean grayscale brightness.
      - If mean < threshold → invert with bitwise_not.

    When to use:
      - If your dataset alternates between dark and bright backgrounds.
      - If segmentation/edge steps behave inconsistently across images.

    When NOT to use:
      - If you rely on absolute brightness semantics (e.g., "dark = object").
      - If your images are already standardized / controlled lighting.

    How to choose `threshold`:
      - Print mean brightness for several images.
      - Pick a value that separates "dark scenes" from "normal scenes".
      - Typical range: 90–130 depending on camera/exposure.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    if mean_val < threshold:
        return cv.bitwise_not(img_bgr), mean_val, True
    return img_bgr, mean_val, False


# -----------------------------------------------------------------------------
# 2) Preprocessing choices: blur + edges
# -----------------------------------------------------------------------------
def bilateral_blur(gray: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """
    Bilateral filter:
      - Smooths noise / background texture while preserving edges.
      - Good when background is textured (banknotes, fabric, wood) and you still
        want sharp object boundaries.

    Parameters:
      d:
        - Neighborhood diameter (in pixels).
        - Larger d → stronger smoothing, slower.
        - Typical: 5–11 for 1–2MP images.
      sigmaColor:
        - How different intensities can be to still be smoothed together.
        - Larger → more smoothing across intensity differences (risk: blur edges).
        - Typical: 30–100.
      sigmaSpace:
        - Spatial extent of smoothing.
        - Larger → smoothing spreads farther (risk: merge close structures).
        - Typical: 30–100.

    When to prefer other blurs:
      - GaussianBlur: fastest and best when noise is mild and you don't care about
        preserving sharp edges as much.
      - medianBlur: best for salt-and-pepper noise, but can distort fine structures.
    """
    return cv.bilateralFilter(gray, d, sigma_color, sigma_space)


def canny_edges(gray_or_blur: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """
    Canny edge detector:
      - Produces a binary edge map (0/255).
      - Sensitive to noise; almost always use after a blur.

    Parameters:
      low/high:
        - low: hysteresis lower threshold; high: upper threshold.
        - If edges are missing → lower thresholds.
        - If too many edges/noise → raise thresholds and/or blur more.

    How to choose:
      - Start with (100, 200) for 8-bit images.
      - Keep high roughly 2x to 3x low as a baseline.
      - Then tune per dataset. There is no universal best pair.
    """
    return cv.Canny(gray_or_blur, low, high)


# -----------------------------------------------------------------------------
# 3) Morphology on binary images: erode / dilate / open / close
# -----------------------------------------------------------------------------
def morphology_suite(binary: np.ndarray, k: int = 5, it: int = 1):
    """
    Morphological operations must be applied to binary-ish images:
      - Canny edges (0/255)
      - threshold masks (0/255)

    Kernel choice:
      - Use MORPH_ELLIPSE for round-ish objects (coins, circles).
      - Use MORPH_RECT for boxy structures (documents, text blocks).

    Kernel size `k`:
      - Controls the "scale" of effect.
      - Small k (3–5): tiny gap filling / tiny noise removal.
      - Medium k (7–11): stronger cleanup but can merge nearby objects.

    Iterations `it`:
      - Strength multiplier; use small numbers (1–2).
      - Large iterations can destroy shape geometry.

    Operations:
      - Erode: shrinks white regions (removes tiny white noise, breaks thin links).
      - Dilate: grows white regions (bridges small gaps, thickens edges).
      - Opening (erode→dilate): removes small white specks / noise.
      - Closing (dilate→erode): fills small gaps / breaks, closes small holes.

    Practical usage:
      - On Canny edges:
          closing helps connect broken edges before findContours.
          dilation thickens edges if contours are too thin.
      - On masks:
          opening removes small blobs; closing fills holes in object regions.
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    eroded = cv.erode(binary, kernel, iterations=it)
    dilated = cv.dilate(binary, kernel, iterations=it)

    opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=it)
    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=it)

    return kernel, eroded, dilated, opened, closed


# -----------------------------------------------------------------------------
# 4) Main loop: show intermediate results for each image
# -----------------------------------------------------------------------------
if not IMAGE_PATHS:
    print(f"No JPG images found under: {IMAGE_DIR}")
    raise SystemExit

# Display settings:
# On some systems OpenCV's GUI font rendering is limited/ugly; this sandbox uses windows only.
# For a "real UI", use PySide/PyQt + OpenCV processing.
cv.namedWindow("Image Viewer", cv.WINDOW_NORMAL)
cv.namedWindow("Blur", cv.WINDOW_NORMAL)
cv.namedWindow("Edges", cv.WINDOW_NORMAL)
cv.namedWindow("Eroded", cv.WINDOW_NORMAL)
cv.namedWindow("Dilated", cv.WINDOW_NORMAL)
cv.namedWindow("Opened", cv.WINDOW_NORMAL)
cv.namedWindow("Closed", cv.WINDOW_NORMAL)

for img_path in IMAGE_PATHS:
    img = cv.imread(str(img_path))
    if img is None:
        print(f"Failed to load {img_path.name}")
        continue

    # --- Polarity normalization (optional but often helpful on mixed lighting) ---
    img_norm, mean_brightness, inverted = auto_invert_if_dark(img, threshold=110.0)

    # Grayscale for classical CV steps (edges, gradients, circle detection)
    gray = cv.cvtColor(img_norm, cv.COLOR_BGR2GRAY)

    # --- Blur choice: bilateral keeps edges sharp, suppresses background texture ---
    # How to choose params quickly:
    #   - if background texture is still causing lots of edges → increase d or sigmaSpace
    #   - if object edges start washing out → reduce sigmaColor/sigmaSpace or use smaller d
    blur = bilateral_blur(gray, d=9, sigma_color=75, sigma_space=75)

    # --- Edge detection (Canny) on blurred grayscale ---
    # How to choose params quickly:
    #   - missing object boundaries → lower low/high
    #   - too many edges everywhere → raise thresholds and/or blur more
    edges = canny_edges(blur, low=100, high=200)

    # --- Morphology suite on binary edge map ---
    # How to choose k/it:
    #   - if edges are broken into fragments → increase k (5→7) or it (1→2) for closing
    #   - if thin noise edges remain → opening with small k (3–5) can reduce speckles
    kernel, eroded, dilated, opened, closed = morphology_suite(edges, k=5, it=1)

    # --- Show results ---
    cv.imshow("Image Viewer", img_norm)
    cv.imshow("Blur", blur)
    cv.imshow("Edges", edges)
    cv.imshow("Eroded", eroded)
    cv.imshow("Dilated", dilated)
    cv.imshow("Opened", opened)
    cv.imshow("Closed", closed)

    print(
        f"Showing: {img_path.name} | mean={mean_brightness:.1f} | inverted={inverted} | "
        f"bilateral(d=9, sc=75, ss=75) | canny(100,200) | morph(k=5,it=1)"
    )

    # Controls:
    #   - Any key: next image
    #   - ESC: quit
    key = cv.waitKey(0) & 0xFF
    if key == 27:
        break

cv.destroyAllWindows()