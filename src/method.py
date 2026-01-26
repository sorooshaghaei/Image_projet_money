import cv2
import numpy as np
from pathlib import Path


# =========================
# CONFIG
# =========================
MAX_HEIGHT = 480
BORDER = 30
K = 4
RNG_SEED = 0


# =========================
# UTILITIES
# =========================
def find_project_root() -> Path:
    p = Path.cwd()
    while not (p / "data").exists():
        if p.parent == p:
            raise RuntimeError("Project root not found")
        p = p.parent
    return p


def load_images():
    root = find_project_root()
    images = sorted(
        p for p in (root / "data" / "images").rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError("No images found")
    return images


def resize_keep_aspect(img: np.ndarray, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / h
    return cv2.resize(img, (int(w * scale), max_h), cv2.INTER_AREA)


def background_stats(hsv: np.ndarray, border: int):
    b = border
    samples = np.concatenate([
        hsv[:b].reshape(-1, 3),
        hsv[-b:].reshape(-1, 3),
        hsv[:, :b].reshape(-1, 3),
        hsv[:, -b:].reshape(-1, 3),
    ])
    return samples.mean(0), samples.std(0)


def foreground_mask(hsv: np.ndarray) -> np.ndarray:
    mean, _ = background_stats(hsv, BORDER)
    diff = np.abs(hsv - mean)

    score = (
        0.3 * diff[..., 0] +
        0.5 * diff[..., 1] +
        0.8 * diff[..., 2]
    )

    mask = score > 1.2 * score.mean()
    return (mask * 255).astype(np.uint8)


def kmeans_numpy(X: np.ndarray, k: int, iters: int = 30):
    rng = np.random.default_rng(RNG_SEED)
    centers = X[rng.choice(len(X), k, replace=False)]

    for _ in range(iters):
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = d.argmin(axis=1)

        new = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])

        if np.allclose(centers, new):
            break
        centers = new

    return labels


def cluster_foreground(mask, hsv):
    ys, xs = np.where(mask == 255)
    h, w = mask.shape

    features = np.column_stack([
        hsv[ys, xs, 0],
        hsv[ys, xs, 1],
        hsv[ys, xs, 2],
        xs / w,
        ys / h,
    ])

    labels = kmeans_numpy(features, K)

    out = np.zeros_like(mask)
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[y, x] = (labels[i] + 1) * (255 // K)

    return out

# =========================
# CONTOURS
# =========================
def find_contours_from_mask(mask: np.ndarray):
    # бинаризация на всякий случай (mask уже 0/255)
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # очень мягкая фильтрация по площади
    h, w = mask.shape
    min_area = 0.001 * h * w

    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# =========================
# MAIN (with navigation)
# =========================
images = load_images()
idx = 0

while True:
    img = cv2.imread(str(images[idx]))
    img = resize_keep_aspect(img, MAX_HEIGHT)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    mask = foreground_mask(hsv)
    clusters = cluster_foreground(mask, hsv)
    
    contours = find_contours_from_mask(mask)

    vis = img.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Contours", vis)


    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Clusters", clusters)
    cv2.imshow("Contours", vis)

    key = cv2.waitKey(0) & 0xFF

    if key in (27, ord('q')):          # ESC / q
        break
    elif key in (ord('d'), 83):        # d / →
        idx = (idx + 1) % len(images)
    elif key in (ord('a'), 81):        # a / ←
        idx = (idx - 1) % len(images)

cv2.destroyAllWindows()
