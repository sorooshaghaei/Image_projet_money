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


def load_image() -> np.ndarray:
    root = find_project_root()
    images = sorted(
        p for p in (root / "data" / "images").rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError("No images found")

    img = cv2.imread(str(images[11]))
    if img is None:
        raise RuntimeError("Failed to load image")

    return img


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


def edge_map(labels: np.ndarray) -> np.ndarray:
    e = np.zeros_like(labels)
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    e[1:, :] |= labels[1:, :] != labels[:-1, :]
    return (e * 255).astype(np.uint8)


# =========================
# MAIN
# =========================
img = resize_keep_aspect(load_image(), MAX_HEIGHT)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

mask = foreground_mask(hsv)
clusters = cluster_foreground(mask, hsv)
edges = edge_map(clusters)

cv2.imshow("Image", img)
cv2.imshow("Mask", mask)
cv2.imshow("Clusters", clusters)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
