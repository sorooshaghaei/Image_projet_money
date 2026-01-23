import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple


class ImageRepository:
    def __init__(
        self,
        image_dir: Path,
        patterns: Tuple[str, ...] = (
            "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"
        ),
    ):
        self.image_dir = image_dir
        self.patterns = patterns

    def list_images(self) -> List[Path]:
        paths: List[Path] = []
        for p in self.patterns:
            paths.extend(self.image_dir.rglob(p))
        return sorted(paths)


def kmeans_labels_lab_hue(img_bgr: np.ndarray, K: int, sample_n: int = 80000, attempts: int = 2):
    h, w = img_bgr.shape[:2]
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # features: Lab + Hue (float32)
    X = np.stack(
        [
            lab[:, :, 0].astype(np.float32) / 255.0,
            lab[:, :, 1].astype(np.float32) / 255.0,
            lab[:, :, 2].astype(np.float32) / 255.0,
            hsv[:, :, 0].astype(np.float32) / 179.0,
        ],
        axis=2,
    ).reshape(-1, 4)

    # fit sample
    N = X.shape[0]
    if N > sample_n:
        idx = np.random.choice(N, sample_n, replace=False)
        Xfit = X[idx]
    else:
        Xfit = X

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    _, _, centers = cv.kmeans(Xfit, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)

    # assign all pixels (batch)
    labels = np.empty((N,), dtype=np.int32)
    batch = 250000
    for s in range(0, N, batch):
        e = min(N, s + batch)
        Xi = X[s:e]  # (m,4)
        d = Xi[:, None, :] - centers[None, :, :]  # (m,K,4)
        dist2 = np.sum(d * d, axis=2)  # (m,K)
        labels[s:e] = np.argmin(dist2, axis=1)

    return labels.reshape(h, w), centers


def palette_bgr(n: int):
    # distinct-ish colors (BGR)
    base = [
        (0, 0, 255),     # red
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (0, 128, 255),   # orange
        (255, 128, 0),   # light blue-ish
        (128, 0, 255),   # purple
        (0, 255, 128),   # spring green
        (128, 255, 0),   # lime
        (255, 0, 128),   # pink
        (180, 180, 180), # gray
        (80, 80, 80),    # dark gray
        (255, 255, 255), # white
    ]
    if n <= len(base):
        return np.array(base[:n], dtype=np.uint8)

    # extend deterministically
    pal = []
    for i in range(n):
        pal.append(((37 * i) % 256, (97 * i) % 256, (181 * i) % 256))
    return np.array(pal, dtype=np.uint8)


def make_cluster_viz(img_bgr: np.ndarray, labels_hw: np.ndarray, K: int, alpha_overlay: float = 0.55):
    h, w = img_bgr.shape[:2]
    pal = palette_bgr(K)  # (K,3) BGR
    color_map = pal[labels_hw.reshape(-1)].reshape(h, w, 3)
    overlay = cv.addWeighted(img_bgr, 1.0 - alpha_overlay, color_map, alpha_overlay, 0)
    return color_map, overlay


def make_cluster_mask(labels_hw: np.ndarray, k: int):
    return ((labels_hw == k).astype(np.uint8) * 255)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    repo = ImageRepository(root)

    # UI
    WIN = "KMEANS CLUSTERS"
    cv.namedWindow(WIN, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN, 1500, 900)

    cv.namedWindow("cluster_mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("cluster_mask", 700, 700)

    def nothing(_): pass

    cv.createTrackbar("K", WIN, 6, 14, nothing)
    cv.createTrackbar("cluster_id", WIN, 0, 13, nothing)
    cv.createTrackbar("overlay_alpha_%", WIN, 55, 100, nothing)

    paths = repo.list_images()
    if not paths:
        raise SystemExit(f"No images found under: {root}")

    idx = 0
    last_path = None
    lastK = None
    labels = None

    print("Controls:")
    print("  n = next image")
    print("  p = previous image")
    print("  q / ESC = quit")
    print("Trackbars: K, cluster_id, overlay alpha")
    print()

    while True:
        path = paths[idx]
        img = cv.imread(str(path))
        if img is None:
            print(f"Skip unreadable: {path}")
            idx = (idx + 1) % len(paths)
            continue

        K = max(2, cv.getTrackbarPos("K", WIN))
        cluster_id = cv.getTrackbarPos("cluster_id", WIN)
        if cluster_id >= K:
            cluster_id = K - 1

        alpha = cv.getTrackbarPos("overlay_alpha_%", WIN) / 100.0
        alpha = min(0.95, max(0.05, alpha))

        # recompute if image changed or K changed
        if last_path != path or lastK != K or labels is None:
            labels, centers = kmeans_labels_lab_hue(img, K=K, sample_n=80000, attempts=2)
            last_path = path
            lastK = K
            cv.setTrackbarPos("cluster_id", WIN, 0)
            cluster_id = 0

        color_map, overlay = make_cluster_viz(img, labels, K, alpha_overlay=alpha)
        mask = make_cluster_mask(labels, cluster_id)

        # also show "cluster-only image": keep original pixels where label==cluster_id else black
        cluster_only = np.zeros_like(img)
        cluster_only[labels == cluster_id] = img[labels == cluster_id]

        # montage: [overlay | color_map | cluster_only]
        top = np.hstack([overlay, color_map, cluster_only])

        cv.putText(
            top,
            f"{path.name}   K={K}   cluster_id={cluster_id}   alpha={alpha:.2f}",
            (12, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.imshow(WIN, top)
        cv.imshow("cluster_mask", mask)

        key = cv.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("n"):
            idx = (idx + 1) % len(paths)
            labels = None
        if key == ord("p"):
            idx = (idx - 1) % len(paths)
            labels = None

    cv.destroyAllWindows()
