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
# WATERSHED
# =========================
def watershed_separation(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    # mask: 0/255
    # img: BGR image

    # 1. distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 2. sure foreground
    _, sure_fg = cv2.threshold(
        dist, 0.4, 1.0, cv2.THRESH_BINARY
    )
    sure_fg = (sure_fg * 255).astype(np.uint8)

    # 3. sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)

    # 4. unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6. watershed
    markers = cv2.watershed(img, markers)

    # 7. build separated mask
    separated = np.zeros_like(mask)
    separated[markers > 1] = 255

    return separated


# =========================
# CONTOURS + CIRCLES
# =========================
def contours_to_min_circles(mask: np.ndarray):
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = mask.shape
    min_area = 0.001 * h * w   # мягкий порог

    circles = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        circles.append((int(x), int(y), int(r)))

    return circles

# =========================
# PREPROCESSING
# =========================
def preprocess_image(img: np.ndarray) -> np.ndarray:
    # 1. лёгкое шумоподавление
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. CLAHE по яркости
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[..., 2] = clahe.apply(hsv[..., 2])

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

# =========================
# MORPHOLOGY
# =========================
def open_close_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    # 1. opening — убрать мелкий шум
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. closing — закрыть дырки и разрывы
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# =========================
# FILL CONTOURS
# =========================
def fill_mask_from_contours(mask: np.ndarray) -> np.ndarray:
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled


# =========================
# MAIN (with navigation)
# =========================
images = load_images()
idx = 0

while True:
    img = cv2.imread(str(images[idx]))
    img = resize_keep_aspect(img, MAX_HEIGHT)
    img = preprocess_image(img)   # ← ВАЖНОЕ МЕСТО

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    mask = foreground_mask(hsv)
    mask = open_close_mask(mask)   # ← ВАЖНО
    mask = fill_mask_from_contours(mask)  
    mask = watershed_separation(mask, img) 
    clusters = cluster_foreground(mask, hsv)
    
    # contours = find_contours_from_mask(mask)

    # vis = img.copy()
    # cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    circles = contours_to_min_circles(mask)

    vis = img.copy()
    for (x, y, r) in circles:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)  # центр


    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Clusters", clusters)
    cv2.imshow("Contours", vis)

    key = cv2.waitKey(0) & 0xFF

    if key in (27, ord('q')):          # ESC / q
        break
    elif key in (ord('d'), 83):        # d / →
        idx = (idx + 1) % len(images)
    elif key in (ord('a'), 81):        # a / ←
        idx = (idx - 1) % len(images)

cv2.destroyAllWindows()
