#!/usr/bin/env python3
# src/coin_segment_border_ws_sliders.py
#
# Same pipeline as before, but with REAL-TIME sliders (OpenCV trackbars).
# You can tune:
#   - K (k-means clusters)
#   - betaXY (spatial regularization)
#   - border threshold (background via border prior)
#   - label median (salt-pepper removal)
#   - close kernel/iters (fg cleanup)
#   - DT fore frac (watershed seeding)
#   - final gates: min circularity / min edge agree / min ring contrast
#
# Project layout:
#   project_root/
#     data/images/
#     src/
#
# Run:
#   python3 src/coin_segment_border_ws_sliders.py
#
# Keys:
#   n / SPACE : next image
#   p         : previous image
#   r         : recompute now
#   d         : toggle debug windows
#   q / ESC   : quit
#
from pathlib import Path
import math
import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"

WIN_OVERLAY = "coin | overlay"
WIN_LABELS  = "coin | labels"
WIN_FG      = "coin | fg_mask"
WIN_DT      = "coin | dist"
WIN_WS      = "coin | watershed"
WIN_CTRL    = "coin | controls"

# Fixed knobs (not sliders)
SCALE = 0.85
SAMPLE_N = 70000
ATTEMPTS = 2
MAX_ITERS = 45
EPS = 1e-3
ALPHA_S = 1.0
BORDER_RING = 2
OPEN_K = 3
WATERSHED_EDGES_BOOST = True
DT_BLUR_K = 5
MAX_AXIS_RATIO = 2.8
MIN_AREA_FRAC = 0.0005
MAX_AREA_FRAC = 0.45


def iter_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def resize_keep(img, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)


def ellipse_kernel(k: int):
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))


def draw_text(img, text, x=12, y=30, scale=0.9):
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)


def compute_edges(bgr: np.ndarray):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv.Canny(gray, lo, hi, L2gradient=True)


def circularity(area, perim):
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def boundary_u8(mask_u8: np.ndarray):
    dil = cv.dilate(mask_u8, ellipse_kernel(3), 1)
    ero = cv.erode(mask_u8, ellipse_kernel(3), 1)
    return cv.subtract(dil, ero)


def edge_agreement(mask_u8: np.ndarray, edges_u8: np.ndarray):
    bd = boundary_u8(mask_u8)
    bd_n = cv.countNonZero(bd)
    if bd_n == 0:
        return 0.0
    edges_d = cv.dilate(edges_u8, ellipse_kernel(5), 1)
    ov = cv.bitwise_and(bd, edges_d)
    return float(cv.countNonZero(ov) / bd_n)


def ring_contrast(bgr: np.ndarray, mask_u8: np.ndarray):
    h, w = mask_u8.shape[:2]
    k = ellipse_kernel(max(5, int(0.01 * min(h, w)) | 1))
    inner = cv.erode(mask_u8, k, 1)
    outer = cv.dilate(mask_u8, k, 1)
    ring = cv.subtract(outer, mask_u8)
    if cv.countNonZero(inner) < 50 or cv.countNonZero(ring) < 50:
        return 0.0
    mi = np.array(cv.mean(bgr, mask=inner)[:3], dtype=np.float32)
    mo = np.array(cv.mean(bgr, mask=ring)[:3], dtype=np.float32)
    return float(np.linalg.norm(mi - mo) / 255.0)


def labels_viz(labels_hw: np.ndarray, K_: int):
    rng = np.random.default_rng(1000 + K_)
    colors = rng.integers(0, 255, size=(K_, 3), dtype=np.uint8)
    vis = colors[labels_hw]
    vis[::40, :] = (255, 255, 255)
    vis[:, ::40] = (255, 255, 255)
    return vis


def build_features(bgr: np.ndarray, alpha_s: float, beta_xy: float):
    h, w = bgr.shape[:2]
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)

    a = (lab[:, :, 1] / 255.0).astype(np.float32)
    b = (lab[:, :, 2] / 255.0).astype(np.float32)
    s = (hsv[:, :, 1] / 255.0).astype(np.float32)

    feats = [a, b, (alpha_s * s)]

    if beta_xy > 1e-9:
        xs = (np.tile(np.arange(w, dtype=np.float32), (h, 1)) / max(1.0, float(w - 1)))
        ys = (np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) / max(1.0, float(h - 1)))
        feats.append(beta_xy * xs)
        feats.append(beta_xy * ys)

    F = np.dstack(feats).astype(np.float32)
    return F.reshape(-1, F.shape[2])  # N x D


def kmeans_centers_sample(X: np.ndarray, K_: int, sample_n: int, seed: int):
    X = X.astype(np.float32)
    N = X.shape[0]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, MAX_ITERS, EPS)
    flags = cv.KMEANS_PP_CENTERS

    if N <= sample_n:
        _, _, centers = cv.kmeans(X, K_, None, criteria, ATTEMPTS, flags)
        return centers.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=sample_n, replace=False)
    Xs = X[idx]
    _, _, centers = cv.kmeans(Xs, K_, None, criteria, ATTEMPTS, flags)
    return centers.astype(np.float32)


def assign_labels(X: np.ndarray, centers: np.ndarray):
    X = X.astype(np.float32)
    C = centers.astype(np.float32)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C * C, axis=1, keepdims=True).T
    d2 = X2 + C2 - 2.0 * (X @ C.T)
    return np.argmin(d2, axis=1).astype(np.int32)


def border_pixels(labels_hw: np.ndarray, ring: int):
    h, w = labels_hw.shape
    r = max(1, int(ring))
    r = min(r, min(h // 4, w // 4, 8)) if min(h, w) >= 8 else 1
    top = labels_hw[0:r, :].reshape(-1)
    bot = labels_hw[h - r:h, :].reshape(-1)
    lef = labels_hw[:, 0:r].reshape(-1)
    rig = labels_hw[:, w - r:w].reshape(-1)
    return np.concatenate([top, bot, lef, rig], axis=0)


def select_bg_clusters_border(labels_hw: np.ndarray, K_: int, border_thr: float, border_ring: int):
    h, w = labels_hw.shape
    all_counts = np.bincount(labels_hw.reshape(-1), minlength=K_)
    fracs = all_counts.astype(np.float32) / float(h * w)

    b = border_pixels(labels_hw, border_ring)
    bcounts = np.bincount(b.astype(np.int32), minlength=K_)
    bfracs = bcounts.astype(np.float32) / float(max(1, b.size))

    bg = set()
    bg.add(int(np.argmax(bcounts)))
    for k in range(K_):
        if float(bfracs[k]) >= border_thr:
            bg.add(k)
    bg.add(int(np.argmax(all_counts)))  # safety for solid backgrounds

    return bg, fracs, bfracs


def clean_mask(mask_u8: np.ndarray, close_k: int, close_iters: int):
    m = mask_u8.copy()
    m = cv.morphologyEx(m, cv.MORPH_OPEN, ellipse_kernel(OPEN_K), iterations=1)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, ellipse_kernel(close_k), iterations=close_iters)
    return m


def watershed_split(bgr: np.ndarray, fg_mask_u8: np.ndarray, edges_u8: np.ndarray, dt_fore_frac: float):
    fg = (fg_mask_u8 > 0).astype(np.uint8)

    sure_bg = cv.dilate(fg, ellipse_kernel(7), iterations=2)

    dist = cv.distanceTransform(fg, cv.DIST_L2, 5)
    if DT_BLUR_K >= 3 and DT_BLUR_K % 2 == 1:
        dist = cv.GaussianBlur(dist, (DT_BLUR_K, DT_BLUR_K), 0)

    dmax = float(dist.max()) if dist.size else 0.0
    thr = dt_fore_frac * dmax if dmax > 1e-6 else 0.0
    sure_fg = (dist > thr).astype(np.uint8) * 255

    if WATERSHED_EDGES_BOOST and edges_u8 is not None:
        e = cv.dilate(edges_u8, ellipse_kernel(3), 1)
        # safer than killing seeds: increase unknown where edges are
        unknown = cv.subtract((sure_bg.astype(np.uint8) * 255), sure_fg)
        unknown = cv.bitwise_or(unknown, e)
    else:
        unknown = cv.subtract((sure_bg.astype(np.uint8) * 255), sure_fg)

    n, markers = cv.connectedComponents((sure_fg > 0).astype(np.uint8), connectivity=8)
    markers = markers.astype(np.int32)
    markers += 1
    markers[unknown > 0] = 0

    ws_in = bgr.copy()
    cv.watershed(ws_in, markers)
    return markers, dist


# -----------------------
# Slider mapping
# -----------------------
def tb_get_int(name: str) -> int:
    return cv.getTrackbarPos(name, WIN_CTRL)


def tb_get_float(name: str, div: float) -> float:
    return tb_get_int(name) / div


def make_odd(x: int, min_v: int = 1, max_v: int = 31) -> int:
    x = max(min_v, min(max_v, int(x)))
    if x % 2 == 0:
        x += 1
    return min(max_v, x)


# -----------------------
# Pipeline with caching:
# recompute kmeans labels only when K/betaXY changes
# -----------------------
class Cache:
    def __init__(self):
        self.key = None
        self.img = None
        self.img_blur = None
        self.edges = None
        self.labels = None
        self.centers = None
        self.K = None
        self.beta_xy = None


def compute_labels_cached(cache: Cache, bgr0: np.ndarray, K_: int, beta_xy: float):
    img = resize_keep(bgr0, SCALE)
    img_blur = cv.bilateralFilter(img, 7, 50, 50)
    edges = compute_edges(img_blur)

    need = (cache.img is None) or (cache.K != K_) or (abs((cache.beta_xy or 0.0) - beta_xy) > 1e-6)
    if need:
        X = build_features(img_blur, alpha_s=ALPHA_S, beta_xy=beta_xy)
        centers = kmeans_centers_sample(X, K_, SAMPLE_N, seed=123 + K_)
        labels = assign_labels(X, centers).reshape(img.shape[0], img.shape[1])
        cache.img = img
        cache.img_blur = img_blur
        cache.edges = edges
        cache.labels = labels
        cache.centers = centers
        cache.K = K_
        cache.beta_xy = beta_xy
    else:
        # image still changes when navigating; we handle that externally by resetting cache.img=None
        cache.edges = edges
        cache.img = img
        cache.img_blur = img_blur

    return cache.img, cache.img_blur, cache.edges, cache.labels


def detect_from_labels(img: np.ndarray, img_blur: np.ndarray, edges: np.ndarray, labels: np.ndarray,
                       K_: int,
                       border_thr: float,
                       label_median: int,
                       close_k: int,
                       close_iters: int,
                       dt_fore_frac: float,
                       min_circ: float,
                       min_edge: float,
                       min_ring: float):
    h, w = labels.shape[:2]

    # label smoothing
    labels_sm = labels.copy()
    if label_median >= 3 and label_median % 2 == 1 and K_ <= 255:
        labels_sm = cv.medianBlur(labels_sm.astype(np.uint8), label_median).astype(np.int32)

    bg_set, fracs, bfracs = select_bg_clusters_border(labels_sm, K_, border_thr, BORDER_RING)

    fg = np.ones((h, w), np.uint8) * 255
    for k in bg_set:
        fg[labels_sm == k] = 0

    # safety: if fg nearly empty -> too aggressive bg
    fg_frac = float(cv.countNonZero(fg)) / float(h * w)
    if fg_frac < 0.02:
        b = border_pixels(labels_sm, BORDER_RING)
        bcounts = np.bincount(b.astype(np.int32), minlength=K_)
        bg_set = {int(np.argmax(bcounts))}
        fg[:] = 255
        for k in bg_set:
            fg[labels_sm == k] = 0

    fg = clean_mask(fg, close_k=close_k, close_iters=close_iters)

    markers, dist = watershed_split(img_blur, fg, edges, dt_fore_frac=dt_fore_frac)

    # instances
    total = h * w
    min_area = max(80, int(MIN_AREA_FRAC * total))
    max_area = int(MAX_AREA_FRAC * total)

    overlay = img.copy()
    coins = []

    max_label = int(markers.max())
    for lab in range(2, max_label + 1):
        mask = (markers == lab).astype(np.uint8) * 255
        area_px = int(cv.countNonZero(mask))
        if area_px < min_area or area_px > max_area:
            continue

        cnts, _ = cv.findContours((mask > 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv.contourArea)
        if len(cnt) < 5:
            continue

        a = float(cv.contourArea(cnt))
        p = float(cv.arcLength(cnt, True))
        circ = circularity(a, p)
        if circ < min_circ:
            continue

        x, y, ww, hh = cv.boundingRect(cnt)
        aspect = float(ww) / float(hh) if hh > 0 else 999.0
        aspect = max(aspect, 1.0 / max(1e-6, aspect))

        axis_ratio = None
        if len(cnt) >= 20:
            try:
                ell = cv.fitEllipse(cnt)
                (_, _), (ma, Mi), _ = ell
                A = max(ma, Mi) / 2.0
                B = min(ma, Mi) / 2.0
                axis_ratio = float(A / max(1e-6, B))
            except cv.error:
                axis_ratio = None
        if axis_ratio is not None and axis_ratio > MAX_AXIS_RATIO:
            continue

        ea = edge_agreement(mask, edges)
        if ea < min_edge:
            continue

        rc = ring_contrast(img_blur, mask)
        if rc < min_ring:
            continue

        score = 0.0
        score += 2.0 * circ
        score += 1.2 * ea
        score += 0.8 * min(1.0, rc * 2.0)
        score -= 0.7 * min(2.0, (aspect - 1.0))
        if axis_ratio is not None:
            score -= 0.4 * min(2.0, (axis_ratio - 1.0))

        coins.append((score, cnt, (x, y, ww, hh), circ, ea, rc))

    coins.sort(key=lambda t: t[0], reverse=True)

    for i, (score, cnt, bbox, circ, ea, rc) in enumerate(coins):
        x, y, ww, hh = bbox
        cv.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        cv.rectangle(overlay, (x, y), (x + ww, y + hh), (255, 0, 0), 1)
        txt = f"#{i+1} s={score:.2f} C={circ:.2f} e={ea:.2f} r={rc:.2f}"
        cv.putText(overlay, txt, (x, max(0, y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(overlay, txt, (x, max(0, y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

    bg_border_frac = float(sum(float(bfracs[k]) for k in bg_set))
    header = f"K={K_} betaXY={tb_get_int('betaXY')/100:.2f} med={label_median} bgK={len(bg_set)} bgBorder={bg_border_frac:.2f} coins={len(coins)}"
    draw_text(overlay, header)
    return overlay, labels_viz(labels_sm, K_), fg, dist, markers, len(coins)


def make_controls():
    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_CTRL, 520, 420)

    def noop(_=None):
        pass

    # int sliders
    cv.createTrackbar("K", WIN_CTRL, 8, 20, noop)                     # 2..20
    cv.createTrackbar("betaXY", WIN_CTRL, 12, 50, noop)               # /100 => 0.00..0.50
    cv.createTrackbar("borderThr", WIN_CTRL, 6, 30, noop)             # /100 => 0.00..0.30
    cv.createTrackbar("median", WIN_CTRL, 5, 15, noop)                # will be made odd
    cv.createTrackbar("closeK", WIN_CTRL, 11, 31, noop)               # odd
    cv.createTrackbar("closeIt", WIN_CTRL, 3, 6, noop)                # 1..6
    cv.createTrackbar("DT_fore", WIN_CTRL, 32, 70, noop)              # /100 => 0.00..0.70
    cv.createTrackbar("minCirc", WIN_CTRL, 48, 90, noop)              # /100 => 0.00..0.90
    cv.createTrackbar("minEdge", WIN_CTRL, 8, 40, noop)               # /100 => 0.00..0.40
    cv.createTrackbar("minRing", WIN_CTRL, 2, 30, noop)               # /100 => 0.00..0.30

    # enforce lower bounds by initial values
    cv.setTrackbarPos("K", WIN_CTRL, 8)
    cv.setTrackbarPos("betaXY", WIN_CTRL, 12)
    cv.setTrackbarPos("borderThr", WIN_CTRL, 6)
    cv.setTrackbarPos("median", WIN_CTRL, 5)
    cv.setTrackbarPos("closeK", WIN_CTRL, 11)
    cv.setTrackbarPos("closeIt", WIN_CTRL, 3)
    cv.setTrackbarPos("DT_fore", WIN_CTRL, 32)
    cv.setTrackbarPos("minCirc", WIN_CTRL, 48)
    cv.setTrackbarPos("minEdge", WIN_CTRL, 8)
    cv.setTrackbarPos("minRing", WIN_CTRL, 2)


def read_controls():
    K_ = max(2, tb_get_int("K"))
    beta_xy = tb_get_float("betaXY", 100.0)
    border_thr = tb_get_float("borderThr", 100.0)
    label_median = make_odd(tb_get_int("median"), min_v=1, max_v=31)
    close_k = make_odd(tb_get_int("closeK"), min_v=3, max_v=31)
    close_iters = max(1, min(6, tb_get_int("closeIt")))
    dt_fore = tb_get_float("DT_fore", 100.0)
    min_circ = tb_get_float("minCirc", 100.0)
    min_edge = tb_get_float("minEdge", 100.0)
    min_ring = tb_get_float("minRing", 100.0)
    return K_, beta_xy, border_thr, label_median, close_k, close_iters, dt_fore, min_circ, min_edge, min_ring


def main():
    if not IN_DIR.exists():
        print(f"Input dir not found: {IN_DIR}")
        return
    paths = iter_images_recursive(IN_DIR)
    if not paths:
        print(f"No images found under: {IN_DIR}")
        return

    cv.namedWindow(WIN_OVERLAY, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_LABELS, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_FG, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_DT, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_WS, cv.WINDOW_NORMAL)

    make_controls()

    idx = 0
    show_debug = True
    cache = Cache()

    last_state = None
    force = True

    while True:
        p = paths[idx]
        img0 = cv.imread(str(p), cv.IMREAD_COLOR)
        if img0 is None:
            print(f"[skip] failed to read: {p}")
            idx = (idx + 1) % len(paths)
            cache.img = None
            force = True
            continue

        # controls
        K_, beta_xy, border_thr, label_median, close_k, close_iters, dt_fore, min_circ, min_edge, min_ring = read_controls()

        # recompute logic:
        # - if image changed -> reset cache to recompute kmeans
        # - if K or betaXY changed -> recompute kmeans
        state = (p, K_, round(beta_xy, 4), border_thr, label_median, close_k, close_iters, dt_fore, min_circ, min_edge, min_ring, show_debug)
        if force or (last_state is None) or (state != last_state):
            # if image changed, ensure kmeans recomputes
            if last_state is None or p != last_state[0]:
                cache.img = None

            img, img_blur, edges, labels = compute_labels_cached(cache, img0, K_, beta_xy)

            overlay, lab_vis, fg, dist, markers, ncoins = detect_from_labels(
                img, img_blur, edges, labels,
                K_=K_,
                border_thr=border_thr,
                label_median=label_median,
                close_k=close_k,
                close_iters=close_iters,
                dt_fore_frac=dt_fore,
                min_circ=min_circ,
                min_edge=min_edge,
                min_ring=min_ring,
            )

            ov = overlay.copy()
            draw_text(ov, f"[{idx+1}/{len(paths)}] {p.relative_to(PROJECT_ROOT)}", y=56, scale=0.55)
            cv.imshow(WIN_OVERLAY, ov)

            if show_debug:
                cv.imshow(WIN_LABELS, lab_vis)
                cv.imshow(WIN_FG, fg)

                if dist is not None:
                    d = dist.copy()
                    d = d / (d.max() + 1e-6)
                    dvis = (d * 255).astype(np.uint8)
                    dvis = cv.applyColorMap(dvis, cv.COLORMAP_TURBO)
                    cv.imshow(WIN_DT, dvis)

                ws = markers.copy().astype(np.int32)
                ws_img = np.zeros((*ws.shape, 3), np.uint8)
                ws_img[ws == -1] = (0, 0, 255)
                ws_img[ws >= 2] = (0, 255, 0)
                cv.imshow(WIN_WS, ws_img)
            else:
                cv.destroyWindow(WIN_LABELS)
                cv.destroyWindow(WIN_FG)
                cv.destroyWindow(WIN_DT)
                cv.destroyWindow(WIN_WS)

            last_state = state
            force = False

        # key handling: small wait so sliders stay responsive
        k = cv.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
        elif k in (ord("n"), ord(" ")):
            idx = (idx + 1) % len(paths)
            cache.img = None
            force = True
        elif k == ord("p"):
            idx = (idx - 1) % len(paths)
            cache.img = None
            force = True
        elif k == ord("r"):
            cache.img = None
            force = True
        elif k == ord("d"):
            show_debug = not show_debug
            force = True

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
