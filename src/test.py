#!/usr/bin/env python3
# src/coin_detect_clusters_simple.py
#
# OpenCV-only coin detector using:
#   1) color clustering (k-means) on LAB(a,b)+HSV(S)+optional XY
#   2) background is a SET of clusters (largest + spatially spread / large fractions)
#      -> exclude ONLY background clusters, keep all other colors (coins may be different colors)
#   3) connected components + contour geometry filtering
#
# No CLI args. Designed for project layout:
#   project_root/
#     data/images/
#     src/
#
# Run from anywhere:
#   python3 src/coin_detect_clusters_simple.py
#
# Keys:
#   n / SPACE : next image
#   p         : previous image
#   s         : save overlay (+ debug) to <project_root>/out_simple/
#   d         : toggle debug windows
#   q / ESC   : quit
#
import math
from pathlib import Path

import cv2 as cv
import numpy as np


# -----------------------------
# Paths (robust from src/)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"
OUT_DIR = PROJECT_ROOT / "out_simple"


# -----------------------------
# Performance + quality knobs
# -----------------------------
SCALE = 0.75          # reduce if images are large (0.5..1.0)
K_MIN = 5
K_MAX = 6             # fewer K => much faster; typically 5-6 enough
SAMPLE_N = 25000      # kmeans fit sample size; 15k..50k
ALPHA_S = 1.0         # weight for HSV saturation feature
BETA_XY = 0.12        # spatial regularization (0 disables)
MAX_PICKED = 30


# -----------------------------
# Helpers
# -----------------------------
def iter_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resize_keep(img, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)


def normalize01(x: np.ndarray, eps=1e-6):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def ellipse_kernel(ksize: int):
    ksize = int(ksize)
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))


def compute_edge_map(bgr: np.ndarray):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv.Canny(gray, lower, upper, L2gradient=True)


def contour_circularity(area: float, perimeter: float):
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def mask_boundary(mask_u8: np.ndarray):
    k = ellipse_kernel(3)
    dil = cv.dilate(mask_u8, k, iterations=1)
    ero = cv.erode(mask_u8, k, iterations=1)
    return cv.subtract(dil, ero)


def edge_agreement_score(mask_u8: np.ndarray, edges_u8: np.ndarray):
    bd = mask_boundary(mask_u8)
    bd_n = cv.countNonZero(bd)
    if bd_n == 0:
        return 0.0
    k = ellipse_kernel(5)
    edges_d = cv.dilate(edges_u8, k, iterations=1)
    overlap = cv.bitwise_and(bd, edges_d)
    return float(cv.countNonZero(overlap) / bd_n)


def ring_contrast_score(bgr: np.ndarray, mask_u8: np.ndarray):
    h, w = mask_u8.shape[:2]
    k = ellipse_kernel(max(5, int(0.01 * min(h, w)) | 1))

    inner = cv.erode(mask_u8, k, iterations=1)
    outer = cv.dilate(mask_u8, k, iterations=1)
    ring = cv.subtract(outer, mask_u8)

    if cv.countNonZero(inner) < 50 or cv.countNonZero(ring) < 50:
        return 0.0

    inside_mean = cv.mean(bgr, mask=inner)[:3]
    ring_mean = cv.mean(bgr, mask=ring)[:3]
    diff = np.array(inside_mean, dtype=np.float32) - np.array(ring_mean, dtype=np.float32)
    return float(np.linalg.norm(diff) / 255.0)


def build_features(bgr: np.ndarray, alpha_s: float, beta_xy: float):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)

    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    a01 = normalize01(a)
    b01 = normalize01(b)
    s01 = normalize01(s)

    h, w = a01.shape
    feats = [a01, b01, alpha_s * s01]

    if beta_xy > 1e-9:
        xs = (np.tile(np.arange(w, dtype=np.float32), (h, 1)) / max(1.0, float(w - 1)))
        ys = (np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) / max(1.0, float(h - 1)))
        feats.append(beta_xy * xs)
        feats.append(beta_xy * ys)

    F = np.dstack(feats).astype(np.float32)      # H x W x D
    return F.reshape((-1, F.shape[2]))           # N x D


def kmeans_sample_then_assign(F: np.ndarray, K: int, sample_n: int, attempts: int = 1, seed: int = 123):
    """
    Speed trick:
      - run kmeans on a random sample of pixels
      - assign all pixels to nearest center (vectorized)
    """
    X = F.astype(np.float32)
    N = X.shape[0]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    flags = cv.KMEANS_PP_CENTERS

    if N <= sample_n:
        _, labels, centers = cv.kmeans(X, K, None, criteria, attempts, flags)
        return labels.reshape(-1).astype(np.int32), centers.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=sample_n, replace=False)
    Xs = X[idx]

    _, _, centers = cv.kmeans(Xs, K, None, criteria, attempts, flags)
    C = centers.astype(np.float32)  # K x D

    # Assign all points: argmin ||x - c||^2
    # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
    X2 = np.sum(X * X, axis=1, keepdims=True)        # N x 1
    C2 = np.sum(C * C, axis=1, keepdims=True).T      # 1 x K
    d2 = X2 + C2 - 2.0 * (X @ C.T)                   # N x K
    labels = np.argmin(d2, axis=1).astype(np.int32)
    return labels, C


def make_cluster_masks(labels_hw: np.ndarray, K: int):
    return [((labels_hw == k).astype(np.uint8) * 255) for k in range(K)]


def clean_mask(mask_u8: np.ndarray, h: int, w: int):
    k1 = max(3, int(0.01 * min(h, w)) | 1)
    k2 = max(k1, int(0.02 * min(h, w)) | 1)
    k_small = ellipse_kernel(k1)
    k_med = ellipse_kernel(k2)

    m = mask_u8.copy()
    m = cv.morphologyEx(m, cv.MORPH_OPEN, k_small, iterations=1)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_med, iterations=2)
    return m


def extract_candidate_components(mask_u8: np.ndarray, min_area: int, max_area: int):
    num, cc, stats, cents = cv.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp_mask = (cc == i).astype(np.uint8) * 255
        comps.append((comp_mask, stats[i], cents[i]))
    return comps


def contour_from_mask(mask_u8: np.ndarray):
    cnts, _ = cv.findContours((mask_u8 > 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv.contourArea)


def score_component(bgr: np.ndarray, comp_mask_u8: np.ndarray, edges_u8: np.ndarray, min_area: int, max_area: int):
    cnt = contour_from_mask(comp_mask_u8)
    if cnt is None or len(cnt) < 5:
        return None

    area = float(cv.contourArea(cnt))
    if area < min_area or area > max_area:
        return None

    perim = float(cv.arcLength(cnt, True))
    circ = contour_circularity(area, perim)

    x, y, ww, hh = cv.boundingRect(cnt)
    aspect = float(ww) / float(hh) if hh > 0 else 999.0
    aspect = max(aspect, 1.0 / max(1e-6, aspect))  # >= 1

    ell = None
    axis_ratio = None
    if len(cnt) >= 20:
        try:
            ell = cv.fitEllipse(cnt)
            (_, _), (ma, Mi), _ = ell
            a = max(ma, Mi) / 2.0
            b = min(ma, Mi) / 2.0
            axis_ratio = float(a / max(1e-6, b))
        except cv.error:
            ell = None

    edge_ag = edge_agreement_score(comp_mask_u8, edges_u8)
    ring_con = ring_contrast_score(bgr, comp_mask_u8)

    # Hard-ish gates
    if circ < 0.55:
        return None
    if axis_ratio is not None and axis_ratio > 2.2:
        return None

    # Weighted score
    score = 0.0
    score += 2.0 * circ
    score += 1.2 * edge_ag
    score += 0.6 * min(1.0, ring_con * 2.0)
    score -= 0.8 * min(2.0, (aspect - 1.0))
    if axis_ratio is not None:
        score -= 0.6 * min(2.0, (axis_ratio - 1.0))

    return {
        "score": float(score),
        "contour": cnt,
        "area": float(area),
        "circularity": float(circ),
        "edge_agreement": float(edge_ag),
        "ring_contrast": float(ring_con),
        "bbox": (int(x), int(y), int(ww), int(hh)),
        "ellipse": ell,
        "axis_ratio": None if axis_ratio is None else float(axis_ratio),
    }


def labels_viz(labels_hw: np.ndarray, K: int):
    h, w = labels_hw.shape
    vis = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(12345 + K)
    colors = (rng.integers(0, 255, size=(K, 3))).astype(np.uint8)
    for k in range(K):
        vis[labels_hw == k] = colors[k].tolist()
    return vis


def cluster_fracs(labels_hw: np.ndarray, K: int):
    counts = np.bincount(labels_hw.reshape(-1), minlength=K)
    fracs = counts.astype(np.float32) / float(labels_hw.size)
    return counts, fracs


def cluster_grid_coverage(labels_hw: np.ndarray, K: int, gy: int = 12, gx: int = 12):
    # fraction of grid cells touched by each cluster (0..1)
    h, w = labels_hw.shape
    cov_hits = np.zeros((K,), dtype=np.int32)
    total_cells = gy * gx

    ys = np.linspace(0, h, gy + 1, dtype=np.int32)
    xs = np.linspace(0, w, gx + 1, dtype=np.int32)

    for iy in range(gy):
        y1, y2 = ys[iy], ys[iy + 1]
        for ix in range(gx):
            x1, x2 = xs[ix], xs[ix + 1]
            cell = labels_hw[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            present = np.unique(cell)
            cov_hits[present] += 1

    return cov_hits.astype(np.float32) / float(total_cells)


def select_background_clusters(labels_hw: np.ndarray, K: int):
    """
    Background can be:
      - one dominant cluster (solid background)
      - multiple clusters spread across the image (textured/multicolor background)

    Heuristic:
      - always include the largest cluster
      - include clusters that are either:
          (A) large by fraction, or
          (B) spatially widespread on a coarse grid + not tiny
    """
    counts, fracs = cluster_fracs(labels_hw, K)
    cov = cluster_grid_coverage(labels_hw, K, gy=12, gx=12)

    bg = set([int(np.argmax(counts))])

    for k in range(K):
        fk = float(fracs[k])
        ck = float(cov[k])

        # large fraction => likely background region/texture
        if fk >= 0.15:
            bg.add(k)
            continue

        # scattered across image + non-trivial fraction => background texture
        if ck >= 0.55 and fk >= 0.05:
            bg.add(k)
            continue

    return bg, counts, fracs, cov


def nms_boxes(candidates, iou_thr=0.35, max_keep=MAX_PICKED):
    picked = []
    for cand in candidates:
        x, y, ww, hh = cand["bbox"]
        bx1, by1, bx2, by2 = x, y, x + ww, y + hh

        ok = True
        for pk in picked:
            x2, y2, w2, h2 = pk["bbox"]
            ax1, ay1, ax2, ay2 = x2, y2, x2 + w2, y2 + h2

            ix1 = max(bx1, ax1)
            iy1 = max(by1, ay1)
            ix2 = min(bx2, ax2)
            iy2 = min(by2, ay2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter > 0:
                area_b = (bx2 - bx1) * (by2 - by1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                union = area_a + area_b - inter
                iou = inter / max(1e-6, union)
                if iou > iou_thr:
                    ok = False
                    break

        if ok:
            picked.append(cand)
        if len(picked) >= max_keep:
            break
    return picked


def run_detector(bgr: np.ndarray):
    img = resize_keep(bgr, SCALE)
    h, w = img.shape[:2]

    img_blur = cv.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

    edges = compute_edge_map(img_blur)
    F = build_features(img_blur, alpha_s=ALPHA_S, beta_xy=BETA_XY)

    total_px = h * w
    min_area = max(80, int(0.0008 * total_px))
    max_area = int(0.35 * total_px)

    best = None

    for K in range(K_MIN, K_MAX + 1):
        labels_flat, _centers = kmeans_sample_then_assign(F, K, sample_n=SAMPLE_N, attempts=1, seed=123 + K)
        labels_hw = labels_flat.reshape((h, w))

        bg_set, _counts, fracs, cov = select_background_clusters(labels_hw, K)
        cluster_masks = make_cluster_masks(labels_hw, K)

        candidates = []

        for k in range(K):
            # Exclude ONLY background clusters; keep all other colors (coins may differ)
            if k in bg_set:
                continue

            m = clean_mask(cluster_masks[k], h, w)
            comps = extract_candidate_components(m, min_area=min_area, max_area=max_area)

            for comp_mask, _, _ in comps:
                sc = score_component(img_blur, comp_mask, edges, min_area=min_area, max_area=max_area)
                if sc is None:
                    continue

                # small bias: clusters that are less spatially spread are more "object-like"
                spread = float(cov[k])  # 0..1
                sc["score"] += 0.15 * (1.0 - spread)

                sc["cluster_k"] = k
                sc["cluster_frac"] = float(fracs[k])
                sc["cluster_cov"] = float(cov[k])
                candidates.append(sc)

        if not candidates:
            continue

        candidates.sort(key=lambda d: d["score"], reverse=True)
        picked = nms_boxes(candidates, iou_thr=0.35, max_keep=MAX_PICKED)

        total_score = float(sum(c["score"] for c in picked))
        total_score -= 0.03 * max(0, len(picked) - 12)
        avg_circ = float(np.mean([c["circularity"] for c in picked])) if picked else 0.0
        total_score += 0.2 * avg_circ

        bg_frac = float(sum(float(fracs[k]) for k in bg_set))
        if bg_frac > 0.35:
            total_score += 0.05

        pack = {
            "K": K,
            "labels_hw": labels_hw,
            "picked": picked,
            "total_score": total_score,
            "bg_set": bg_set,
            "bg_frac": bg_frac,
        }

        if best is None or pack["total_score"] > best["total_score"]:
            best = pack

    overlay = img.copy()

    if best is None or not best["picked"]:
        cv.putText(overlay, "No coins detected", (12, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv.LINE_AA)
        return {
            "img": img,
            "overlay": overlay,
            "edges": edges,
            "clusters": None,
            "picked_mask": None,
            "picked": [],
            "K": None,
            "bg_set": None,
            "bg_frac": None,
        }

    K = best["K"]
    picked = best["picked"]

    for i, c in enumerate(picked):
        cnt = c["contour"]
        x, y, ww, hh = c["bbox"]

        cv.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        cv.rectangle(overlay, (x, y), (x + ww, y + hh), (255, 0, 0), 1)

        if c["ellipse"] is not None:
            cv.ellipse(overlay, c["ellipse"], (0, 200, 255), 2)

        label = f"#{i+1} s={c['score']:.2f} C={c['circularity']:.2f}"
        cv.putText(overlay, label, (x, max(0, y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(overlay, label, (x, max(0, y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

    header = f"K={K} coins={len(picked)} bg={best['bg_frac']:.2f} bgK={len(best['bg_set'])}"
    cv.putText(overlay, header, (12, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(overlay, header, (12, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv.LINE_AA)

    picked_mask = np.zeros((overlay.shape[0], overlay.shape[1]), np.uint8)
    for c in picked:
        cv.drawContours(picked_mask, [c["contour"]], -1, 255, -1)

    clusters = labels_viz(best["labels_hw"], K)

    return {
        "img": img,
        "overlay": overlay,
        "edges": edges,
        "clusters": clusters,
        "picked_mask": picked_mask,
        "picked": picked,
        "K": K,
        "bg_set": best["bg_set"],
        "bg_frac": best["bg_frac"],
    }


def main():
    if not IN_DIR.exists():
        print(f"Input dir not found: {IN_DIR}")
        return

    paths = iter_images_recursive(IN_DIR)
    if not paths:
        print(f"No images found under: {IN_DIR}")
        return

    ensure_dir(OUT_DIR)

    debug_on = True
    idx = 0

    def show_all(result, path_str: str):
        cv.imshow("coin | overlay", result["overlay"])
        if debug_on:
            cv.imshow("coin | edges", result["edges"])
            if result["clusters"] is not None:
                cv.imshow("coin | clusters", result["clusters"])
            if result["picked_mask"] is not None:
                cv.imshow("coin | picked_mask", result["picked_mask"])
        else:
            cv.destroyWindow("coin | edges")
            cv.destroyWindow("coin | clusters")
            cv.destroyWindow("coin | picked_mask")

        K = result["K"]
        bg = result["bg_frac"]
        if K is None:
            print(f"[{idx+1}/{len(paths)}] {path_str} | coins=0")
        else:
            print(f"[{idx+1}/{len(paths)}] {path_str} | coins={len(result['picked'])} K={K} bg={bg:.2f}")

    while True:
        p = paths[idx]
        img0 = cv.imread(str(p), cv.IMREAD_COLOR)
        if img0 is None:
            print(f"[skip] failed to read: {p}")
            idx = (idx + 1) % len(paths)
            continue

        result = run_detector(img0)
        show_all(result, str(p))

        key = cv.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            break
        elif key in (ord("n"), ord(" ")):
            idx = (idx + 1) % len(paths)
        elif key == ord("p"):
            idx = (idx - 1) % len(paths)
        elif key == ord("d"):
            debug_on = not debug_on
        elif key == ord("s"):
            stem = p.as_posix().replace("/", "__")
            cv.imwrite(str(OUT_DIR / f"{stem}__overlay.png"), result["overlay"])
            if result["clusters"] is not None:
                cv.imwrite(str(OUT_DIR / f"{stem}__clusters.png"), result["clusters"])
            cv.imwrite(str(OUT_DIR / f"{stem}__edges.png"), result["edges"])
            if result["picked_mask"] is not None:
                cv.imwrite(str(OUT_DIR / f"{stem}__picked_mask.png"), result["picked_mask"])
            print(f"Saved to: {OUT_DIR}")
        else:
            idx = (idx + 1) % len(paths)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
