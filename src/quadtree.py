import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class QTParams:
    min_size: int = 16
    var_thresh: float = 18.0
    bg_dist_thresh: float = 14.0
    border_frac: float = 0.08
    morph_ksize: int = 7
    min_area: int = 300
    max_area_frac: float = 0.6
    min_circularity: float = 0.65

def estimate_bg_lab(lab: np.ndarray, border_frac: float) -> np.ndarray:
    H, W = lab.shape[:2]
    b = int(max(1, round(min(H, W) * border_frac)))
    border = np.vstack([
        lab[:b, :, :].reshape(-1, 3),
        lab[H-b:, :, :].reshape(-1, 3),
        lab[:, :b, :].reshape(-1, 3),
        lab[:, W-b:, :].reshape(-1, 3),
    ]).astype(np.float32)
    return np.median(border, axis=0)  # robust

def lab_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt((d * d).sum()))

def block_stats(lab: np.ndarray, x0: int, y0: int, w: int, h: int):
    roi = lab[y0:y0+h, x0:x0+w].astype(np.float32)
    mean = roi.reshape(-1, 3).mean(axis=0)
    stdL = float(roi[..., 0].reshape(-1).std())  # L channel
    return mean, stdL

def quadtree_mask(lab: np.ndarray, p: QTParams) -> np.ndarray:
    H, W = lab.shape[:2]
    bg = estimate_bg_lab(lab, p.border_frac)
    mask = np.zeros((H, W), np.uint8)

    stack: List[Tuple[int, int, int, int]] = [(0, 0, W, H)]
    while stack:
        x0, y0, w, h = stack.pop()
        if w <= 0 or h <= 0:
            continue

        mean, stdL = block_stats(lab, x0, y0, w, h)

        stop = (w <= p.min_size) or (h <= p.min_size) or (stdL <= p.var_thresh)
        if stop:
            if lab_dist(mean, bg) > p.bg_dist_thresh:
                mask[y0:y0+h, x0:x0+w] = 255
            continue

        hw, hh = w // 2, h // 2
        if hw == 0 or hh == 0:
            if lab_dist(mean, bg) > p.bg_dist_thresh:
                mask[y0:y0+h, x0:x0+w] = 255
            continue

        stack.append((x0,      y0,      hw,    hh))
        stack.append((x0+hw,   y0,      w-hw,  hh))
        stack.append((x0,      y0+hh,   hw,    h-hh))
        stack.append((x0+hw,   y0+hh,   w-hw,  h-hh))

    return mask

def postprocess_mask(mask: np.ndarray, p: QTParams) -> np.ndarray:
    k = max(3, p.morph_ksize | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=2)

    # hole fill
    inv = cv2.bitwise_not(m)
    h, w = inv.shape[:2]
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 0)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(m, holes)

def detect_coins(mask: np.ndarray, p: QTParams):
    H, W = mask.shape[:2]
    max_area = p.max_area_frac * (H * W)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coins = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < p.min_area or area > max_area:
            continue
        per = float(cv2.arcLength(c, True))
        if per <= 1e-6:
            continue
        circ = 4.0 * np.pi * area / (per * per)
        if circ < p.min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        coins.append(((int(round(x)), int(round(y))), int(round(r)), float(circ)))
    coins.sort(key=lambda t: t[1], reverse=True)
    return coins

def draw(img: np.ndarray, coins):
    out = img.copy()
    for (cx, cy), r, circ in coins:
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(out, f"r={r} c={circ:.2f}", (cx + r + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return out

if __name__ == "__main__":
    # ABSOLUTE PATH ONLY
    IMG_PATH = "/Users/sigmoid/Desktop/Coding/Git/Image_projet_money/data/images/data/gp1/18.png"  # <- change this

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise SystemExit(f"Cannot read image at: {IMG_PATH}")

    p = QTParams(
        min_size=8,
        var_thresh=18.0,
        bg_dist_thresh=14.0,
        border_frac=0.14,
        morph_ksize=7,
        min_area=200,
        max_area_frac=0.6,
        min_circularity=0.65
    )

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    raw = quadtree_mask(lab, p)
    clean = postprocess_mask(raw, p)
    coins = detect_coins(clean, p)
    vis = draw(img, coins)

    print("Coins:", len(coins))
    for i, (c, r, circ) in enumerate(coins):
        print(f"#{i}: center={c}, r={r}, circularity={circ:.3f}")

    cv2.imshow("raw_mask", raw)
    cv2.imshow("clean_mask", clean)
    cv2.imshow("coins", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
