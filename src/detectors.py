"""Core algorithm primitives: circle detection, analysis and value estimation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from src.config import HoughPreset
from src.preprocessing import auto_hough_param1_from_gradient
@dataclass(frozen=True)
class CoinDetectionResult:
    circles: np.ndarray | None
    circle_count: int
    hough_params: dict[str, float | int]
    sweep_debug: dict[str, int | float | list[int] | str]
    sweep_results: list[tuple[int, dict[str, int | float]]]
    circles_overlay: np.ndarray


class CoinDetector:
    def __init__(
        self,
        preset: HoughPreset,
        min_radius_sweep_start: int = 10,
        min_radius_sweep_end: int = 140,
        min_radius_sweep_step: int = 2,
        max_radius: int = 100,
        auto_param1_blur_ksize: int = 5,
        auto_param1_percentile: float = 65.0,
        auto_param1_scale: float = 1.0,
        auto_param1_clamp: tuple[int, int] = (30, 220),
        circle_outline_color: tuple[int, int, int] = (0, 255, 0),
        circle_outline_thickness: int = 2,
        center_color: tuple[int, int, int] = (255, 0, 0),
        center_radius: int = 2,
        center_thickness: int = 3,
    ):
        self._preset = preset
        self._min_radius_sweep_start = int(min_radius_sweep_start)
        self._min_radius_sweep_end = int(min_radius_sweep_end)
        self._min_radius_sweep_step = int(min_radius_sweep_step)
        self._max_radius = int(max_radius)
        self._auto_param1_blur_ksize = int(auto_param1_blur_ksize)
        self._auto_param1_percentile = float(auto_param1_percentile)
        self._auto_param1_scale = float(auto_param1_scale)
        self._auto_param1_clamp = auto_param1_clamp
        self._circle_outline_color = circle_outline_color
        self._circle_outline_thickness = int(circle_outline_thickness)
        self._center_color = center_color
        self._center_radius = int(center_radius)
        self._center_thickness = int(center_thickness)

    def detect(self, gray: np.ndarray, blurred: np.ndarray, image_rgb: np.ndarray) -> CoinDetectionResult:
        hough_params: dict[str, float | int] = {
            "dp": self._preset.dp,
            "minDist": self._preset.min_dist,
            "param2": self._preset.param2,
        }

        param1 = auto_hough_param1_from_gradient(
            gray,
            blur_ksize=self._auto_param1_blur_ksize,
            perc=self._auto_param1_percentile,
            scale=self._auto_param1_scale,
            clamp=self._auto_param1_clamp,
        )
        hough_params["param1"] = int(param1)
        hough_params["maxRadius"] = int(self._max_radius)

        sweep_end = min(self._min_radius_sweep_end, int(hough_params["maxRadius"]) - 1)
        if sweep_end < self._min_radius_sweep_start:
            sweep_end = self._min_radius_sweep_start

        best_min_radius, sweep_debug, sweep_results = auto_minradius_plateau(
            blurred,
            base_params=hough_params,
            minR_start=self._min_radius_sweep_start,
            minR_end=sweep_end,
            step=self._min_radius_sweep_step,
        )
        hough_params["minRadius"] = int(best_min_radius)

        circles = detect_hough_circles(blurred, hough_params)
        circles_overlay, circle_count = draw_circles_on_rgb(
            image_rgb,
            circles,
            outline_color=self._circle_outline_color,
            outline_thickness=self._circle_outline_thickness,
            center_color=self._center_color,
            center_radius=self._center_radius,
            center_thickness=self._center_thickness,
        )

        return CoinDetectionResult(
            circles=circles,
            circle_count=circle_count,
            hough_params=hough_params,
            sweep_debug=sweep_debug,
            sweep_results=sweep_results,
            circles_overlay=circles_overlay,
        )


def detect_hough_circles(blurred_gray: np.ndarray, params: dict[str, float | int]) -> np.ndarray | None:
    return cv2.HoughCircles(
        blurred_gray,
        cv2.HOUGH_GRADIENT,
        dp=float(params["dp"]),
        minDist=float(params["minDist"]),
        param1=float(params["param1"]),
        param2=float(params["param2"]),
        minRadius=int(params["minRadius"]),
        maxRadius=int(params["maxRadius"]),
    )


def run_hough_with_params(blurred: np.ndarray, hough_params: dict[str, float | int]) -> np.ndarray | None:
    circles = detect_hough_circles(blurred, hough_params)
    if circles is None:
        return None
    return np.round(circles[0, :]).astype(int)


def circle_nesting_score(circles_int: np.ndarray | None) -> tuple[float, dict[str, int | float]]:
    if circles_int is None or len(circles_int) == 0:
        return 1e9, {
            "n": 0,
            "concentric_pairs": 0,
            "nested_pairs": 0,
            "intrusion_pairs": 0,
            "max_intrusions_in_one": 0,
            "score": 1e9,
        }

    circles_float = circles_int.astype(np.float32)
    n = circles_float.shape[0]
    concentric_pairs = 0
    nested_pairs = 0
    intrusion_pairs_directed = 0
    intrusions_per_big = np.zeros((n,), dtype=np.int32)

    same_center_frac = 0.12
    nested_ratio = 1.01
    center_margin = 0.05

    for i in range(n):
        xi, yi, ri = float(circles_float[i, 0]), float(circles_float[i, 1]), float(circles_float[i, 2])
        for j in range(i + 1, n):
            xj, yj, rj = float(circles_float[j, 0]), float(circles_float[j, 1]), float(circles_float[j, 2])

            dx = xi - xj
            dy = yi - yj
            dist = float(np.sqrt(dx * dx + dy * dy))

            r_small = min(ri, rj)
            r_big = max(ri, rj)
            if dist < same_center_frac * r_small:
                concentric_pairs += 1
            if (dist + r_small) <= r_big and (r_big >= 1.05 * r_small):
                nested_pairs += 1

            if ri >= nested_ratio * rj:
                big_idx = i
                r_big_dir, r_small_dir = ri, rj
                dist_dir = dist
            elif rj >= nested_ratio * ri:
                big_idx = j
                r_big_dir, r_small_dir = rj, ri
                dist_dir = dist
            else:
                continue

            if dist_dir < (r_big_dir - center_margin * r_small_dir):
                intrusion_pairs_directed += 1
                intrusions_per_big[big_idx] += 1

    max_intrusions_in_one = int(np.max(intrusions_per_big)) if n > 0 else 0
    score = (
        12.0 * concentric_pairs
        + 8.0 * nested_pairs
        + 10.0 * intrusion_pairs_directed
        + 6.0 * max_intrusions_in_one
        + 0.10 * n
    )
    metrics: dict[str, int | float] = {
        "n": int(n),
        "concentric_pairs": int(concentric_pairs),
        "nested_pairs": int(nested_pairs),
        "intrusion_pairs": int(intrusion_pairs_directed),
        "max_intrusions_in_one": max_intrusions_in_one,
        "score": float(score),
    }
    return float(score), metrics


def auto_minradius_plateau(
    blurred: np.ndarray,
    base_params: dict[str, float | int],
    minR_start: int = 10,
    minR_end: int = 120,
    step: int = 2,
) -> tuple[int, dict[str, int | float | list[int] | str], list[tuple[int, dict[str, int | float]]]]:
    results: list[tuple[int, dict[str, int | float]]] = []
    for min_radius in range(minR_start, minR_end + 1, step):
        params = base_params.copy()
        params["minRadius"] = int(min_radius)
        circles_int = run_hough_with_params(blurred, params)
        _, metrics = circle_nesting_score(circles_int)
        results.append((min_radius, metrics))

    good = [
        (min_radius, metrics)
        for (min_radius, metrics) in results
        if metrics["nested_pairs"] == 0 and metrics["max_intrusions_in_one"] == 0
    ]
    if len(good) == 0:
        best_nested = min(int(metrics["nested_pairs"]) for (_, metrics) in results)
        candidates = [
            (min_radius, metrics)
            for (min_radius, metrics) in results
            if int(metrics["nested_pairs"]) == best_nested
        ]
        best_intrusions = min(int(metrics["max_intrusions_in_one"]) for (_, metrics) in candidates)
        candidates = [
            (min_radius, metrics)
            for (min_radius, metrics) in candidates
            if int(metrics["max_intrusions_in_one"]) == best_intrusions
        ]
        best_n = max(int(metrics["n"]) for (_, metrics) in candidates)
        min_radius_list = sorted(
            min_radius for (min_radius, metrics) in candidates if int(metrics["n"]) == best_n
        )
        chosen = min_radius_list[len(min_radius_list) // 2]
        return chosen, {
            "reason": "no_good_results",
            "best_nested": best_nested,
            "best_intrusions": best_intrusions,
            "best_n": best_n,
        }, results

    n_to_min_radius: dict[int, list[int]] = {}
    for min_radius, metrics in good:
        n = int(metrics["n"])
        n_to_min_radius.setdefault(n, []).append(int(min_radius))

    positive_counts = {n: values for n, values in n_to_min_radius.items() if n > 0}
    selected_counts = positive_counts if len(positive_counts) > 0 else n_to_min_radius

    counts = [(n, len(values)) for n, values in selected_counts.items()]
    max_frequency = max(freq for _, freq in counts)
    n_candidates = [n for n, freq in counts if freq == max_frequency]
    n_mode = max(n_candidates)

    min_radius_candidates = sorted(selected_counts[n_mode])
    chosen_min_radius = min_radius_candidates[len(min_radius_candidates) // 2]

    return int(chosen_min_radius), {
        "n_mode": n_mode,
        "freq": max_frequency,
        "min_radius_candidates": min_radius_candidates,
    }, results


def draw_circles_on_rgb(
    image_rgb: np.ndarray,
    circles: np.ndarray | None,
    outline_color: tuple[int, int, int],
    outline_thickness: int,
    center_color: tuple[int, int, int],
    center_radius: int,
    center_thickness: int,
) -> tuple[np.ndarray, int]:
    output = image_rgb.copy()
    if circles is None:
        return output, 0

    circles_int = np.round(circles[0, :]).astype(int)
    for x, y, radius in circles_int:
        cv2.circle(output, (x, y), radius, outline_color, outline_thickness)
        cv2.circle(output, (x, y), center_radius, center_color, center_thickness)
    return output, int(len(circles_int))


# -----------------------------------------------------------------------------
# Inlined from src/value_estimator.py
# -----------------------------------------------------------------------------








# -----------------------------------------------------------------------------
# CoinValueEstimator internals (consolidated for pipeline-stage OOP structure)
# -----------------------------------------------------------------------------

def mean_hsv_from_pixels(hsv_pixels: np.ndarray) -> tuple[int, int, int]:
    if hsv_pixels is None or len(hsv_pixels) == 0:
        return 0, 0, 0

    h_vals = hsv_pixels[:, 0].astype(np.float32)
    s_vals = hsv_pixels[:, 1].astype(np.float32)
    v_vals = hsv_pixels[:, 2].astype(np.float32)

    theta = h_vals * (2.0 * np.pi / 180.0)
    weights = np.clip(s_vals, 1.0, 255.0)
    mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
    mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))

    hue_deg = np.degrees(np.arctan2(mean_sin, mean_cos))
    if hue_deg < 0.0:
        hue_deg += 360.0

    h_cv = int(np.clip(round(hue_deg / 2.0), 0, 179))
    s_cv = int(np.clip(round(np.median(s_vals)), 0, 255))
    v_cv = int(np.clip(round(np.median(v_vals)), 0, 255))
    return h_cv, s_cv, v_cv


def hsv_similarity_score(hsv_a: tuple[int, int, int], hsv_b: tuple[int, int, int]) -> float:
    _, sa, _ = hsv_a
    _, sb, _ = hsv_b

    ds = abs(float(sa) - float(sb)) / 255.0
    sim = 1.0 - ds
    return float(np.clip(sim, 0.0, 1.0))


def hue_circular_delta_cv(h1: int, h2: int) -> float:
    d = abs(float(h1) - float(h2))
    return float(min(d, 180.0 - d))


def choose_dynamic_sat_delta_threshold(sat_deltas: list[float], default: float = 18.0) -> float:
    if sat_deltas is None:
        return float(default)

    vals = np.asarray(sat_deltas, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)

    vals = np.sort(vals)
    n = int(vals.size)
    if n < 4:
        return float(np.max(vals) + 1.0)

    gaps = vals[1:] - vals[:-1]
    if gaps.size == 0:
        return float(np.max(vals) + 1.0)

    median_gap = float(np.median(gaps))
    min_low = max(3, int(np.ceil(0.55 * n)))

    valid_idx = [i for i in range(len(gaps)) if (i + 1) >= min_low and (n - i - 1) >= 1]
    if len(valid_idx) == 0:
        return float(np.max(vals) + 1.0)

    best_i = max(valid_idx, key=lambda i: float(gaps[i]))
    best_gap = float(gaps[best_i])

    if best_gap < max(6.0, 2.2 * max(median_gap, 1.0)):
        return float(np.max(vals) + 1.0)

    low = vals[: best_i + 1]
    high = vals[best_i + 1 :]

    n_high = int(high.size)
    high_frac = float(n_high / n)
    if high_frac > 0.45:
        return float(np.max(vals) + 1.0)

    sep = float(np.mean(high) - np.mean(low))
    if sep < max(8.0, 1.3 * float(np.std(low) + 1.0)):
        return float(np.max(vals) + 1.0)

    thr = float(0.5 * (vals[best_i] + vals[best_i + 1]))
    p60 = float(np.percentile(vals, 60))
    p99 = float(np.percentile(vals, 99))
    return float(np.clip(thr, p60, p99))

def clamp_value(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def bronze_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    h_term = clamp_value((18.0 - float(h)) / 3.0, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 45.0) / 75.0, 0.0, 1.0)
    val_conf = clamp_value((float(v) - 35.0) / 90.0, 0.0, 1.0)
    conf = 0.65 * sat_conf + 0.35 * val_conf
    return float(np.clip(h_term * max(0.35, conf), 0.0, 1.0))


def gold_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    hue_center = 22.0
    hue_width = 8.5
    h_term = 1.0 - clamp_value(abs(float(h) - hue_center) / hue_width, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 40.0) / 90.0, 0.0, 1.0)
    val_conf = clamp_value((float(v) - 55.0) / 110.0, 0.0, 1.0)
    conf = 0.60 * sat_conf + 0.40 * val_conf
    return float(np.clip(h_term * max(0.30, conf), 0.0, 1.0))


def label_material_from_inner_hsv(h: int, s: int, v: int) -> str:
    if s < 45 or v < 35:
        return "borderline"
    if h <= 15:
        return "bronze"
    if h >= 18:
        return "gold"
    return "borderline"

def label_bimetal_euro_from_saturation(inner_s: int, border_s: int) -> str:
    if inner_s < border_s:
        return "1-euro-like"
    if inner_s > border_s:
        return "2-euro-like"
    return "bi-metal-euro-uncertain"


def fallback_material_from_inner_hsv(h: int, s: int, v: int) -> str:
    material = label_material_from_inner_hsv(h, s, v)
    if material != "borderline":
        return material

    bronze_score = bronze_score_from_inner_hsv(h, s, v)
    gold_score = gold_score_from_inner_hsv(h, s, v)
    if max(bronze_score, gold_score) >= 0.30 and abs(bronze_score - gold_score) >= 0.02:
        return "bronze" if bronze_score > gold_score else "gold"

    # Keep a deterministic fallback for very low-confidence borderline colors.
    return "bronze" if h <= 17 else "gold"


def material_family_from_inner_hsv(inner_hsv: tuple[int, int, int] | list[int]) -> str:
    if not isinstance(inner_hsv, (tuple, list)) or len(inner_hsv) < 3:
        return "unknown"
    h = int(inner_hsv[0])
    s = int(inner_hsv[1])
    v = int(inner_hsv[2])

    bronze_score = bronze_score_from_inner_hsv(h, s, v)
    gold_score = gold_score_from_inner_hsv(h, s, v)
    if max(bronze_score, gold_score) >= 0.30 and abs(bronze_score - gold_score) >= 0.02:
        return "bronze" if bronze_score > gold_score else "gold"

    if s >= 45:
        if h <= 15:
            return "bronze"
        if h >= 16:
            return "gold"
    return "unknown"

def coin_kmeans_radial_structure(hsv_pixels: np.ndarray, radial_norm: np.ndarray) -> dict[str, float | bool]:
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0

    w = np.clip(s, 0.2, 1.0)
    feats = np.stack([np.cos(h) * w, np.sin(h) * w, s, 0.35 * v], axis=1).astype(np.float32)

    if n > 4000:
        rng = np.random.default_rng(123)
        idx = rng.choice(n, size=4000, replace=False)
        feats_fit = feats[idx]
    else:
        feats_fit = feats

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.02)
    cv2.setRNGSeed(1234)
    _, _, centers = cv2.kmeans(feats_fit, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    d0 = np.sum((feats - centers[0]) ** 2, axis=1)
    d1 = np.sum((feats - centers[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.int32)

    cnt0 = int(np.sum(labels == 0))
    cnt1 = int(np.sum(labels == 1))
    balance = float(min(cnt0, cnt1) / max(n, 1))
    if balance < 0.08:
        return {
            "ok": False,
            "score": float(np.clip(balance / 0.08, 0.0, 1.0)),
            "radial_sep": 0.0,
            "agreement": 0.0,
            "balance": balance,
        }

    r0 = float(np.mean(radial_norm[labels == 0]))
    r1 = float(np.mean(radial_norm[labels == 1]))
    radial_sep = abs(r0 - r1)

    outer_id = 0 if r0 > r1 else 1
    inner_mean = min(r0, r1)
    outer_mean = max(r0, r1)
    radial_cut = 0.5 * (inner_mean + outer_mean)

    pred_outer = radial_norm >= radial_cut
    lbl_outer = labels == outer_id
    agreement = float(np.mean(pred_outer == lbl_outer))

    score = (
        0.45 * np.clip(radial_sep / 0.22, 0.0, 1.0)
        + 0.35 * np.clip((agreement - 0.5) / 0.4, 0.0, 1.0)
        + 0.20 * np.clip(balance / 0.22, 0.0, 1.0)
    )
    ok = (radial_sep >= 0.13) and (agreement >= 0.66) and (balance >= 0.10)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "radial_sep": float(radial_sep),
        "agreement": float(agreement),
        "balance": float(balance),
    }


def coin_radial_step_score(
    hsv_pixels: np.ndarray,
    radial_norm: np.ndarray,
    split_radius: float,
    bins: int = 8,
) -> dict[str, float | bool]:
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0
    feats = np.stack([np.cos(h) * s, np.sin(h) * s, s, 0.20 * v], axis=1).astype(np.float32)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_vecs: list[np.ndarray | None] = []
    boundary_r: list[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (radial_norm >= lo) & (radial_norm <= hi)
        else:
            mask = (radial_norm >= lo) & (radial_norm < hi)
        if int(np.sum(mask)) < max(20, int(0.03 * n)):
            bin_vecs.append(None)
        else:
            bin_vecs.append(np.mean(feats[mask], axis=0))
        if i < bins - 1:
            boundary_r.append(float(edges[i + 1]))

    steps: list[float] = []
    for i in range(bins - 1):
        a = bin_vecs[i]
        b = bin_vecs[i + 1]
        if (a is None) or (b is None):
            steps.append(0.0)
        else:
            steps.append(float(np.linalg.norm(a - b)))

    if len(steps) == 0:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    idx = int(np.argmax(steps))
    max_step = float(steps[idx])
    step_radius = float(boundary_r[idx])
    dist_to_split = abs(step_radius - float(split_radius))

    strength = np.clip(max_step / 0.17, 0.0, 1.0)
    location = np.clip(1.0 - (dist_to_split / 0.28), 0.0, 1.0)
    score = float(0.65 * strength + 0.35 * location)
    ok = (max_step >= 0.10) and (dist_to_split <= 0.26)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "max_step": float(max_step),
        "step_radius": float(step_radius),
    }


def coin_edge_roughness_score(
    gray_u8: np.ndarray,
    x: int,
    y: int,
    r: int,
    n_bins: int = 72,
) -> dict[str, float]:
    if r < 8:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    h, w = gray_u8.shape[:2]
    x0 = max(0, int(x - 1.3 * r))
    x1 = min(w, int(x + 1.3 * r) + 1)
    y0 = max(0, int(y - 1.3 * r))
    y1 = min(h, int(y + 1.3 * r) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    roi = gray_u8[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, 60, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    xs = xs.astype(np.float32) + float(x0)
    ys = ys.astype(np.float32) + float(y0)
    dx = xs - float(x)
    dy = ys - float(y)
    dist = np.sqrt(dx * dx + dy * dy)

    mask = (dist >= 0.62 * float(r)) & (dist <= 1.20 * float(r))
    if int(np.sum(mask)) < 18:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    dist = dist[mask]
    ang = np.arctan2(dy[mask], dx[mask])
    ang[ang < 0.0] += 2.0 * np.pi

    bins = np.full((n_bins,), np.nan, dtype=np.float32)
    idx = np.floor((ang / (2.0 * np.pi)) * n_bins).astype(np.int32)
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        vals = dist[idx == b]
        if vals.size > 0:
            bins[b] = float(np.median(vals))

    valid = np.isfinite(bins)
    coverage = float(np.mean(valid))
    if int(np.sum(valid)) < max(14, int(0.30 * n_bins)):
        return {"roughness": 0.0, "coverage": coverage, "score": 0.0}

    vals = bins[valid].astype(np.float32)
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    roughness = float((q3 - q1) / (med + 1e-6))

    score = np.clip((roughness - 0.040) / 0.032, 0.0, 1.0)
    score *= np.clip((coverage - 0.35) / 0.45, 0.0, 1.0)
    return {
        "roughness": float(roughness),
        "coverage": float(coverage),
        "score": float(np.clip(score, 0.0, 1.0)),
    }

@dataclass(frozen=True)
class CoinAnalyzerConfig:
    border_ratio: float = 0.24
    sat_delta_threshold: float | None = None
    bimetal_mode: str = "hybrid"
    material_mode: str = "hsv"


class CoinAnalyzer:
    def __init__(self, config: CoinAnalyzerConfig | None = None):
        self._cfg = config or CoinAnalyzerConfig()

    def analyze(self, image_bgr: np.ndarray, circles: np.ndarray | None) -> tuple[np.ndarray, list[dict]]:
        return draw_and_analyze_circle_inner_border_colors(
            image_bgr,
            circles,
            border_ratio=self._cfg.border_ratio,
            sat_delta_threshold=self._cfg.sat_delta_threshold,
            bimetal_mode=self._cfg.bimetal_mode,
            material_mode=self._cfg.material_mode,
        )

    @staticmethod
    def draw_circles_filled_with_average_hue(
        image_bgr: np.ndarray, circles: np.ndarray | None
    ) -> tuple[np.ndarray, int]:
        output_bgr = image_bgr.copy()
        if circles is None:
            return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), 0

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        circles_int = np.round(circles[0, :]).astype(int)
        h_img, w_img = image_bgr.shape[:2]

        for idx, (x, y, r) in enumerate(circles_int, start=1):
            if r <= 1:
                continue
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

            pix = hsv[mask > 0]
            if pix.size == 0:
                continue

            h_vals = pix[:, 0].astype(np.float32)
            s_vals = pix[:, 1].astype(np.float32)
            v_vals = pix[:, 2].astype(np.float32)

            theta = h_vals * (2.0 * np.pi / 180.0)
            weights = np.clip(s_vals, 1.0, 255.0)
            mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
            mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))
            hue = np.degrees(np.arctan2(mean_sin, mean_cos))
            if hue < 0.0:
                hue += 360.0

            hue_cv = int(np.clip(round(hue / 2.0), 0, 179))
            sat_cv = int(np.clip(round(np.median(s_vals)), 35, 255))
            val_cv = int(np.clip(round(np.median(v_vals)), 35, 255))
            fill_hsv = np.uint8([[[hue_cv, sat_cv, val_cv]]])
            fill_bgr = cv2.cvtColor(fill_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            cv2.circle(output_bgr, (int(x), int(y)), int(r), tuple(int(c) for c in fill_bgr), -1)

            text = str(idx)
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), int(len(circles_int))


def radius_spread_metric(circles_int: np.ndarray | None) -> float:
    if circles_int is None or len(circles_int) == 0:
        return 1e9
    r = circles_int[:, 2].astype(np.float32)
    med = float(np.median(r))
    if med <= 1e-6:
        return 1e9
    q1, q3 = np.percentile(r, [25, 75])
    iqr = float(q3 - q1)
    return iqr / med


def draw_and_analyze_circle_inner_border_colors(
    image_bgr: np.ndarray,
    circles: np.ndarray | None,
    border_ratio: float = 0.24,
    sat_delta_threshold: float | None = None,
    bimetal_mode: str = "hybrid",
    material_mode: str = "hsv",
) -> tuple[np.ndarray, list[dict]]:
    bimetal_mode = str(bimetal_mode).strip().lower()
    if bimetal_mode != "hybrid":
        raise ValueError("bimetal_mode must be: 'hybrid'")
    material_mode = str(material_mode).strip().lower()
    if material_mode != "hsv":
        raise ValueError("material_mode must be: 'hsv'")

    output_bgr = image_bgr.copy()
    stats: list[dict] = []
    if circles is None:
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray_shape = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    circles_int = np.round(circles[0, :]).astype(int)
    h_img, w_img = image_bgr.shape[:2]

    rows: list[dict] = []
    color_deltas_hybrid: list[float] = []

    for idx, (x, y, r) in enumerate(circles_int):
        x = int(x)
        y = int(y)
        r = int(max(1, r))
        inner_r = int(max(1, round(r * (1.0 - border_ratio))))
        split_radius = float(inner_r) / float(r)

        outer_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r, 255, -1)
        cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
        border_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        pix_full_hsv = hsv[outer_mask > 0]
        pix_inner_hsv = hsv[inner_mask > 0]
        pix_border_hsv = hsv[border_mask > 0]

        full_hsv = mean_hsv_from_pixels(pix_full_hsv)
        inner_hsv = mean_hsv_from_pixels(pix_inner_hsv)
        border_hsv = mean_hsv_from_pixels(pix_border_hsv)

        similarity = hsv_similarity_score(inner_hsv, border_hsv)
        sat_delta = abs(float(inner_hsv[1]) - float(border_hsv[1]))
        hue_delta = hue_circular_delta_cv(int(inner_hsv[0]), int(border_hsv[0]))
        val_delta = abs(float(inner_hsv[2]) - float(border_hsv[2]))

        sat_mean = 0.5 * (float(inner_hsv[1]) + float(border_hsv[1]))
        sat_conf = float(np.clip(sat_mean / 70.0, 0.0, 1.0))
        hue_term = hue_delta * sat_conf
        value_term = 0.15 * val_delta
        color_delta = float(sat_delta + hue_term + value_term)

        color_deltas_hybrid.append(color_delta)

        ys, xs = np.where(outer_mask > 0)
        if len(xs) > 0:
            radial_norm = (
                np.sqrt((xs.astype(np.float32) - float(x)) ** 2 + (ys.astype(np.float32) - float(y)) ** 2)
                / float(r)
            )
            radial_norm = np.clip(radial_norm, 0.0, 1.0)
            coin_hsv = hsv[ys, xs]
            radial_kmeans = coin_kmeans_radial_structure(coin_hsv, radial_norm)
            radial_step = coin_radial_step_score(coin_hsv, radial_norm, split_radius=split_radius, bins=8)
        else:
            radial_kmeans = {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}
            radial_step = {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": split_radius}

        edge_shape = coin_edge_roughness_score(gray_shape, x=x, y=y, r=r, n_bins=72)
        rows.append(
            {
                "id": idx,
                "x": x,
                "y": y,
                "r": r,
                "inner_r": inner_r,
                "full_hsv": full_hsv,
                "inner_hsv": inner_hsv,
                "border_hsv": border_hsv,
                "full_lab": (0.0, 0.0, 0.0),
                "inner_lab": (0.0, 0.0, 0.0),
                "border_lab": (0.0, 0.0, 0.0),
                "similarity": float(similarity),
                "sat_delta": float(sat_delta),
                "hue_delta": float(hue_delta),
                "val_delta": float(val_delta),
                "sat_conf": float(sat_conf),
                "hue_term": float(hue_term),
                "value_term": float(value_term),
                "color_delta": float(color_delta),
                "lab_delta": 0.0,
                "mean_warm_delta": 0.0,
                "mean_color_delta": 0.0,
                "kmeans_radial_ok": bool(radial_kmeans["ok"]),
                "kmeans_radial_score": float(radial_kmeans["score"]),
                "kmeans_radial_sep": float(radial_kmeans["radial_sep"]),
                "kmeans_radial_agreement": float(radial_kmeans["agreement"]),
                "kmeans_radial_balance": float(radial_kmeans["balance"]),
                "step_ok": bool(radial_step["ok"]),
                "step_score": float(radial_step["score"]),
                "step_max": float(radial_step["max_step"]),
                "step_radius": float(radial_step["step_radius"]),
                "edge_roughness": float(edge_shape["roughness"]),
                "edge_coverage": float(edge_shape["coverage"]),
                "gold_flower_score": float(edge_shape["score"]),
            }
        )

    outlier_threshold_hybrid = choose_dynamic_sat_delta_threshold(color_deltas_hybrid, default=18.0)
    if sat_delta_threshold is not None:
        outlier_threshold_hybrid = float(sat_delta_threshold)

    hybrid_abs_mid = 16.0
    hybrid_abs_hi = 22.0
    hybrid_abs_vhi = 30.0
    hybrid_abs_lo = 10.0

    for row in rows:
        x = row["x"]
        y = row["y"]
        r = row["r"]
        inner_r = row["inner_r"]
        full_hsv = row["full_hsv"]
        inner_hsv = row["inner_hsv"]
        border_hsv = row["border_hsv"]

        color_delta = float(row["color_delta"])
        sat_delta = float(row["sat_delta"])
        hue_delta = float(row["hue_delta"])
        bronze_veto_applied = False

        outlier_threshold = outlier_threshold_hybrid
        abs_mid = hybrid_abs_mid
        abs_hi = hybrid_abs_hi
        abs_vhi = hybrid_abs_vhi
        abs_lo = hybrid_abs_lo

        outlier_vote = color_delta >= outlier_threshold
        mid_delta = color_delta >= abs_mid
        strong_delta = color_delta >= abs_hi
        very_strong_delta = color_delta >= abs_vhi

        kmeans_ok = bool(row["kmeans_radial_ok"])
        step_ok = bool(row["step_ok"])
        kmeans_support = bool(
            kmeans_ok
            and (float(row["kmeans_radial_agreement"]) >= 0.64)
            and (float(row["kmeans_radial_balance"]) >= 0.12)
        )
        step_support = bool(step_ok and (float(row["step_score"]) >= 0.75))
        structure_votes = int(kmeans_support) + int(step_support)

        strong_color_evidence = bool(
            (sat_delta >= 22.0) or (hue_delta >= 10.0 and sat_delta >= 10.0) or (color_delta >= abs_vhi)
        )
        evidence = int(strong_delta) + int(outlier_vote) + structure_votes + int(strong_color_evidence)

        if strong_color_evidence and very_strong_delta and (outlier_vote or kmeans_support):
            detector_type = "bi-metal-like"
        elif strong_color_evidence and strong_delta and ((outlier_vote and structure_votes >= 1) or kmeans_support):
            detector_type = "bi-metal-like"
        elif strong_color_evidence and mid_delta and outlier_vote and (kmeans_support and step_support):
            detector_type = "bi-metal-like"
        elif (not outlier_vote) and (color_delta < 20.0) and (sat_delta <= 18.0) and (hue_delta <= 8.0):
            detector_type = "one-color-like"
        elif (color_delta <= 14.0) and (sat_delta <= 12.0) and (hue_delta <= 10.0):
            detector_type = "one-color-like"
        elif (color_delta <= abs_lo) and (structure_votes <= 1):
            detector_type = "one-color-like"
        elif (not strong_color_evidence) and (not outlier_vote) and (color_delta <= (abs_hi + 2.0)):
            detector_type = "one-color-like"
        elif (
            (not outlier_vote)
            and (not kmeans_support)
            and strong_color_evidence
            and (int(inner_hsv[0]) <= 15)
            and (int(inner_hsv[1]) >= 120)
        ):
            detector_type = "one-color-like"
        else:
            detector_type = "uncertain"

        inner_h = int(inner_hsv[0])
        inner_s = int(inner_hsv[1])
        inner_v = int(inner_hsv[2])
        border_h = int(border_hsv[0])
        border_s = int(border_hsv[1])
        kmeans_score = float(row["kmeans_radial_score"])
        step_score = float(row["step_score"])
        sat_mean = 0.5 * (float(inner_s) + float(border_s))

        promote_uncertain_to_bimetal = bool(
            (detector_type == "uncertain")
            and (float(r) >= 34.0)
            and (sat_mean < 60.0)
            and (val_delta >= 8.0)
            and (color_delta >= 22.0)
            and ((kmeans_score >= 0.62) or (step_score >= 0.55))
        )
        if promote_uncertain_to_bimetal:
            detector_type = "bi-metal-like"

        false_bimetal_copper = bool(
            (detector_type == "bi-metal-like")
            and (inner_h <= 18)
            and (border_h <= 18)
            and (inner_s >= 140)
            and (border_s >= 95)
            and (kmeans_score < 0.85)
        )
        false_bimetal_warm_uniform = bool(
            (detector_type == "bi-metal-like")
            and (inner_h >= 19)
            and (border_h >= 19)
            and (hue_delta <= 2.0)
            and (min(inner_s, border_s) >= 100)
            and (kmeans_score < 0.85)
            and (sat_delta < 30.0)
            and (color_delta < 30.0)
        )
        if false_bimetal_copper or false_bimetal_warm_uniform:
            detector_type = "one-color-like"
            bronze_veto_applied = True

        if detector_type == "bi-metal-like":
            bimetal_euro_label = label_bimetal_euro_from_saturation(inner_s, border_s)
            final_label = bimetal_euro_label
            bronze_score = 0.0
            bronze_score_hsv = 0.0
            bronze_score_lab = 0.0
            bronze_score_hybrid = 0.0
            gold_score_hsv = 0.0
            gold_score_lab = 0.0
            gold_score_hybrid = 0.0
            material_label_hsv = "n/a"
            material_label_lab = "n/a"
            material_label_hybrid = "n/a"
            material_label = "n/a"
        else:
            bimetal_euro_label = "n/a"

            bronze_score_hsv = bronze_score_from_inner_hsv(inner_h, inner_s, inner_v)
            gold_score_hsv = gold_score_from_inner_hsv(inner_h, inner_s, inner_v)
            bronze_score_lab = bronze_score_hsv
            gold_score_lab = gold_score_hsv
            bronze_score_hybrid = bronze_score_hsv
            gold_score_hybrid = gold_score_hsv
            material_label_hsv = label_material_from_inner_hsv(inner_h, inner_s, inner_v)
            material_label_lab = material_label_hsv
            material_label_hybrid = fallback_material_from_inner_hsv(inner_h, inner_s, inner_v)
            bronze_score = bronze_score_hsv
            material_label = material_label_hybrid

            if detector_type == "uncertain":
                final_label = "uncertain"
            elif material_label == "borderline":
                final_label = "one-color-like/borderline"
            else:
                final_label = f"one-color-like/{material_label}"

        full_bgr = cv2.cvtColor(np.uint8([[[*full_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        inner_bgr = cv2.cvtColor(np.uint8([[[*inner_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        border_bgr = cv2.cvtColor(np.uint8([[[*border_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]

        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), r, tuple(int(c) for c in border_bgr), -1)
            cv2.circle(output_bgr, (x, y), inner_r, tuple(int(c) for c in inner_bgr), -1)
        else:
            cv2.circle(output_bgr, (x, y), r, tuple(int(c) for c in full_bgr), -1)

        cv2.circle(output_bgr, (x, y), r, (0, 0, 0), 2)
        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), inner_r, (0, 0, 0), 2)

        text = str(row["id"] + 1)
        if detector_type == "bi-metal-like":
            if bimetal_euro_label == "2-euro-like":
                short_label = "(2)"
            elif bimetal_euro_label == "1-euro-like":
                short_label = "(1)"
            else:
                short_label = "(?)"
        else:
            if material_label == "gold":
                short_label = "G"
            elif material_label == "bronze":
                short_label = "B"
            else:
                short_label = "?"

        text_fg = (0, 220, 255) if detector_type == "uncertain" else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_scale = 0.55
        lbl_scale = 0.45
        num_size, _ = cv2.getTextSize(text, font, num_scale, 1)
        lbl_size, _ = cv2.getTextSize(short_label, font, lbl_scale, 1)
        num_org = (int(x) - (num_size[0] // 2), int(y) - 2)
        lbl_org = (int(x) - (lbl_size[0] // 2), int(y) + 14)
        cv2.putText(output_bgr, text, num_org, font, num_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, text, num_org, font, num_scale, text_fg, 1, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, text_fg, 1, cv2.LINE_AA)

        row["color_delta_threshold"] = float(outlier_threshold)
        row["abs_threshold_mid"] = float(abs_mid)
        row["abs_threshold_hi"] = float(abs_hi)
        row["abs_threshold_vhi"] = float(abs_vhi)
        row["decision_evidence"] = int(evidence)
        row["bronze_score"] = float(bronze_score)
        row["bronze_score_hsv"] = float(bronze_score_hsv)
        row["bronze_score_lab"] = float(bronze_score_lab)
        row["bronze_score_hybrid"] = float(bronze_score_hybrid)
        row["gold_score_hsv"] = float(gold_score_hsv)
        row["gold_score_lab"] = float(gold_score_lab)
        row["gold_score_hybrid"] = float(gold_score_hybrid)
        row["material_label_hsv"] = material_label_hsv
        row["material_label_lab"] = material_label_lab
        row["material_label_hybrid"] = material_label_hybrid
        row["material_label"] = material_label
        row["detector_type"] = detector_type
        row["bronze_veto"] = bool(bronze_veto_applied)
        row["bimetal_euro_label"] = bimetal_euro_label
        row["bimetal_mode"] = bimetal_mode
        row["material_mode"] = material_mode
        row["short_label"] = short_label
        row["label"] = final_label
        row["type"] = detector_type
        stats.append(row)

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats

@dataclass
class ValueEstimationResult:
    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int


@dataclass
class CoinValueEstimationOutput:
    split_rgb: np.ndarray
    split_stats: list[dict]
    value_labeled_rgb: np.ndarray
    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int


def _coin_marker_token(index: int) -> str:
    """Convert zero-based index into Excel-like marker tokens: A, B, ..., Z, AA, AB..."""
    n = max(0, int(index))
    out: list[str] = []
    while True:
        n, rem = divmod(n, 26)
        out.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(out))


class ValueEstimator:
    EURO_DIAMETER_MM: dict[str, float] = {
        "1c": 16.25,
        "2c": 18.75,
        "5c": 21.25,
        "10c": 19.75,
        "20c": 22.25,
        "50c": 24.25,
        "1e": 23.25,
        "2e": 25.75,
    }

    FAMILY_TO_DENOMS: dict[str, list[str]] = {
        "bronze": ["1c", "2c", "5c"],
        "gold": ["10c", "20c", "50c"],
        "bimetal": ["1e", "2e"],
        "unknown": ["1c", "2c", "5c", "10c", "20c", "50c"],
    }

    DENOM_TEXT: dict[str, str] = {
        "1c": "1c",
        "2c": "2c",
        "5c": "5c",
        "10c": "10c",
        "20c": "20c",
        "50c": "50c",
        "1e": "1EUR",
        "2e": "2EUR",
    }

    DENOM_TO_CENTS: dict[str, int] = {
        "1c": 1,
        "2c": 2,
        "5c": 5,
        "10c": 10,
        "20c": 20,
        "50c": 50,
        "1e": 100,
        "2e": 200,
    }

    DENOM_PRINT_ORDER: list[str] = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    @classmethod
    def estimate_from_stats(cls, rows: list[dict]) -> ValueEstimationResult:
        predictions, scale_info, family_models = cls._estimate_coin_values_from_stats(rows)
        counts, total_cents = cls.summarize_prediction_totals(predictions)
        return ValueEstimationResult(
            predictions=predictions,
            scale_info=scale_info,
            family_models=family_models,
            counts=counts,
            total_cents=total_cents,
        )

    @classmethod
    def coin_family_from_row(cls, row: dict) -> str:
        if row.get("type") == "bi-metal-like":
            return "bimetal"

        material = row.get("material_label")
        if material in ("bronze", "gold"):
            return material

        inner_hsv = row.get("inner_hsv")
        material_hint = material_family_from_inner_hsv(inner_hsv) if isinstance(inner_hsv, (tuple, list)) else "unknown"
        if material_hint in ("bronze", "gold"):
            return material_hint

        return "unknown"

    @classmethod
    def summarize_prediction_totals(cls, predictions: dict[int, dict]) -> tuple[dict[str, int], int]:
        counts = {d: 0 for d in cls.DENOM_PRINT_ORDER}
        total_cents = 0
        for pred in predictions.values():
            denom = pred.get("best_denom")
            if denom not in cls.DENOM_TO_CENTS:
                continue
            counts[denom] += 1
            total_cents += int(cls.DENOM_TO_CENTS[denom])
        return counts, int(total_cents)

    @classmethod
    def prob_string(cls, prob_map: dict[str, float]) -> str:
        ordered = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(f"{cls.DENOM_TEXT[k]}={100.0 * v:.1f}%" for k, v in ordered)

    @classmethod
    def draw_coin_value_labels(
        cls,
        image_bgr: np.ndarray,
        rows: list[dict],
        predictions: dict[int, dict],
    ) -> np.ndarray:
        out = image_bgr.copy()
        family_colors = {
            "bronze": (40, 120, 230),
            "gold": (0, 200, 255),
            "bimetal": (255, 180, 40),
            "unknown": (180, 180, 180),
        }

        for row in rows:
            coin_id = int(row["id"])
            pred = predictions.get(coin_id)
            if pred is None:
                continue

            x = int(row["x"])
            y = int(row["y"])
            r = int(row["r"])
            family = pred["family"]
            color = family_colors.get(family, (180, 180, 180))

            cv2.circle(out, (x, y), r, color, 2)
            cv2.circle(out, (x, y), 2, (0, 0, 0), 3)
            marker = _coin_marker_token(coin_id)
            marker_radius = max(8, min(16, int(0.34 * max(1, r))))
            cv2.circle(out, (x, y), marker_radius, (15, 15, 15), -1)
            cv2.circle(out, (x, y), marker_radius, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52 if len(marker) == 1 else 0.42
            thickness = 1
            text_size, _ = cv2.getTextSize(marker, font, scale, thickness)
            tx = int(x) - (text_size[0] // 2)
            ty = int(y) + (text_size[1] // 2)
            cv2.putText(out, marker, (tx, ty), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, marker, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    @classmethod
    def _kmeans_1d_sorted(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        n = int(vals.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        k = int(max(1, min(k, n)))
        if k == 1:
            center = np.array([float(np.mean(vals[:, 0]))], dtype=np.float32)
            labels = np.zeros((n,), dtype=np.int32)
            return labels, center

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.02)
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(vals, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        centers = centers[:, 0]
        order = np.argsort(centers)
        centers_sorted = centers[order]
        remap = np.zeros((k,), dtype=np.int32)
        for rank, old_idx in enumerate(order):
            remap[int(old_idx)] = int(rank)

        labels_flat = labels[:, 0].astype(np.int32)
        labels_sorted = remap[labels_flat]
        return labels_sorted, centers_sorted.astype(np.float32)

    @classmethod
    def _kmeans_1d_with_stats(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        labels, centers = cls._kmeans_1d_sorted(values, k)
        vals = np.asarray(values, dtype=np.float64)

        if vals.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)
        if centers.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)

        labels_i = labels.astype(np.int32)
        pred = centers[labels_i].astype(np.float64)
        inertia = float(np.sum((vals - pred) ** 2))
        counts = np.bincount(labels_i, minlength=int(centers.size)).astype(np.int32)
        return labels_i, centers.astype(np.float32), inertia, counts

    @classmethod
    def _choose_k_for_family(cls, values: list[float], max_k: int, family: str) -> tuple[int, dict]:
        vals = np.asarray(values, dtype=np.float64)
        n = int(vals.size)
        if n <= 1:
            return 1, {"reason": "n<=1", "evaluated": []}

        uniq = int(np.unique(np.round(vals, 2)).size)
        max_k = int(max(1, min(max_k, n, uniq)))
        if max_k <= 1:
            return 1, {"reason": "max_k<=1", "evaluated": []}

        min_count_abs = 1 if n < 6 else 2
        min_count_frac = 0.14 if family in ("gold", "bronze") else 0.10

        rows = []
        for k in range(1, max_k + 1):
            _, centers, inertia, counts = cls._kmeans_1d_with_stats(vals.tolist(), k)
            sigma2 = max(float(inertia) / max(n, 1), 1e-6)
            dof = max(1, 2 * k - 1)
            bic = float(n * np.log(sigma2) + dof * np.log(max(n, 2)))

            min_count = int(np.min(counts)) if counts.size > 0 else 0
            min_needed = max(int(min_count_abs), int(np.ceil(min_count_frac * n)))
            tiny = max(0, min_needed - min_count)
            if tiny > 0:
                bic += float(4.0 * tiny)

            rows.append(
                {
                    "k": int(k),
                    "bic": float(bic),
                    "inertia": float(inertia),
                    "min_count": int(min_count),
                    "counts": [int(c) for c in counts.tolist()],
                }
            )

        best = min(rows, key=lambda r: r["bic"])
        best_k = int(best["k"])
        if best_k > 1:
            prev = [r for r in rows if r["k"] == (best_k - 1)][0]
            rel_gain = float((prev["inertia"] - best["inertia"]) / (prev["inertia"] + 1e-9))
            min_gain = {"gold": 0.12, "bronze": 0.10, "bimetal": 0.08, "unknown": 0.14}.get(family, 0.10)
            if rel_gain < min_gain:
                best_k = int(best_k - 1)

        return best_k, {"reason": "bic+min_cluster", "evaluated": rows, "chosen_k": int(best_k)}

    @classmethod
    def _best_denom_subset_for_centers(
        cls,
        sorted_centers_px: np.ndarray,
        candidate_denoms: list[str],
        px_per_mm: float | None,
    ) -> list[str]:
        k = int(len(sorted_centers_px))
        if k <= 0:
            return []
        if k >= len(candidate_denoms):
            return candidate_denoms[:]

        centers = np.asarray(sorted_centers_px, dtype=np.float64)
        best_score = 1e18
        best: list[str] | None = None

        for comb in combinations(candidate_denoms, k):
            half_mm = np.array([0.5 * cls.EURO_DIAMETER_MM[d] for d in comb], dtype=np.float64)
            if px_per_mm is not None and px_per_mm > 0:
                expected_px = half_mm * float(px_per_mm)
                denom = np.mean(expected_px ** 2) + 1e-9
                score = float(np.mean((centers - expected_px) ** 2) / denom)
            else:
                c = centers / (np.median(centers) + 1e-9)
                e = half_mm / (np.median(half_mm) + 1e-9)
                score = float(np.mean((c - e) ** 2))
            if score < best_score:
                best_score = score
                best = list(comb)

        return best if best is not None else candidate_denoms[:k]

    @classmethod
    def _estimate_scale_from_all_radii_voting(cls, rows: list[dict]) -> tuple[float | None, dict]:
        scale_candidates: list[float] = []
        for row in rows:
            r_px = float(max(1, int(row.get("r", 1))))
            family = cls.coin_family_from_row(row)
            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            for denom in denom_pool:
                d_mm = cls.EURO_DIAMETER_MM[denom]
                scale_candidates.append((2.0 * r_px) / d_mm)

        if len(scale_candidates) == 0:
            return None, {"support": 0, "total": 0}

        arr = np.asarray(scale_candidates, dtype=np.float64)
        tol = 0.18
        diffs = np.abs(arr[:, None] - arr[None, :])
        support_counts = np.sum(diffs <= tol, axis=1)
        best_idx = int(np.argmax(support_counts))
        support_mask = diffs[best_idx] <= tol
        robust_scale = float(np.median(arr[support_mask]))
        return robust_scale, {"support": int(np.sum(support_mask)), "total": int(arr.size), "window_tol": float(tol)}

    @classmethod
    def _estimate_px_per_mm(cls, rows: list[dict]) -> tuple[float | None, dict]:
        ref_scales: list[float] = []
        for row in rows:
            if row.get("type") != "bi-metal-like":
                continue
            label = row.get("bimetal_euro_label", "")
            if label == "1-euro-like":
                denom = "1e"
            elif label == "2-euro-like":
                denom = "2e"
            else:
                continue
            r_px = float(max(1, int(row.get("r", 1))))
            ref_scales.append((2.0 * r_px) / cls.EURO_DIAMETER_MM[denom])

        fallback_scale, fallback_dbg = cls._estimate_scale_from_all_radii_voting(rows)
        if len(ref_scales) >= 2:
            return float(np.median(np.asarray(ref_scales, dtype=np.float64))), {
                "method": "bimetal_reference",
                "count": int(len(ref_scales)),
                "raw_scales": [float(x) for x in ref_scales],
            }
        if len(ref_scales) == 1:
            ref = float(ref_scales[0])
            info = {"method": "single_bimetal_reference", "count": 1, "raw_scales": [float(ref)]}
            if fallback_scale is not None:
                info["fallback_scale"] = float(fallback_scale)
                info.update(fallback_dbg)
            return ref, info
        if fallback_scale is not None:
            return float(fallback_scale), {"method": "all_radii_voting", **fallback_dbg}
        return None, {"method": "none", "count": 0}

    @classmethod
    def _build_family_radius_models(cls, rows: list[dict], px_per_mm: float | None) -> dict:
        models = {}
        for family, denom_pool in cls.FAMILY_TO_DENOMS.items():
            if family == "bimetal":
                continue
            fam_rows = [row for row in rows if cls.coin_family_from_row(row) == family]
            if len(fam_rows) == 0:
                continue

            radii = [float(row["r"]) for row in fam_rows]
            max_k = min(len(denom_pool), len(radii))
            k, k_dbg = cls._choose_k_for_family(radii, max_k=max_k, family=family)
            labels, centers, _, counts = cls._kmeans_1d_with_stats(radii, k)
            subset = cls._best_denom_subset_for_centers(centers, denom_pool, px_per_mm=px_per_mm)

            cluster_to_denom = {int(i): subset[i] for i in range(len(subset))}
            coin_to_cluster = {int(fam_rows[i]["id"]): int(labels[i]) for i in range(len(fam_rows))}
            models[family] = {
                "k": int(k),
                "centers_px": [float(c) for c in centers.tolist()],
                "cluster_counts": [int(c) for c in counts.tolist()],
                "subset_denoms": subset,
                "cluster_to_denom": cluster_to_denom,
                "coin_to_cluster": coin_to_cluster,
                "k_debug": k_dbg,
            }
        return models

    @classmethod
    def _normalize_scores(cls, score_map: dict[str, float]) -> dict[str, float]:
        vals = np.array([max(1e-12, float(v)) for v in score_map.values()], dtype=np.float64)
        total = float(np.sum(vals))
        if total <= 0:
            n = max(1, len(score_map))
            return {k: 1.0 / n for k in score_map.keys()}
        return {k: float(max(1e-12, score_map[k]) / total) for k in score_map.keys()}

    @classmethod
    def _estimate_coin_values_from_stats(cls, rows: list[dict]) -> tuple[dict[int, dict], dict, dict]:
        px_per_mm, scale_info = cls._estimate_px_per_mm(rows)
        family_models = cls._build_family_radius_models(rows, px_per_mm=px_per_mm)
        predictions: dict[int, dict] = {}

        for row in rows:
            coin_id = int(row["id"])
            family = cls.coin_family_from_row(row)

            if family == "bimetal":
                bm = row.get("bimetal_euro_label", "")
                if bm == "1-euro-like":
                    probs = {"1e": 1.0, "2e": 0.0}
                    best_denom = "1e"
                elif bm == "2-euro-like":
                    probs = {"1e": 0.0, "2e": 1.0}
                    best_denom = "2e"
                else:
                    probs = {"1e": 0.5, "2e": 0.5}
                    best_denom = "1e"
                predictions[coin_id] = {
                    "coin_id": int(coin_id),
                    "family": family,
                    "cluster_idx": None,
                    "estimated_diameter_mm": None,
                    "probs": probs,
                    "best_denom": best_denom,
                    "best_label": cls.DENOM_TEXT[best_denom],
                    "best_prob": float(probs[best_denom]),
                }
                continue

            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            model = family_models.get(family, {})
            coin_to_cluster = model.get("coin_to_cluster", {})
            cluster_to_denom = model.get("cluster_to_denom", {})
            cluster_idx = coin_to_cluster.get(coin_id)

            centers_px = np.asarray(model.get("centers_px", []), dtype=np.float64)
            denom_to_center = {}
            for c_idx, d_name in cluster_to_denom.items():
                ci = int(c_idx)
                if 0 <= ci < int(centers_px.size):
                    denom_to_center[d_name] = float(centers_px[ci])

            r_px = float(row["r"])
            d_est_mm = None if px_per_mm is None else (2.0 * r_px / float(px_per_mm))

            raw_scores = {}
            for denom in denom_pool:
                d_ref = cls.EURO_DIAMETER_MM[denom]
                if d_est_mm is None:
                    size_like = 1.0
                else:
                    sigma_map = {"gold": 0.48, "bronze": 0.62, "unknown": 0.85}
                    sigma_mm = float(sigma_map.get(family, 0.80))
                    z = (float(d_est_mm) - float(d_ref)) / sigma_mm
                    size_like = float(np.exp(-0.5 * z * z))

                k_model = int(model.get("k", 0))
                full_k = int(len(denom_pool))
                if cluster_idx is None:
                    cluster_prior = 1.0
                else:
                    mapped = cluster_to_denom.get(int(cluster_idx))
                    if mapped == denom:
                        cluster_prior = 1.0
                    elif k_model >= full_k:
                        strict_map = {"gold": 0.08, "bronze": 0.10, "unknown": 0.30}
                        cluster_prior = float(strict_map.get(family, 0.25))
                    elif k_model >= 2:
                        partial_map = {"gold": 0.46, "bronze": 0.38, "unknown": 0.55}
                        cluster_prior = float(partial_map.get(family, 0.45))
                    else:
                        cluster_prior = 0.75

                if denom in denom_to_center:
                    c_px = float(denom_to_center[denom])
                    sigma_px = max(0.80, 0.055 * c_px)
                    zc = (r_px - c_px) / sigma_px
                    center_prior = float(np.exp(-0.5 * zc * zc))
                elif k_model >= full_k:
                    strict_miss = {"gold": 0.12, "bronze": 0.14, "unknown": 0.35}
                    center_prior = float(strict_miss.get(family, 0.25))
                elif k_model >= 2:
                    partial_miss = {"gold": 0.70, "bronze": 0.66, "unknown": 0.80}
                    center_prior = float(partial_miss.get(family, 0.65))
                else:
                    center_prior = 1.0

                shape_prior = 1.0
                raw_scores[denom] = float(max(1e-9, size_like * cluster_prior * center_prior * shape_prior))

            probs = cls._normalize_scores(raw_scores)
            best_denom = max(probs.keys(), key=lambda d: probs[d])
            predictions[coin_id] = {
                "coin_id": int(coin_id),
                "family": family,
                "cluster_idx": None if cluster_idx is None else int(cluster_idx),
                "estimated_diameter_mm": None if d_est_mm is None else float(d_est_mm),
                "probs": probs,
                "best_denom": best_denom,
                "best_label": cls.DENOM_TEXT[best_denom],
                "best_prob": float(probs[best_denom]),
            }

        scale_info = {**scale_info, "px_per_mm": None if px_per_mm is None else float(px_per_mm)}
        return predictions, scale_info, family_models


class CoinValueEstimator:
    def __init__(
        self,
        border_ratio: float = 0.24,
        sat_delta_threshold: float | None = None,
        bimetal_mode: str = "hybrid",
        material_mode: str = "hsv",
    ):
        self._analyzer = CoinAnalyzer(
            CoinAnalyzerConfig(
                border_ratio=border_ratio,
                sat_delta_threshold=sat_delta_threshold,
                bimetal_mode=bimetal_mode,
                material_mode=material_mode,
            )
        )

    def estimate(self, image_bgr: np.ndarray, circles: np.ndarray | None) -> CoinValueEstimationOutput:
        split_rgb, split_stats = self._analyzer.analyze(image_bgr, circles)

        if len(split_stats) > 0:
            result = ValueEstimator.estimate_from_stats(split_stats)
            value_labeled_rgb = ValueEstimator.draw_coin_value_labels(
                image_bgr,
                split_stats,
                result.predictions,
            )
            return CoinValueEstimationOutput(
                split_rgb=split_rgb,
                split_stats=split_stats,
                value_labeled_rgb=value_labeled_rgb,
                predictions=result.predictions,
                scale_info=result.scale_info,
                family_models=result.family_models,
                counts=result.counts,
                total_cents=int(result.total_cents),
            )

        empty_counts = {d: 0 for d in ValueEstimator.DENOM_PRINT_ORDER}
        return CoinValueEstimationOutput(
            split_rgb=split_rgb,
            split_stats=split_stats,
            value_labeled_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            predictions={},
            scale_info={"method": "none", "count": 0, "px_per_mm": None},
            family_models={},
            counts=empty_counts,
            total_cents=0,
        )
