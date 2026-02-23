"""Routing and heuristic policies."""

from __future__ import annotations

import cv2
import numpy as np


def run_hough_with_params(blurred: np.ndarray, hough_params: dict[str, float | int]) -> np.ndarray | None:
    """Run HoughCircles with explicit params and return integer circles."""
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(hough_params["dp"]),
        minDist=float(hough_params["minDist"]),
        param1=float(hough_params["param1"]),
        param2=float(hough_params["param2"]),
        minRadius=int(hough_params["minRadius"]),
        maxRadius=int(hough_params["maxRadius"]),
    )
    if circles is None:
        return None
    return np.round(circles[0, :]).astype(int)


def circle_nesting_score(circles_int: np.ndarray | None) -> tuple[float, dict[str, int | float]]:
    """Geometric penalty for overlapping/nested circle detections."""
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
    """Pick a robust Hough `minRadius` via sweep + stability voting.

    Why this exists
    ---------------
    `minRadius` is one of the most sensitive Hough parameters:
    - too small -> many duplicate/nested detections,
    - too large -> small valid coins disappear.

    Instead of fixing one value globally, we sweep a radius range and score each
    result with `circle_nesting_score`, then select a *stable* region.

    Selection policy
    ----------------
    1. Evaluate every candidate `minRadius` in `[minR_start, minR_end]`.
    2. Keep only geometrically clean results:
       - `nested_pairs == 0`
       - `max_intrusions_in_one == 0`
    3. If no clean result exists, fall back to a least-bad compromise by:
       - minimizing nested pairs,
       - then minimizing intrusions,
       - then maximizing detected count `n`,
       - and finally choosing the median radius among ties.
    4. If clean results exist, group by detected count `n`, choose the modal
       count (prefer larger `n` on ties), then take the median radius in that
       group.

    The median tie-break is intentional: it avoids edge-of-range picks and
    favors parameter values in the middle of stable plateaus.
    """
    results: list[tuple[int, dict[str, int | float]]] = []

    # Phase 1: sweep `minRadius` and collect geometric quality metrics.
    for min_radius in range(minR_start, minR_end + 1, step):
        params = base_params.copy()
        params["minRadius"] = int(min_radius)
        circles_int = run_hough_with_params(blurred, params)
        _, metrics = circle_nesting_score(circles_int)
        results.append((min_radius, metrics))

    # Phase 2: keep only clean candidates (no nesting, no severe intrusions).
    good = [
        (min_radius, metrics)
        for (min_radius, metrics) in results
        if metrics["nested_pairs"] == 0 and metrics["max_intrusions_in_one"] == 0
    ]

    # Fallback path when *all* candidates are imperfect.
    if len(good) == 0:
        # First objective: minimize nesting artefacts.
        best_nested = min(int(metrics["nested_pairs"]) for (_, metrics) in results)
        candidates = [(min_radius, metrics) for (min_radius, metrics) in results if int(metrics["nested_pairs"]) == best_nested]

        # Second objective: minimize heavy intrusions.
        best_intrusions = min(int(metrics["max_intrusions_in_one"]) for (_, metrics) in candidates)
        candidates = [
            (min_radius, metrics)
            for (min_radius, metrics) in candidates
            if int(metrics["max_intrusions_in_one"]) == best_intrusions
        ]

        # Third objective: among similarly clean candidates, keep more detections.
        best_n = max(int(metrics["n"]) for (_, metrics) in candidates)
        min_radius_list = sorted(min_radius for (min_radius, metrics) in candidates if int(metrics["n"]) == best_n)

        # Final tie-break: middle value for plateau robustness.
        chosen = min_radius_list[len(min_radius_list) // 2]
        return chosen, {
            "reason": "no_good_results",
            "best_nested": best_nested,
            "best_intrusions": best_intrusions,
            "best_n": best_n,
        }, results

    # Main path: analyze stable clean regions by detected-count frequency.
    n_to_min_radius: dict[int, list[int]] = {}
    for min_radius, metrics in good:
        n = int(metrics["n"])
        n_to_min_radius.setdefault(n, []).append(int(min_radius))

    # Prefer positive counts when available; otherwise allow zero-only plateaus.
    positive_counts = {n: values for n, values in n_to_min_radius.items() if n > 0}
    selected_counts = positive_counts if len(positive_counts) > 0 else n_to_min_radius

    # Mode of count-frequency: choose the count that remains valid longest
    # across the sweep; prefer higher count in case of equal frequency.
    counts = [(n, len(values)) for n, values in selected_counts.items()]
    max_frequency = max(freq for _, freq in counts)
    n_candidates = [n for n, freq in counts if freq == max_frequency]
    n_mode = max(n_candidates)

    # Final robust pick inside the selected plateau.
    min_radius_candidates = sorted(selected_counts[n_mode])
    chosen_min_radius = min_radius_candidates[len(min_radius_candidates) // 2]

    return int(chosen_min_radius), {
        "n_mode": n_mode,
        "freq": max_frequency,
        "min_radius_candidates": min_radius_candidates,
    }, results


def choose_dynamic_sat_delta_threshold(sat_deltas: list[float], default: float = 18.0) -> float:
    """Infer robust color-delta split threshold from ordered samples."""
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


def radius_spread_metric(circles_int: np.ndarray | None) -> float:
    """Normalized radius spread (IQR/median) used as consistency indicator."""
    if circles_int is None or len(circles_int) == 0:
        return 1e9
    r = circles_int[:, 2].astype(np.float32)
    med = float(np.median(r))
    if med <= 1e-6:
        return 1e9
    q1, q3 = np.percentile(r, [25, 75])
    iqr = float(q3 - q1)
    return iqr / med
