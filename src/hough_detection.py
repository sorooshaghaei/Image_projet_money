from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import HoughPreset
from .preprocessing_ops import auto_hough_param1_from_gradient


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
