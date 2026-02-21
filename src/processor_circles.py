from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import letterbox_resize_to_canvas, read_bgr_or_raise


@dataclass
class PipelineStep:
    name: str
    image: np.ndarray
    cmap: str


@dataclass
class PipelineResult:
    source_path: Path
    steps: list[PipelineStep]
    circle_count: int
    hough_params: dict[str, float | int]
    debug_info: dict[str, Any]


class CirclePipelineProcessor:
    def __init__(self, config: PipelineConfig, preset_name: str | None = None):
        self._cfg = config
        self._preset_name = preset_name or config.active_preset
        self._preset = config.get_preset(self._preset_name)

    def process_path(self, image_path: Path) -> PipelineResult:
        image_bgr = read_bgr_or_raise(image_path)
        return self.process_image(image_bgr, image_path)

    def process_image(self, image_bgr: np.ndarray, source_path: Path) -> PipelineResult:
        image_bgr = letterbox_resize_to_canvas(
            image_bgr,
            self._cfg.target_width,
            self._cfg.target_height,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        clahe_bgr, clahe_rgb = apply_clahe_on_l_channel(
            image_bgr,
            clip_limit=self._cfg.clahe_clip_limit,
            tile_grid_size=self._cfg.clahe_tile_grid_size,
        )

        # Notebook keeps grayscale from resized original (not clahe_bgr).
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self._odd(self._cfg.gauss_ksize), self._odd(self._cfg.gauss_ksize)),
            self._cfg.gauss_sigma,
        )

        hough_params: dict[str, float | int] = {
            "dp": self._preset.dp,
            "minDist": self._preset.min_dist,
            "param2": self._preset.param2,
        }
        param1 = auto_hough_param1_from_gradient(
            gray,
            blur_ksize=self._cfg.auto_param1_blur_ksize,
            perc=self._cfg.auto_param1_percentile,
            scale=self._cfg.auto_param1_scale,
            clamp=self._cfg.auto_param1_clamp,
        )
        hough_params["param1"] = int(param1)
        hough_params["maxRadius"] = int(self._cfg.max_radius)

        sweep_end = min(self._cfg.min_radius_sweep_end, int(hough_params["maxRadius"]) - 1)
        if sweep_end < self._cfg.min_radius_sweep_start:
            sweep_end = self._cfg.min_radius_sweep_start
        best_min_radius, sweep_debug, sweep_results = auto_minradius_plateau(
            blurred,
            base_params=hough_params,
            minR_start=self._cfg.min_radius_sweep_start,
            minR_end=sweep_end,
            step=self._cfg.min_radius_sweep_step,
        )
        hough_params["minRadius"] = int(best_min_radius)

        circles = detect_hough_circles(blurred, hough_params)
        circles_overlay, circle_count = draw_circles_on_rgb(
            image_rgb,
            circles,
            outline_color=self._cfg.circle_outline_color,
            outline_thickness=self._cfg.circle_outline_thickness,
            center_color=self._cfg.center_color,
            center_radius=self._cfg.center_radius,
            center_thickness=self._cfg.center_thickness,
        )

        result = PipelineResult(
            source_path=source_path,
            steps=[
                PipelineStep("Original (Letterbox 640x480)", image_rgb, "rgb"),
                PipelineStep("CLAHE (L channel)", clahe_rgb, "rgb"),
                PipelineStep("Grayscale", gray, "gray"),
                PipelineStep("Gaussian Blur", blurred, "gray"),
                PipelineStep("Hough Circles", circles_overlay, "rgb"),
            ],
            circle_count=circle_count,
            hough_params=hough_params,
            debug_info={
                "preset": self._preset_name,
                "plateau_debug": sweep_debug,
                "sweep_results": sweep_results,
                "clahe_bgr": clahe_bgr,
            },
        )
        return result

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1


def apply_clahe_on_l_channel(
    image_bgr: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    clahe_bgr = cv2.cvtColor(cv2.merge((l_clahe, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    return clahe_bgr, clahe_rgb


def auto_hough_param1_from_gradient(
    gray_u8: np.ndarray,
    blur_ksize: int = 5,
    perc: float = 90.0,
    scale: float = 1.0,
    clamp: tuple[int, int] = (20, 250),
) -> int:
    if gray_u8.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)

    if blur_ksize >= 3:
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray_proc = cv2.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)
    else:
        gray_proc = gray_u8

    gx = cv2.Scharr(gray_proc, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_proc, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)

    non_zero = magnitude[magnitude > 1e-3]
    if non_zero.size == 0:
        return 100

    threshold = np.percentile(non_zero, perc) * scale
    return int(np.clip(threshold, clamp[0], clamp[1]))


def detect_hough_circles(blurred_gray: np.ndarray, params: dict[str, float | int]) -> np.ndarray | None:
    circles = cv2.HoughCircles(
        blurred_gray,
        cv2.HOUGH_GRADIENT,
        dp=float(params["dp"]),
        minDist=float(params["minDist"]),
        param1=float(params["param1"]),
        param2=float(params["param2"]),
        minRadius=int(params["minRadius"]),
        maxRadius=int(params["maxRadius"]),
    )
    return circles


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
) -> tuple[int, dict[str, Any], list[tuple[int, dict[str, int | float]]]]:
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

