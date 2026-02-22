from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class HoughSettings:
    """Parameters for circle proposals in textured scenes."""

    dp: float = 1.2
    min_dist: int = 57
    param1: int = 120
    param2: int = 38
    min_radius: int = 10
    max_radius: int = 160


@dataclass(frozen=True)
class ContourSettings:
    """Parameters for contour-based detection on clean backgrounds."""

    blur_kernel: int = 5
    morph_kernel: int = 3
    min_area: int = 320
    min_circularity: float = 0.85
    max_aspect_ratio: float = 1.45


@dataclass(frozen=True)
class WatershedSettings:
    """Parameters for separating touching or overlapping coins."""

    open_kernel: int = 3
    close_kernel: int = 5
    fg_ratio: float = 0.27
    min_seed_area: int = 120
    min_region_area: int = 450


@dataclass(frozen=True)
class PolicySettings:
    """Scene policy for difficulty labels and algorithm routing."""

    easy_threshold: float = 0.23
    medium_threshold: float = 0.43
    overlap_distance_scale: float = 0.88
    contour_merge_circularity: float = 0.72
    contour_merge_area_ratio: float = 0.05
    overlap_merge_score: float = 0.20
    target_width: int = 1100
    min_coin_radius_px: int = 9
    max_coin_radius_px: int = 180
    median_radius_low_scale: float = 0.45
    median_radius_high_scale: float = 2.20
    max_label_rel_error: float = 0.12
    color_unknown_label_threshold: float = 0.20
    color_mismatch_penalty: float = 0.12
    bimetal_high_confidence: float = 0.70
    bimetal_strong_2e_margin: float = 0.14
    denom_radius_rel_tol: float = 0.22
    coin_mm_min: float = 16.25
    coin_mm_max: float = 25.75
    coin_mm_rel_tol: float = 0.10
    dynamic_hough_enabled: bool = True
    dynamic_hough_count_threshold: int = 18
    dynamic_hough_min_dist_scale: float = 1.05
    dynamic_hough_param2_scale: float = 1.10
    sparse_rescue_max_base_count: int = 5
    sparse_rescue_min_dist_scale: float = 0.72
    sparse_rescue_param2_scale: float = 0.68
    sparse_rescue_dp_scale: float = 1.10
    sparse_rescue_sat_median_max: float = 145.0
    sparse_rescue_sat_p75_max: float = 130.0
    sparse_rescue_min_mask_coverage: float = 0.35
    color_scene_sat_low: float = 70.0
    color_scene_sat_high: float = 150.0
    color_scene_sat_p75_high: float = 200.0
    # Classical value calibration (fitted once on development dataset):
    # calibrated = alpha * raw_value + beta * coin_count + bias
    value_calibration_enabled: bool = True
    value_calibration_alpha: float = 0.39004081
    value_calibration_count_beta: float = 0.19950035
    value_calibration_bias: float = 1.56925937


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime defaults for CLI entrypoints."""

    image_directory: str = field(default_factory=lambda: _find_image_directory())
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    report_csv_path: str = field(default_factory=lambda: _default_report_csv_path())
    dataset_eval_csv_path: str = field(default_factory=lambda: _default_dataset_eval_csv_path())


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "images").exists():
            return parent
    return (current.parent / "..").resolve()


def _find_image_directory() -> str:
    """Locate `data/images` relative to this package."""
    return (_find_project_root() / "data" / "images").resolve().as_posix()


def _default_report_csv_path() -> str:
    return (_find_project_root() / "report" / "runtime_v2_policy_trace.csv").resolve().as_posix()


def _default_dataset_eval_csv_path() -> str:
    return (_find_project_root() / "report" / "runtime_v2_dataset_eval.csv").resolve().as_posix()
