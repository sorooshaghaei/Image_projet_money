from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .hough_detection import CoinDetector
from .io_utils import letterbox_resize_to_canvas, read_bgr_or_raise
from .preprocessing_ops import ImagePreprocessing
from .value_estimator import CoinValueEstimator, ValueEstimator


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
        preset = config.get_preset(self._preset_name)

        self._preprocessing = ImagePreprocessing(
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_tile_grid_size=config.clahe_tile_grid_size,
            gauss_ksize=config.gauss_ksize,
            gauss_sigma=config.gauss_sigma,
        )
        self._detector = CoinDetector(
            preset=preset,
            min_radius_sweep_start=config.min_radius_sweep_start,
            min_radius_sweep_end=config.min_radius_sweep_end,
            min_radius_sweep_step=config.min_radius_sweep_step,
            max_radius=config.max_radius,
            auto_param1_blur_ksize=config.auto_param1_blur_ksize,
            auto_param1_percentile=config.auto_param1_percentile,
            auto_param1_scale=config.auto_param1_scale,
            auto_param1_clamp=config.auto_param1_clamp,
            circle_outline_color=config.circle_outline_color,
            circle_outline_thickness=config.circle_outline_thickness,
            center_color=config.center_color,
            center_radius=config.center_radius,
            center_thickness=config.center_thickness,
        )
        self._value_estimator = CoinValueEstimator(
            border_ratio=config.analysis_border_ratio,
            sat_delta_threshold=config.analysis_sat_delta_threshold,
            bimetal_mode=config.analysis_bimetal_mode,
            material_mode=config.analysis_material_mode,
        )

    def process_path(self, image_path: Path) -> PipelineResult:
        image_bgr = read_bgr_or_raise(image_path)
        return self.process_image(image_bgr, image_path)

    def process_image(self, image_bgr: np.ndarray, source_path: Path) -> PipelineResult:
        image_bgr = letterbox_resize_to_canvas(
            image_bgr,
            self._cfg.target_width,
            self._cfg.target_height,
        )

        prep = self._preprocessing.process(image_bgr)
        detection = self._detector.detect(prep.gray, prep.blurred, prep.image_rgb)
        valuation = self._value_estimator.estimate(image_bgr, detection.circles)

        value_counts = valuation.counts if len(valuation.counts) > 0 else {d: 0 for d in ValueEstimator.DENOM_PRINT_ORDER}

        return PipelineResult(
            source_path=source_path,
            steps=[
                PipelineStep("Original (Letterbox 640x480)", prep.image_rgb, "rgb"),
                PipelineStep("CLAHE (L channel)", prep.clahe_rgb, "rgb"),
                PipelineStep("Grayscale", prep.gray, "gray"),
                PipelineStep("Gaussian Blur", prep.blurred, "gray"),
                PipelineStep("Hough Circles", detection.circles_overlay, "rgb"),
                PipelineStep("Inner/Border Mean Color Analysis", valuation.split_rgb, "rgb"),
                PipelineStep("Coin Value Estimation", valuation.value_labeled_rgb, "rgb"),
            ],
            circle_count=detection.circle_count,
            hough_params=detection.hough_params,
            debug_info={
                "preset": self._preset_name,
                "plateau_debug": detection.sweep_debug,
                "sweep_results": detection.sweep_results,
                "clahe_bgr": prep.clahe_bgr,
                "split_stats": valuation.split_stats,
                "value_predictions": valuation.predictions,
                "value_scale_info": valuation.scale_info,
                "family_models": valuation.family_models,
                "value_counts": value_counts,
                "total_cents": int(valuation.total_cents),
            },
        )
