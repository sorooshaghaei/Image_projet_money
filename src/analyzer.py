"""Main orchestrator: preprocessing + policy + detectors -> AnalysisResult."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.config import PipelineConfig
from src.detectors import CoinDetector, CoinValueEstimator, ValueEstimator
from src.io_utils import letterbox_resize_to_canvas, read_bgr_or_raise
from src.models import AnalysisResult, PipelineStep
from src.preprocessing import ImagePreprocessing, normalize_odd_ksize


class Analyzer:
    """Project analyzer orchestrating the modular pipeline."""

    def __init__(self, config: PipelineConfig | None = None, preset_name: str | None = None):
        self._cfg = config or PipelineConfig()
        self._preset_name = preset_name or self._cfg.active_preset
        self._cfg = replace(self._cfg, active_preset=self._preset_name)

        preset = self._cfg.get_preset(self._preset_name)
        self._preprocessing = ImagePreprocessing(
            clahe_enabled=self._cfg.clahe_enabled,
            clahe_clip_limit=self._cfg.clahe_clip_limit,
            clahe_tile_grid_size=self._cfg.clahe_tile_grid_size,
            histogram_normalization_enabled=self._cfg.histogram_normalization_enabled,
            histogram_clip_limit=self._cfg.histogram_clip_limit,
            histogram_tile_grid_size=self._cfg.histogram_tile_grid_size,
            histogram_stretch_percentiles=self._cfg.histogram_stretch_percentiles,
            blur_mode=self._cfg.blur_mode,
            gauss_ksize=self._cfg.gauss_ksize,
            gauss_sigma=self._cfg.gauss_sigma,
        )
        self._detector = CoinDetector(
            preset=preset,
            min_radius_sweep_start=self._cfg.min_radius_sweep_start,
            min_radius_sweep_end=self._cfg.min_radius_sweep_end,
            min_radius_sweep_step=self._cfg.min_radius_sweep_step,
            max_radius=self._cfg.max_radius,
            auto_param1_blur_ksize=self._cfg.auto_param1_blur_ksize,
            auto_param1_percentile=self._cfg.auto_param1_percentile,
            auto_param1_scale=self._cfg.auto_param1_scale,
            auto_param1_clamp=self._cfg.auto_param1_clamp,
            circle_outline_color=self._cfg.circle_outline_color,
            circle_outline_thickness=self._cfg.circle_outline_thickness,
            center_color=self._cfg.center_color,
            center_radius=self._cfg.center_radius,
            center_thickness=self._cfg.center_thickness,
        )
        self._value_estimator = CoinValueEstimator(
            border_ratio=self._cfg.analysis_border_ratio,
            sat_delta_threshold=self._cfg.analysis_sat_delta_threshold,
            bimetal_mode=self._cfg.analysis_bimetal_mode,
            material_mode=self._cfg.analysis_material_mode,
        )

    @property
    def config(self) -> PipelineConfig:
        return self._cfg

    def analyze_path(self, image_path: Path) -> AnalysisResult:
        """Read one image from disk and run the full analysis pipeline."""

        image_bgr = read_bgr_or_raise(image_path)
        return self.analyze(image_bgr, source_path=image_path)

    def analyze(self, image_bgr: np.ndarray, source_path: Path | None = None) -> AnalysisResult:
        """Run analysis for one image matrix."""

        image_bgr = letterbox_resize_to_canvas(
            image_bgr,
            self._cfg.target_width,
            self._cfg.target_height,
        )

        prep = self._preprocessing.process(image_bgr)
        detection_gray = cv2.cvtColor(prep.image_bgr, cv2.COLOR_BGR2GRAY)
        ksize = normalize_odd_ksize(self._cfg.gauss_ksize)
        if self._cfg.blur_mode == "gauss":
            detection_blurred = cv2.GaussianBlur(detection_gray, (ksize, ksize), self._cfg.gauss_sigma)
        else:
            detection_blurred = detection_gray if ksize <= 1 else cv2.medianBlur(detection_gray, ksize)

        detection = self._detector.detect(detection_gray, detection_blurred, prep.image_rgb)
        valuation = self._value_estimator.estimate(prep.image_bgr, detection.circles)

        value_counts = (
            valuation.counts
            if len(valuation.counts) > 0
            else {d: 0 for d in ValueEstimator.DENOM_PRINT_ORDER}
        )
        steps = [
            PipelineStep("Original (Letterbox 640x480)", prep.image_rgb, "rgb"),
            PipelineStep("Grayscale", prep.gray, "gray"),
            PipelineStep(self._preprocessing.blur_step_name, prep.blurred, "gray"),
            PipelineStep("Hough Circles", detection.circles_overlay, "rgb"),
            PipelineStep("Inner/Border Mean Color Analysis", valuation.split_rgb, "rgb"),
            PipelineStep("Coin Value Estimation", valuation.value_labeled_rgb, "rgb"),
        ]
        debug_info = {
            "preset": self._preset_name,
            "plateau_debug": detection.sweep_debug,
            "sweep_results": detection.sweep_results,
            "clahe_enabled": bool(self._cfg.clahe_enabled),
            "blur_mode": self._cfg.blur_mode,
            "histogram_normalization_enabled": False,
            "hist_norm_debug": prep.hist_norm_debug,
            "valuation_input_stage": "image_bgr",
            "split_stats": valuation.split_stats,
            "value_predictions": valuation.predictions,
            "value_scale_info": valuation.scale_info,
            "family_models": valuation.family_models,
            "value_counts": value_counts,
            "total_cents": int(valuation.total_cents),
        }

        return AnalysisResult(
            source_path=source_path,
            steps=steps,
            circle_count=detection.circle_count,
            hough_params=detection.hough_params,
            debug_info=debug_info,
        )


__all__ = ["Analyzer"]
