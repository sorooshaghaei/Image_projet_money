"""Main orchestrator for the image analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common.image_io import letterbox_resize_to_canvas, read_bgr_or_raise
from src.pipeline.config import PipelineConfig
from src.pipeline.detectors import CoinDetector, CoinValueEstimator, ValueEstimator
from src.pipeline.models import AnalysisResult, PipelineStep
from src.pipeline.preprocessing import ImagePreprocessing, normalize_odd_ksize


class Analyzer:
    """Run preprocessing, detection and valuation for one image."""

    def __init__(self, config: PipelineConfig, preset_name: str | None = None):
        self._cfg = config
        self._preset_name = preset_name or config.active_preset
        preset = config.get_preset(self._preset_name)

        self._preprocessing = ImagePreprocessing(
            blur_mode=config.blur_mode,
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

    def analyze_path(self, image_path: Path) -> AnalysisResult:
        """Load image from disk and run full pipeline."""
        image_bgr = read_bgr_or_raise(image_path)
        return self.analyze(image_bgr, image_path)

    def analyze(self, image_bgr: np.ndarray, source_path: Path) -> AnalysisResult:
        """Alias for `process_image` kept for compatibility."""
        return self.process_image(image_bgr, source_path)

    def process_path(self, image_path: Path) -> AnalysisResult:
        """Path-based processing entrypoint used by the runner."""
        image_bgr = read_bgr_or_raise(image_path)
        return self.process_image(image_bgr, image_path)

    def process_image(self, image_bgr: np.ndarray, source_path: Path) -> AnalysisResult:
        """Execute preprocessing -> detection -> value estimation pipeline."""
        # Normalize input geometry so downstream thresholds stay consistent.
        image_bgr = letterbox_resize_to_canvas(
            image_bgr,
            self._cfg.target_width,
            self._cfg.target_height,
        )

        prep = self._preprocessing.process(image_bgr)
        # Detection blur can be tuned independently from preprocessing output.
        detection_gray = cv2.cvtColor(prep.image_bgr, cv2.COLOR_BGR2GRAY)
        ksize = normalize_odd_ksize(self._cfg.gauss_ksize)
        if self._cfg.blur_mode == "gauss":
            detection_blurred = cv2.GaussianBlur(detection_gray, (ksize, ksize), self._cfg.gauss_sigma)
        else:
            detection_blurred = detection_gray if ksize <= 1 else cv2.medianBlur(detection_gray, ksize)

        # Stage A: circle localization (Hough + adaptive min-radius sweep).
        detection = self._detector.detect(detection_gray, detection_blurred, prep.image_rgb)
        # Stage B: per-coin color split + denomination inference.
        valuation = self._value_estimator.estimate(prep.image_bgr, detection.circles)

        value_counts = valuation.counts if len(valuation.counts) > 0 else {d: 0 for d in ValueEstimator.DENOM_PRINT_ORDER}
        # Ordered pipeline steps are consumed by the interactive viewer/exporter.
        steps = [
            PipelineStep("Original (Letterbox 640x480)", prep.image_rgb, "rgb"),
            PipelineStep("Grayscale", prep.gray, "gray"),
            PipelineStep(self._preprocessing.blur_step_name, prep.blurred, "gray"),
            PipelineStep("Hough Circles", detection.circles_overlay, "rgb"),
            PipelineStep("Inner/Border Mean Color Analysis", valuation.split_rgb, "rgb"),
            PipelineStep("Coin Value Estimation", valuation.value_labeled_rgb, "rgb"),
        ]
        # Debug payload is intentionally rich for viewer/export tooling.
        debug_info = {
            "preset": self._preset_name,
            "plateau_debug": detection.sweep_debug,
            "sweep_results": detection.sweep_results,
            "blur_mode": self._cfg.blur_mode,
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


CirclePipelineProcessor = Analyzer
