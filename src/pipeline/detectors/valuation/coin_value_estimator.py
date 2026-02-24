"""High-level coin-value stage composed of analyzer + estimator."""

from __future__ import annotations

import cv2
import numpy as np

from ..color_analysis.coin_analyzer import CoinAnalyzer
from ..color_analysis.coin_analyzer_config import CoinAnalyzerConfig
from .coin_value_estimation_output import CoinValueEstimationOutput
from .value_estimator import ValueEstimator


class CoinValueEstimator:
    """Bridge object: run color split analysis then denomination inference."""

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
        """Estimate per-coin denominations and aggregate total value."""
        split_rgb, split_stats = self._analyzer.analyze(image_bgr, circles)

        if len(split_stats) > 0:
            # Non-empty analysis rows: infer per-coin labels and draw overlay.
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

        # No circles detected: return a zero-valued but shape-compatible payload.
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
