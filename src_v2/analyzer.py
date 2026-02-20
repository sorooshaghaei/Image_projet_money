from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import cv2
import numpy as np

from .config import ContourSettings, HoughSettings, PolicySettings, WatershedSettings
from .detectors import (
    build_binary_mask,
    count_hough_overlap_pairs,
    detect_contour_circles,
    detect_hough_circles,
    detect_watershed_circles,
    draw_overlay,
    merge_circle_sets,
    prepare_gray,
)
from .models import AnalysisResult, DebugFrames, SceneMetrics
from .policy import choose_auto_method, classify_background


class HybridCoinAnalyzer:
    """Scene-aware detector with policy routing across contour/hough/watershed methods."""

    def __init__(
        self,
        *,
        hough: Optional[HoughSettings] = None,
        contour: Optional[ContourSettings] = None,
        watershed: Optional[WatershedSettings] = None,
        policy: Optional[PolicySettings] = None,
    ):
        self.hough = hough or HoughSettings()
        self.contour = contour or ContourSettings()
        self.watershed = watershed or WatershedSettings()
        self.policy = policy or PolicySettings()

    def analyze(
        self,
        img_bgr: np.ndarray,
        *,
        source_path: str,
        short_path: str,
        mode: str = "auto",
        overrides: Optional[Dict[str, float]] = None,
    ) -> AnalysisResult:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("input image is empty")

        hough_cfg, contour_cfg, watershed_cfg, policy_cfg = self._apply_overrides(overrides or {})

        resized = self._resize_if_needed(img_bgr, int(policy_cfg.target_width))
        gray = prepare_gray(resized)
        edges = cv2.Canny(gray, threshold1=60, threshold2=150)
        binary = build_binary_mask(gray, contour_cfg)

        contour_circles, merge_score = detect_contour_circles(binary, contour_cfg, policy_cfg)
        hough_circles = detect_hough_circles(gray, hough_cfg)
        overlap_pairs = count_hough_overlap_pairs(hough_circles, policy_cfg.overlap_distance_scale)
        pair_total = (len(hough_circles) * max(0, len(hough_circles) - 1)) / 2.0
        overlap_ratio = float(overlap_pairs / pair_total) if pair_total > 0 else 0.0

        border_cv = self._border_cv(gray)
        edge_density = float(cv2.countNonZero(edges) / max(1, edges.size))
        background_label, texture_score = classify_background(border_cv, edge_density, policy_cfg)

        likely_overlap_from_hough = len(hough_circles) >= 8 and overlap_ratio >= 0.22
        likely_overlap = bool(likely_overlap_from_hough or merge_score >= float(policy_cfg.overlap_merge_score))

        requested_mode = _canonical_mode(mode)
        if requested_mode == "auto":
            selected_method = choose_auto_method(background_label, likely_overlap, len(hough_circles))
        else:
            selected_method = requested_mode

        watershed_markers = np.zeros_like(resized)
        circles = []

        if selected_method == "contours":
            circles = contour_circles
        elif selected_method == "hough":
            circles = hough_circles
        elif selected_method == "watershed":
            circles, watershed_markers = detect_watershed_circles(
                resized,
                binary,
                watershed_cfg,
                contour_cfg,
            )
        elif selected_method == "hough+watershed":
            ws_circles, watershed_markers = detect_watershed_circles(
                resized,
                binary,
                watershed_cfg,
                contour_cfg,
            )
            circles = merge_circle_sets(ws_circles, hough_circles)
        else:
            raise ValueError(f"unsupported mode: {selected_method}")

        if not circles and contour_circles:
            circles = contour_circles

        overlay = draw_overlay(resized, circles, selected_method)
        metrics = SceneMetrics(
            border_cv=border_cv,
            edge_density=edge_density,
            texture_score=texture_score,
            contour_merge_score=merge_score,
            hough_overlap_pairs=int(overlap_pairs),
            likely_overlap=likely_overlap,
            background_label=background_label,
        )

        return AnalysisResult(
            source_path=source_path,
            short_path=short_path,
            selected_method=selected_method,
            circles=circles,
            metrics=metrics,
            frames=DebugFrames(
                overlay=overlay,
                gray=gray,
                edges=edges,
                binary_mask=binary,
                watershed_markers=watershed_markers,
            ),
        )

    def _apply_overrides(
        self,
        overrides: Dict[str, float],
    ) -> tuple[HoughSettings, ContourSettings, WatershedSettings, PolicySettings]:
        hough_cfg = self.hough
        contour_cfg = self.contour
        watershed_cfg = self.watershed
        policy_cfg = self.policy

        hough_updates = {
            "dp": overrides.get("hough_dp"),
            "min_dist": overrides.get("hough_min_dist"),
            "param1": overrides.get("hough_param1"),
            "param2": overrides.get("hough_param2"),
            "min_radius": overrides.get("hough_min_radius"),
            "max_radius": overrides.get("hough_max_radius"),
        }
        contour_updates = {
            "min_circularity": overrides.get("contour_min_circularity"),
            "min_area": overrides.get("contour_min_area"),
        }
        watershed_updates = {
            "fg_ratio": overrides.get("watershed_fg_ratio"),
            "min_seed_area": overrides.get("watershed_min_seed_area"),
        }
        policy_updates = {
            "easy_threshold": overrides.get("policy_easy_threshold"),
            "medium_threshold": overrides.get("policy_medium_threshold"),
            "target_width": overrides.get("target_width"),
        }

        hough_kwargs = {k: v for k, v in hough_updates.items() if v is not None}
        contour_kwargs = {k: v for k, v in contour_updates.items() if v is not None}
        watershed_kwargs = {k: v for k, v in watershed_updates.items() if v is not None}
        policy_kwargs = {k: v for k, v in policy_updates.items() if v is not None}

        if hough_kwargs:
            hough_cfg = replace(
                hough_cfg,
                **{
                    "dp": float(hough_kwargs.get("dp", hough_cfg.dp)),
                    "min_dist": int(hough_kwargs.get("min_dist", hough_cfg.min_dist)),
                    "param1": int(hough_kwargs.get("param1", hough_cfg.param1)),
                    "param2": int(hough_kwargs.get("param2", hough_cfg.param2)),
                    "min_radius": int(hough_kwargs.get("min_radius", hough_cfg.min_radius)),
                    "max_radius": int(hough_kwargs.get("max_radius", hough_cfg.max_radius)),
                },
            )

        if contour_kwargs:
            contour_cfg = replace(
                contour_cfg,
                **{
                    "min_circularity": float(contour_kwargs.get("min_circularity", contour_cfg.min_circularity)),
                    "min_area": int(contour_kwargs.get("min_area", contour_cfg.min_area)),
                },
            )

        if watershed_kwargs:
            watershed_cfg = replace(
                watershed_cfg,
                **{
                    "fg_ratio": float(watershed_kwargs.get("fg_ratio", watershed_cfg.fg_ratio)),
                    "min_seed_area": int(watershed_kwargs.get("min_seed_area", watershed_cfg.min_seed_area)),
                },
            )

        if policy_kwargs:
            policy_cfg = replace(
                policy_cfg,
                **{
                    "easy_threshold": float(policy_kwargs.get("easy_threshold", policy_cfg.easy_threshold)),
                    "medium_threshold": float(policy_kwargs.get("medium_threshold", policy_cfg.medium_threshold)),
                    "target_width": int(policy_kwargs.get("target_width", policy_cfg.target_width)),
                },
            )

        return hough_cfg, contour_cfg, watershed_cfg, policy_cfg

    @staticmethod
    def _resize_if_needed(img_bgr: np.ndarray, target_width: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if w <= max(1, target_width):
            return img_bgr.copy()

        scale = float(target_width / max(1, w))
        return cv2.resize(img_bgr, (int(target_width), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _border_cv(gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        border = max(4, int(round(min(h, w) * 0.08)))

        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[:border, :] = 1
        border_mask[-border:, :] = 1
        border_mask[:, :border] = 1
        border_mask[:, -border:] = 1

        values = gray[border_mask > 0].astype(np.float32)
        if values.size <= 0:
            values = gray.astype(np.float32).reshape(-1)

        mean_val = float(np.mean(values))
        if mean_val <= 1e-6:
            return 0.0
        return float(np.std(values) / mean_val)


def _canonical_mode(mode: str) -> str:
    mode_text = str(mode or "auto").strip().lower()
    if mode_text in {"auto", "hough", "contours", "watershed", "hough+watershed"}:
        return mode_text
    if mode_text == "hybrid":
        return "hough+watershed"
    raise ValueError(f"invalid mode: {mode}")
