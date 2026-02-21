from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.coin_metadata import COIN_DIAMETER_MM
from src.processor_color import CoinColorClassifier
from src.processor_scale import ScaleValueClassifier

from .config import ContourSettings, HoughSettings, PolicySettings, WatershedSettings
from .detectors import (
    count_hough_overlap_pairs,
    detect_contour_circles,
    detect_hough_circles,
    detect_watershed_circles,
    draw_overlay,
    merge_circle_sets,
)
from .models import AnalysisResult, Circle, DebugFrames, SceneMetrics
from .policy import choose_auto_method, classify_background
from .preprocessing import build_binary_mask, build_edge_map, prepare_gray, resize_if_needed


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
        self._classifier_signature: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._color_classifier = None
        self._scale_classifier = None
        self._refresh_classifiers_if_needed()

    def analyze(
        self,
        img_bgr: np.ndarray,
        *,
        source_path: str,
        short_path: str,
        mode: str = "auto",
        overrides: Optional[Dict[str, float]] = None,
    ) -> AnalysisResult:
        """Run one full analysis pass and return detections + debug artifacts.

        Why this exists:
        - Keeps all detector decisions in one place so CLI + visualizer stay consistent.
        - Returns intermediate frames to make failures inspectable (gray, mask, watershed map).
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("input image is empty")

        self._refresh_classifiers_if_needed()
        hough_cfg, contour_cfg, watershed_cfg, policy_cfg = self._apply_overrides(overrides or {})
        requested_mode = _canonical_mode(mode)
        if requested_mode == "fast":
            # Fast mode trades a little quality for speed by shrinking large inputs.
            policy_cfg = replace(policy_cfg, target_width=min(int(policy_cfg.target_width), 820))

        # Shared preprocessing used by all detector families.
        resized = resize_if_needed(img_bgr, int(policy_cfg.target_width))
        gray = prepare_gray(resized)
        edges = build_edge_map(gray, threshold1=60, threshold2=150)
        binary = build_binary_mask(gray, contour_cfg)

        # Border texture + edge density are scene cues for easy/medium/difficult routing.
        border_cv = self._border_cv(gray)
        edge_density = float(cv2.countNonZero(edges) / max(1, edges.size))
        background_label, texture_score = classify_background(border_cv, edge_density, policy_cfg)

        watershed_markers = np.zeros_like(resized)
        circles: List[Circle] = []
        contour_circles: List[Circle] = []
        hough_circles: List[Circle] = []
        merge_score = 0.0
        overlap_pairs = 0
        likely_overlap = False

        if requested_mode == "fast":
            # In fast mode we intentionally skip watershed because it is the slowest stage.
            selected_method = self._choose_fast_method(background_label=background_label)
            if selected_method == "contours":
                contour_circles, merge_score = detect_contour_circles(binary, contour_cfg, policy_cfg)
                circles = contour_circles
            else:
                circles = detect_hough_circles(gray, hough_cfg)
        else:
            # Full mode computes both lightweight proposals first; policy selects best path.
            contour_circles, merge_score = detect_contour_circles(binary, contour_cfg, policy_cfg)
            hough_circles = detect_hough_circles(gray, hough_cfg)
            overlap_pairs = count_hough_overlap_pairs(hough_circles, policy_cfg.overlap_distance_scale)
            pair_total = (len(hough_circles) * max(0, len(hough_circles) - 1)) / 2.0
            overlap_ratio = float(overlap_pairs / pair_total) if pair_total > 0 else 0.0
            # Heuristic: many close circle pairs usually means touching/stacked coins.
            likely_overlap_from_hough = len(hough_circles) >= 8 and overlap_ratio >= 0.22
            likely_overlap = bool(likely_overlap_from_hough or merge_score >= float(policy_cfg.overlap_merge_score))

            if requested_mode == "auto":
                selected_method = choose_auto_method(background_label, likely_overlap, len(hough_circles))
                selected_method = self._refine_auto_method(
                    selected_method=selected_method,
                    background_label=background_label,
                    likely_overlap=likely_overlap,
                    contour_count=len(contour_circles),
                    hough_count=len(hough_circles),
                    merge_score=merge_score,
                )
            else:
                selected_method = requested_mode

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
                # Hybrid path keeps watershed splits and also recovers circles watershed may miss.
                circles = merge_circle_sets(ws_circles, hough_circles)
            else:
                raise ValueError(f"unsupported mode: {selected_method}")

            if not circles and contour_circles:
                # Last-resort fallback so a failed advanced path still returns usable detections.
                circles = contour_circles

        # Remove implausible radii (absolute euro-size bounds + robust median-relative gate).
        circles = self._filter_circles_by_radius(circles, policy_cfg)

        coin_labels, coin_color_labels, ratio_fit_errors, estimated_value_eur, labeled_coin_count = self._classify_coin_values(
            img_bgr=resized,
            circles=circles,
            background_label=background_label,
            policy_cfg=policy_cfg,
            max_rel_error=float(policy_cfg.max_label_rel_error),
        )
        overlay = draw_overlay(resized, circles, selected_method, coin_labels)
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
            estimated_value_eur=estimated_value_eur,
            labeled_coin_count=labeled_coin_count,
            coin_labels=coin_labels,
            coin_color_labels=coin_color_labels,
            ratio_fit_errors=ratio_fit_errors,
        )

    def _classify_coin_values(
        self,
        *,
        img_bgr: np.ndarray,
        circles: List[Circle],
        background_label: str,
        policy_cfg: PolicySettings,
        max_rel_error: float,
    ) -> Tuple[List[Optional[int]], List[str], List[Optional[float]], float, int]:
        """Map each circle to denomination with color + size constraints.

        Steps:
        1) classify color family per coin (copper / gold / bimetal / unknown)
        2) fit pixel radii to denomination scale model
        3) reject low-confidence fits using max relative error threshold
        4) optionally calibrate total value to reduce systematic bias
        """
        if not circles:
            return [], [], [], 0.0, 0

        if self._color_classifier is None or self._scale_classifier is None:
            self._refresh_classifiers_if_needed()

        features = [
            self._color_classifier.extract_coin_features(
                img_bgr,
                np.asarray([circle.x, circle.y, circle.r], dtype=np.int32),
            )
            for circle in circles
        ]
        coin_color_labels, group_scores = self._color_classifier.classify_color_groups(features)
        candidate_denoms_per_coin = [
            self._color_classifier.candidate_denoms_from_group_scores(scores) for scores in group_scores
        ]

        radii_px = [float(circle.r) for circle in circles]
        labels, rel_errors, _best_scale = self._scale_classifier.classify(
            radii_px=radii_px,
            candidate_denoms_per_coin=candidate_denoms_per_coin,
        )

        final_labels: List[Optional[int]] = []
        final_rel_errors: List[Optional[float]] = []
        safe_threshold = float(max(1e-6, max_rel_error))
        if background_label == "medium":
            safe_threshold *= 1.05
        elif background_label == "difficult":
            safe_threshold *= 1.12

        for circle, label, rel_err in zip(circles, labels, rel_errors):
            rel = float(rel_err)
            final_rel_errors.append(rel)
            radius_ok = self._is_radius_label_plausible(
                radius_px=float(circle.r),
                denom=int(label),
                best_scale=float(_best_scale),
                policy_cfg=policy_cfg,
            )
            if rel <= safe_threshold and radius_ok:
                # Keep denomination only when radius-scale fit is stable enough.
                final_labels.append(int(label))
            else:
                final_labels.append(None)

        raw_value_eur = float(sum(label for label in final_labels if label is not None) / 100.0)
        labeled_coin_count = int(sum(1 for label in final_labels if label is not None))
        estimated_value_eur = raw_value_eur
        if bool(self.policy.value_calibration_enabled) and len(circles) > 0:
            # Lightweight linear correction learned from dataset behavior.
            estimated_value_eur = (
                float(self.policy.value_calibration_alpha) * raw_value_eur
                + float(self.policy.value_calibration_count_beta) * float(len(circles))
                + float(self.policy.value_calibration_bias)
            )
            estimated_value_eur = max(0.0, float(estimated_value_eur))
        return final_labels, coin_color_labels, final_rel_errors, estimated_value_eur, labeled_coin_count

    def _refresh_classifiers_if_needed(self) -> None:
        """Recreate color/scale classifiers when policy-driven thresholds change."""
        signature = (
            float(self.policy.bimetal_high_confidence),
            float(self.policy.bimetal_strong_2e_margin),
            float(self.policy.color_unknown_label_threshold),
            float(self.policy.color_mismatch_penalty),
        )
        if signature == self._classifier_signature and self._color_classifier is not None and self._scale_classifier is not None:
            return

        self._color_classifier = CoinColorClassifier(
            bimetal_high_confidence=float(self.policy.bimetal_high_confidence),
            bimetal_strong_2e_margin=float(self.policy.bimetal_strong_2e_margin),
            unknown_label_threshold=float(self.policy.color_unknown_label_threshold),
        )
        self._scale_classifier = ScaleValueClassifier(
            color_mismatch_penalty=float(self.policy.color_mismatch_penalty),
        )
        self._classifier_signature = signature

    def _is_radius_label_plausible(
        self,
        *,
        radius_px: float,
        denom: int,
        best_scale: float,
        policy_cfg: PolicySettings,
    ) -> bool:
        """Euro-size sanity gate for denomination assignment."""
        mm = COIN_DIAMETER_MM.get(int(denom))
        if mm is None or best_scale <= 1e-6:
            return False

        measured_diameter_mm = float((2.0 * max(0.0, radius_px)) / max(best_scale, 1e-6))
        min_mm = float(policy_cfg.coin_mm_min) * (1.0 - float(policy_cfg.coin_mm_rel_tol))
        max_mm = float(policy_cfg.coin_mm_max) * (1.0 + float(policy_cfg.coin_mm_rel_tol))
        if measured_diameter_mm < min_mm or measured_diameter_mm > max_mm:
            return False

        rel_mm_err = abs(measured_diameter_mm - float(mm)) / max(float(mm), 1e-6)
        return rel_mm_err <= float(policy_cfg.denom_radius_rel_tol)

    def _refine_auto_method(
        self,
        *,
        selected_method: str,
        background_label: str,
        likely_overlap: bool,
        contour_count: int,
        hough_count: int,
        merge_score: float,
    ) -> str:
        """Post-process the policy decision with guardrails.

        The policy can pick a method that is theoretically right but unsupported by current
        evidence (e.g., zero circles from that detector). This method converts those cases
        into safer fallbacks.
        """
        method = str(selected_method)
        contour_n = int(max(0, contour_count))
        hough_n = int(max(0, hough_count))

        if method == "contours":
            if contour_n <= 0 and hough_n > 0:
                return "hough"
            if hough_n <= 0:
                return "contours"

            # On clean backgrounds, keep contours only when both detectors agree.
            if background_label == "easy" and not likely_overlap:
                tolerance = max(1, int(round(0.35 * hough_n)))
                if abs(contour_n - hough_n) > tolerance:
                    return "hough"

        if method == "hough":
            # Keep hough as default for accuracy; only fallback on catastrophic over-count.
            if not likely_overlap and contour_n > 0:
                if hough_n >= contour_n + 8 and hough_n >= int(round(2.40 * contour_n)):
                    return "contours"
            if likely_overlap and hough_n >= 5 and merge_score >= 0.08:
                return "hough+watershed"

        if method == "hough" and hough_n <= 0 and contour_n > 0:
            return "contours"

        if method == "hough+watershed" and hough_n <= 0 and contour_n > 0:
            return "contours"

        return method

    @staticmethod
    def _choose_fast_method(*, background_label: str) -> str:
        # Fast mode skips watershed and keeps a single robust detector path.
        return "contours" if background_label == "easy" else "hough"

    def _filter_circles_by_radius(self, circles: List[Circle], policy_cfg: PolicySettings) -> List[Circle]:
        """Filter detections by radius plausibility.

        We apply two gates:
        - absolute pixel limits (reject obviously tiny/huge non-coin blobs)
        - median-relative limits (reject outliers in the current scene scale)
        """
        if not circles:
            return circles

        min_r = max(1, int(policy_cfg.min_coin_radius_px))
        max_r = max(min_r + 1, int(policy_cfg.max_coin_radius_px))
        abs_filtered = [c for c in circles if min_r <= int(c.r) <= max_r]
        if not abs_filtered:
            # If absolute thresholds are too strict for a scene, keep original result.
            return circles

        radii = np.asarray([float(c.r) for c in abs_filtered], dtype=np.float32)
        median_r = float(np.median(radii)) if radii.size > 0 else 0.0
        if median_r <= 1e-6:
            return abs_filtered

        lo = max(float(min_r), float(policy_cfg.median_radius_low_scale) * median_r)
        hi = min(float(max_r), float(policy_cfg.median_radius_high_scale) * median_r)
        med_filtered = [c for c in abs_filtered if lo <= float(c.r) <= hi]
        return med_filtered if med_filtered else abs_filtered

    def _apply_overrides(
        self,
        overrides: Dict[str, float],
    ) -> tuple[HoughSettings, ContourSettings, WatershedSettings, PolicySettings]:
        """Apply visualizer slider overrides without mutating base config objects."""
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
    def _border_cv(gray: np.ndarray) -> float:
        """Coefficient of variation on border pixels.

        Border texture is a cheap proxy for background complexity:
        high CV often means textured cloth/wood; low CV is usually clean backdrop.
        """
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
    if mode_text in {"auto", "fast", "hough", "contours", "watershed", "hough+watershed"}:
        return mode_text
    if mode_text == "hybrid":
        return "hough+watershed"
    raise ValueError(f"invalid mode: {mode}")
