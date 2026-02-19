from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.processor_color import CoinColorClassifier
from src.processor_scale import ScaleValueClassifier

from .coin_metadata import COLOR_UNKNOWN
from .config import DetectionConfig
from .models import PipelineResult, PipelineStep
from .processor_circles import CoinGeometryDetector


@dataclass
class _ClassificationResult:
    coin_labels: List[Optional[int]]
    coin_color_labels: List[str]
    coin_candidate_denoms: List[List[int]]
    radius_ratio_matrix: List[List[float]]
    ratio_fit_errors: List[Optional[float]]
    labeled_count: int
    estimated_value_eur: float


@dataclass(frozen=True)
class _SceneAssessment:
    warnings: Tuple[str, ...]
    reject: bool
    profile: str
    metrics: Dict[str, float]


class CoinProcessor:
    """Quality-gated classical OpenCV euro coin pipeline (v2, no deep learning)."""

    _COLOR_MISMATCH_PENALTY: float = 0.10

    def __init__(self, config: DetectionConfig):
        self._cfg = config
        self._geometry_detector = CoinGeometryDetector(config)
        self._color_classifier = CoinColorClassifier()
        self._scale_classifier = ScaleValueClassifier(
            color_mismatch_penalty=self._COLOR_MISMATCH_PENALTY,
        )

    def execute(self, img: np.ndarray, filename: str = "Unknown") -> Optional[PipelineResult]:
        """Default batch path with quality gate and adaptive strict/relaxed profile."""
        if img is None or img.size == 0:
            return None

        img_resized = self._resize(img)
        return self._run_pipeline(
            img_bgr_resized=img_resized,
            filename=filename,
            classify=True,
            apply_quality_gate=True,
            dp=self._cfg.HOUGH_DP,
            minDist=self._cfg.HOUGH_MIN_DIST,
            param1=self._cfg.HOUGH_PARAM1,
            param2=self._cfg.HOUGH_PARAM2,
            minRadius=self._cfg.HOUGH_MIN_RADIUS,
            maxRadius=self._cfg.HOUGH_MAX_RADIUS,
        )

    def detect_with_params(
        self,
        img_bgr_resized: np.ndarray,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
        filename: str = "LIVE_TUNE",
        classify: bool = True,
        config_overrides: Optional[Dict[str, float]] = None,
    ) -> PipelineResult:
        """
        UI path used by interactive browser.

        Quality issues are measured and surfaced as warnings, but not hard-rejected
        to keep tuning behavior inspectable.
        """
        if not config_overrides:
            return self._run_pipeline(
                img_bgr_resized=img_bgr_resized,
                filename=filename,
                classify=classify,
                apply_quality_gate=False,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius,
            )

        original_cfg = self._cfg
        original_detector = self._geometry_detector
        try:
            tuned_cfg = replace(self._cfg, **config_overrides)
            self._cfg = tuned_cfg
            self._geometry_detector = CoinGeometryDetector(tuned_cfg)
            return self._run_pipeline(
                img_bgr_resized=img_bgr_resized,
                filename=filename,
                classify=classify,
                apply_quality_gate=False,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius,
            )
        finally:
            self._cfg = original_cfg
            self._geometry_detector = original_detector

    def _run_pipeline(
        self,
        *,
        img_bgr_resized: np.ndarray,
        filename: str,
        classify: bool,
        apply_quality_gate: bool,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> PipelineResult:
        if img_bgr_resized is None or img_bgr_resized.size == 0:
            return PipelineResult(
                steps=[],
                coin_count=0,
                is_inverted=False,
                source_filename=filename,
                quality_warnings=["invalid input image"],
                quality_rejected=True,
                scene_profile="default",
                scene_metrics={},
            )

        scene = self._assess_scene(img_bgr_resized)
        if apply_quality_gate and scene.reject:
            return PipelineResult(
                steps=[PipelineStep("1. Original", img_bgr_resized.copy(), "rgb")],
                coin_count=0,
                is_inverted=False,
                source_filename=filename,
                quality_warnings=list(scene.warnings),
                quality_rejected=True,
                scene_profile=scene.profile,
                scene_metrics=scene.metrics,
            )

        active_profile = scene.profile if apply_quality_gate else "default"
        steps: List[PipelineStep] = [PipelineStep("1. Original", img_bgr_resized.copy(), "rgb")]

        norm_bgr, gray, blur = self._normalize_illumination(img_bgr_resized)
        hough_gray, hough_blur = self._prepare_hough_inputs(img_bgr_resized)
        steps.append(PipelineStep("2. CLAHE + Denoise", norm_bgr, "rgb"))

        dp, minDist, param1, param2, minRadius, maxRadius = self._select_params_by_profile(
            scene_profile=active_profile,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        det_out = self._geometry_detector.detect(
            img_bgr=norm_bgr,
            gray=hough_gray,
            blur=hough_blur,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
            scene_profile=active_profile,
            scene_metrics=scene.metrics,
        )

        steps.append(PipelineStep("3. BG Distance", det_out.distance_map, "gray"))
        steps.append(PipelineStep("4. Coin Mask", det_out.binary_mask, "gray"))
        steps.append(PipelineStep("5. Watershed", det_out.markers_viz, "rgb"))

        quality_warnings = list(scene.warnings)
        if det_out.edge_like_components > 0:
            quality_warnings.append(f"possible stacked/on-edge coins: {det_out.edge_like_components}")

        circles = list(det_out.circles)
        coin_count = len(circles)
        coin_tags = [self._coin_tag(i) for i in range(coin_count)]
        coin_radii = [float(c[2]) for c in circles]

        if classify:
            class_result = self._classify_circles(norm_bgr, circles, coin_radii)
        else:
            class_result = self._build_unclassified_result(coin_count)

        display_img = norm_bgr.copy()
        mask = np.zeros_like(gray)
        if circles:
            self._draw_coin_annotations(
                display_img=display_img,
                mask=mask,
                circles=circles,
                coin_tags=coin_tags,
                coin_labels=class_result.coin_labels,
            )
        steps.append(PipelineStep("6. Detected + Labels", display_img, "rgb"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=False,
            source_filename=filename,
            estimated_value_eur=class_result.estimated_value_eur,
            labeled_coin_count=class_result.labeled_count,
            coin_labels=class_result.coin_labels,
            coin_color_labels=class_result.coin_color_labels,
            radius_ratio_matrix=class_result.radius_ratio_matrix,
            ratio_fit_errors=class_result.ratio_fit_errors,
            coin_tags=coin_tags,
            coin_radii=coin_radii,
            coin_candidate_denoms=class_result.coin_candidate_denoms,
            quality_warnings=quality_warnings,
            quality_rejected=False,
            scene_profile=active_profile,
            scene_metrics=scene.metrics,
        )

    def classify_coin_color_groups(self, features: Sequence[Dict[str, float]]) -> Tuple[List[str], List[Dict[str, float]]]:
        return self._color_classifier.classify_color_groups(features)

    def classify_coin_values_by_scale_mm(
        self,
        radii_px: Sequence[float],
        candidate_denoms_per_coin: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[float], float]:
        return self._scale_classifier.classify(radii_px, candidate_denoms_per_coin)

    def _assess_scene(self, img_bgr_resized: np.ndarray) -> _SceneAssessment:
        gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        border = max(4, int(round(min(h, w) * float(self._cfg.SCENE_BORDER_FRACTION))))
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[:border, :] = 1
        border_mask[-border:, :] = 1
        border_mask[:, :border] = 1
        border_mask[:, -border:] = 1

        border_pixels = gray[border_mask > 0].astype(np.float32)
        if border_pixels.size == 0:
            border_pixels = gray.astype(np.float32).reshape(-1)

        border_std = float(np.std(border_pixels))

        edge_map = cv2.Canny(gray, threshold1=65, threshold2=145)
        border_edge_ratio = float(
            np.count_nonzero(np.logical_and(edge_map > 0, border_mask > 0)) / max(1, np.count_nonzero(border_mask > 0))
        )
        edge_density = float(cv2.countNonZero(edge_map) / edge_map.size) if edge_map.size > 0 else 0.0

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        dark_thr = int(np.clip(self._cfg.EXPOSURE_DARK_THRESHOLD, 0, 255))
        bright_thr = int(np.clip(self._cfg.EXPOSURE_BRIGHT_THRESHOLD, 0, 255))
        dark_ratio = float(np.mean(gray <= dark_thr))
        bright_ratio = float(np.mean(gray >= bright_thr))

        hsv = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2HSV)
        specular = np.logical_and(hsv[:, :, 2] >= bright_thr, hsv[:, :, 1] <= int(self._cfg.SPECULAR_LOW_SAT_THRESHOLD))
        specular_ratio = float(np.mean(specular))

        bad_background = (
            border_std > float(self._cfg.MAX_BORDER_STD)
            or border_edge_ratio > float(self._cfg.MAX_BORDER_EDGE_DENSITY)
        )
        blurry = lap_var < float(self._cfg.MIN_LAPLACIAN_VAR)
        bad_exposure = dark_ratio > float(self._cfg.MAX_DARK_RATIO) or bright_ratio > float(self._cfg.MAX_BRIGHT_RATIO)
        severe_glare = specular_ratio > float(self._cfg.MAX_SPECULAR_RATIO)

        warnings: List[str] = []
        if bad_background:
            warnings.append(
                "non-homogeneous background "
                f"(border_std={border_std:.1f}, border_edge={border_edge_ratio:.3f})"
            )
        if blurry:
            warnings.append(f"blurry image (lap_var={lap_var:.1f})")
        if bad_exposure:
            warnings.append(f"exposure issue (dark={dark_ratio:.3f}, bright={bright_ratio:.3f})")
        if severe_glare:
            warnings.append(f"specular glare (ratio={specular_ratio:.3f})")

        profile = "default"
        if bool(self._cfg.ADAPTIVE_PROFILE_ENABLED):
            if bad_background or blurry or bad_exposure or severe_glare or edge_density >= float(self._cfg.ADAPTIVE_TEXTURE_EDGE_DENSITY):
                profile = "strict"
            else:
                profile = "relaxed"

        mode = self._quality_mode()
        if mode == "off":
            warnings = []

        return _SceneAssessment(
            warnings=tuple(warnings),
            reject=mode == "reject" and bool(warnings),
            profile=profile,
            metrics={
                "border_std": border_std,
                "border_edge_ratio": border_edge_ratio,
                "laplacian_var": lap_var,
                "dark_ratio": dark_ratio,
                "bright_ratio": bright_ratio,
                "specular_ratio": specular_ratio,
                "edge_density": edge_density,
            },
        )

    def _quality_mode(self) -> str:
        mode = str(self._cfg.SCENE_QUALITY_MODE).strip().lower()
        if mode in ("off", "warn", "reject"):
            return mode
        return "warn"

    def _normalize_illumination(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        tile = max(2, int(self._cfg.CLAHE_TILE_GRID))
        clahe = cv2.createCLAHE(clipLimit=float(self._cfg.CLAHE_CLIP_LIMIT), tileGridSize=(tile, tile))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        norm_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        if str(self._cfg.DENOISE_METHOD).strip().lower() == "median":
            k = int(max(3, self._cfg.MEDIAN_BLUR_KERNEL_SIZE))
            if k % 2 == 0:
                k += 1
            denoised = cv2.medianBlur(norm_bgr, k)
        else:
            denoised = cv2.bilateralFilter(
                norm_bgr,
                d=max(3, int(self._cfg.BILATERAL_DIAMETER)),
                sigmaColor=max(1, int(self._cfg.BILATERAL_SIGMA_COLOR)),
                sigmaSpace=max(1, int(self._cfg.BILATERAL_SIGMA_SPACE)),
            )

        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        k = int(max(3, self._cfg.MEDIAN_BLUR_KERNEL_SIZE))
        if k % 2 == 0:
            k += 1
        blur = cv2.medianBlur(gray, k)
        return denoised, gray, blur

    def _prepare_hough_inputs(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build a Hough-friendly grayscale path close to the baseline detector."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        if float(np.mean(gray)) < 110.0:
            gray = cv2.bitwise_not(gray)

        k = int(max(3, self._cfg.HOUGH_PREP_BLUR_KERNEL_SIZE))
        if k % 2 == 0:
            k += 1
        blur = cv2.medianBlur(gray, k)
        return gray, blur

    def _select_params_by_profile(
        self,
        *,
        scene_profile: str,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> Tuple[float, int, int, int, int, int]:
        safe_dp, safe_min_dist, safe_param1, safe_param2, safe_min_radius, safe_max_radius = self._sanitize_hough_params(
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        if scene_profile == "strict":
            safe_min_dist = int(round(safe_min_dist * float(self._cfg.ADAPTIVE_STRICT_MIN_DIST_SCALE)))
            safe_param2 = int(round(safe_param2 * float(self._cfg.ADAPTIVE_STRICT_PARAM2_SCALE)))
        elif scene_profile == "relaxed":
            safe_min_dist = int(round(safe_min_dist * float(self._cfg.ADAPTIVE_RELAXED_MIN_DIST_SCALE)))
            safe_param2 = int(round(safe_param2 * float(self._cfg.ADAPTIVE_RELAXED_PARAM2_SCALE)))

        return self._sanitize_hough_params(
            dp=safe_dp,
            minDist=safe_min_dist,
            param1=safe_param1,
            param2=safe_param2,
            minRadius=safe_min_radius,
            maxRadius=safe_max_radius,
        )

    def _sanitize_hough_params(
        self,
        *,
        dp: float,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int,
    ) -> Tuple[float, int, int, int, int, int]:
        safe_min_radius = int(max(0, minRadius))
        safe_max_radius = int(max(safe_min_radius + 1, maxRadius))
        safe_min_dist = int(max(1, minDist))
        safe_param1 = int(max(1, param1))
        safe_param2 = int(max(1, param2))
        safe_dp = float(max(0.1, dp))
        return safe_dp, safe_min_dist, safe_param1, safe_param2, safe_min_radius, safe_max_radius

    def _classify_circles(
        self,
        img_bgr_resized: np.ndarray,
        circles: Sequence[Tuple[int, int, int]],
        coin_radii: Sequence[float],
    ) -> _ClassificationResult:
        color_features = [self._color_classifier.extract_coin_features(img_bgr=img_bgr_resized, circle=np.asarray(c)) for c in circles]
        coin_color_labels, color_group_scores = self.classify_coin_color_groups(color_features)
        coin_candidate_denoms = [self._color_classifier.candidate_denoms_from_group_scores(scores) for scores in color_group_scores]

        ratio_matrix = self._build_radius_ratio_matrix(coin_radii)
        raw_labels, ratio_fit_errors, _ = self.classify_coin_values_by_scale_mm(
            radii_px=coin_radii,
            candidate_denoms_per_coin=coin_candidate_denoms,
        )

        coin_labels: List[Optional[int]] = []
        for label, err in zip(raw_labels, ratio_fit_errors):
            if err is None or float(err) > float(self._cfg.MAX_LABEL_REL_ERROR):
                coin_labels.append(None)
            else:
                coin_labels.append(int(label))

        est_value = float(sum(int(label) for label in coin_labels if label is not None)) / 100.0
        labeled_count = sum(1 for label in coin_labels if label is not None)

        return _ClassificationResult(
            coin_labels=coin_labels,
            coin_color_labels=coin_color_labels,
            coin_candidate_denoms=coin_candidate_denoms,
            radius_ratio_matrix=ratio_matrix,
            ratio_fit_errors=[float(err) for err in ratio_fit_errors],
            labeled_count=labeled_count,
            estimated_value_eur=est_value,
        )

    def _build_unclassified_result(self, coin_count: int) -> _ClassificationResult:
        return _ClassificationResult(
            coin_labels=[None] * coin_count,
            coin_color_labels=[COLOR_UNKNOWN] * coin_count,
            coin_candidate_denoms=[[] for _ in range(coin_count)],
            radius_ratio_matrix=[],
            ratio_fit_errors=[None] * coin_count,
            labeled_count=0,
            estimated_value_eur=0.0,
        )

    def _draw_coin_annotations(
        self,
        display_img: np.ndarray,
        mask: np.ndarray,
        circles: Sequence[Tuple[int, int, int]],
        coin_tags: Sequence[str],
        coin_labels: Sequence[Optional[int]],
    ) -> None:
        h_img, w_img = display_img.shape[:2]
        for idx, (x, y, r) in enumerate(circles):
            x_i, y_i, r_i = int(x), int(y), int(r)
            cv2.circle(display_img, (x_i, y_i), r_i, (0, 0, 0), 5)
            cv2.circle(display_img, (x_i, y_i), r_i, (0, 255, 0), 2)
            cv2.circle(display_img, (x_i, y_i), 3, (255, 255, 255), -1)
            cv2.circle(display_img, (x_i, y_i), 2, (0, 0, 255), -1)
            cv2.circle(mask, (x_i, y_i), r_i, 255, -1)

            tag = coin_tags[idx]
            label = coin_labels[idx] if idx < len(coin_labels) else None
            text = f"{tag}" if label is None else f"{tag}:{label}c"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.55, min(0.9, r_i / 45.0))
            thickness = max(1, int(round(font_scale * 2.0)))
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            tx = x_i - (tw // 2)
            ty = y_i - r_i - 8
            if ty - th - baseline < 2:
                ty = y_i + r_i + th + 8

            tx = int(np.clip(tx, 2, max(2, w_img - tw - 2)))
            ty = int(np.clip(ty, th + baseline + 2, h_img - 2))

            pad = 3
            top_left = (max(0, tx - pad), max(0, ty - th - baseline - pad))
            bottom_right = (min(w_img - 1, tx + tw + pad), min(h_img - 1, ty + pad))
            cv2.rectangle(display_img, top_left, bottom_right, (0, 0, 0), -1)
            cv2.putText(display_img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(display_img, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _build_radius_ratio_matrix(self, radii: Sequence[float]) -> List[List[float]]:
        n = len(radii)
        if n == 0:
            return []

        mat: List[List[float]] = []
        for i in range(n):
            row: List[float] = []
            for j in range(n):
                den = float(radii[j])
                row.append(round(float(radii[i] / den) if den > 1e-6 else 0.0, 4))
            mat.append(row)
        return mat

    def _coin_tag(self, idx: int) -> str:
        n = idx + 1
        chars: List[str] = []
        while n > 0:
            n -= 1
            chars.append(chr(ord("A") + (n % 26)))
            n //= 26
        return "".join(reversed(chars))

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w == 0:
            return img
        scale = self._cfg.TARGET_WIDTH / w
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(h * scale)))
