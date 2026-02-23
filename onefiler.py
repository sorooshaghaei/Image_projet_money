#!/usr/bin/env python3
"""Single-file runner for notebook-style pipeline with project dataset evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import cv2

import matplotlib


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


def _configure_matplotlib_backend() -> str:
    current_backend = str(matplotlib.get_backend()).strip().lower()
    if current_backend and current_backend not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
        if not current_backend.startswith("module://matplotlib_inline"):
            return str(matplotlib.get_backend()).strip()

    candidates = [
        ("TkAgg", _module_available("tkinter")),
        (
            "QtAgg",
            any(
                _module_available(module_name)
                for module_name in ("PyQt6", "PySide6", "PyQt5", "PySide2")
            ),
        ),
    ]
    for backend_name, available in candidates:
        if not available:
            continue
        try:
            matplotlib.use(backend_name, force=True)
            return backend_name
        except Exception:
            continue

    try:
        matplotlib.use("WebAgg", force=True)
        return "WebAgg"
    except Exception:
        pass

    matplotlib.use("Agg", force=True)
    return "Agg"


_MPL_BACKEND = _configure_matplotlib_backend()
import matplotlib.pyplot as plt

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text


# -----------------------------------------------------------------------------
# Inlined from src/config.py
# -----------------------------------------------------------------------------



@dataclass(frozen=True)
class HoughPreset:
    dp: float
    min_dist: int
    param2: int


HOUGH_PRESETS: dict[str, HoughPreset] = {
    "test1": HoughPreset(dp=1.15, min_dist=30, param2=55),
    "test2": HoughPreset(dp=1.15, min_dist=20, param2=57),
    "test3": HoughPreset(dp=1.15, min_dist=20, param2=50),
}


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "images"


@dataclass(frozen=True)
class PipelineConfig:
    dataset_dir: Path = field(default_factory=default_dataset_dir)
    valid_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    )

    target_width: int = 640
    target_height: int = 480

    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    color_normalization_enabled: bool = False
    color_balance_strength: float = 0.35
    color_balance_max_shift_ab: float = 6.0
    color_balance_max_shift_l: float = 4.0
    color_saturation_target: float = 108.0
    color_saturation_strength: float = 0.14

    histogram_normalization_enabled: bool = False
    histogram_clip_limit: float = 2.2
    histogram_tile_grid_size: tuple[int, int] = (8, 8)
    histogram_stretch_percentiles: tuple[float, float] = (1.0, 99.0)

    # Euro color anchors in OpenCV LAB scale (L:[0..255], a/b:[0..255]).
    # Source: U.S. Mint "Alternative Metals Study Final Report" (2012),
    # Tables 2-15, 2-16 and 2-18 (CIE Lab measurements).
    # Reference basis:
    # - "copper-plated" coin material: CIE Lab ~ (78.3, 13.6, 17.1)
    # - "manganese brass" coin material: CIE Lab ~ (82.3, 2.9, 14.6)
    # - "cupronickel" coin material: CIE Lab ~ (76.3, 0.8, 6.7)
    # Converted to OpenCV LAB:
    # - L_cv = L* * 255/100, a_cv = a* + 128, b_cv = b* + 128
    euro_reference_lab_bronze: tuple[float, float, float] = (200.0, 142.0, 145.0)
    euro_reference_lab_gold: tuple[float, float, float] = (210.0, 131.0, 143.0)
    euro_reference_lab_nickel: tuple[float, float, float] = (195.0, 129.0, 135.0)

    blur_mode: str = "gauss"  # or median
    gauss_ksize: int = 5
    gauss_sigma: float = 2.0

    active_preset: str = "test1"

    auto_param1_blur_ksize: int = gauss_ksize
    auto_param1_percentile: float = 65.0
    auto_param1_scale: float = 1.0
    auto_param1_clamp: tuple[int, int] = (30, 220)

    max_radius: int = 100
    min_radius_sweep_start: int = 10
    min_radius_sweep_end: int = max_radius - min_radius_sweep_start
    min_radius_sweep_step: int = 2

    circle_outline_color: tuple[int, int, int] = (0, 255, 0)
    circle_outline_thickness: int = 2
    center_color: tuple[int, int, int] = (255, 0, 0)
    center_radius: int = 2
    center_thickness: int = 3

    analysis_border_ratio: float = 0.24
    analysis_sat_delta_threshold: float | None = None
    analysis_bimetal_mode: str = "hybrid"  # hybrid or mean-color
    analysis_material_mode: str = "hybrid"  # hybrid or lab or hsv
    viewer_final_only: bool = True

    def get_preset(self, preset_name: str | None = None) -> HoughPreset:
        key = preset_name or self.active_preset
        if key not in HOUGH_PRESETS:
            available = ", ".join(sorted(HOUGH_PRESETS))
            raise ValueError(f"Unknown preset '{key}'. Available presets: {available}")
        return HOUGH_PRESETS[key]


# -----------------------------------------------------------------------------
# Inlined from src/dataset.py
# -----------------------------------------------------------------------------



@dataclass(frozen=True)
class DatasetImage:
    path: Path
    relative_path: Path


class ImageDataset:
    def __init__(self, root_dir: Path, valid_extensions: Sequence[str]):
        self._root_dir = Path(root_dir)
        self._valid_extensions = {ext.lower() for ext in valid_extensions}

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def list_images(self, limit: int | None = None) -> list[DatasetImage]:
        if not self._root_dir.exists():
            return []

        images: list[DatasetImage] = []
        for path in self._root_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._valid_extensions:
                continue
            relative = path.relative_to(self._root_dir)
            images.append(DatasetImage(path=path, relative_path=relative))

        images.sort(key=lambda item: str(item.relative_path).lower())
        if limit is not None:
            return images[: max(0, limit)]
        return images



# -----------------------------------------------------------------------------
# Inlined from src/ground_truth.py
# -----------------------------------------------------------------------------



def normalize_group_name(group: str) -> str:
    cleaned = (group or "").strip().lower()
    if cleaned.startswith("grp"):
        return "gp" + cleaned[3:]
    return cleaned


@dataclass(frozen=True)
class GroundTruthEntry:
    filename: str
    group: str
    coin_count: int
    value_cents: int | None = None


class GroundTruthRepository:
    def __init__(self, rows: Iterable[GroundTruthEntry] | None = None):
        entries = list(rows) if rows is not None else self._parse_default_rows()
        self._index: Dict[tuple[str, str], GroundTruthEntry] = {}
        for entry in entries:
            key = (normalize_group_name(entry.group), entry.filename.lower())
            self._index[key] = GroundTruthEntry(
                filename=entry.filename,
                group=normalize_group_name(entry.group),
                coin_count=int(entry.coin_count),
                value_cents=None if entry.value_cents is None else int(entry.value_cents),
            )

    def find(self, filename: str, group: str) -> GroundTruthEntry | None:
        key = (normalize_group_name(group), filename.lower())
        return self._index.get(key)

    def _parse_default_rows(self) -> list[GroundTruthEntry]:
        entries: list[GroundTruthEntry] = []
        for raw_line in RAW_ANNOTATIONS.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            filename = parts[0]
            pieces_text = parts[1]
            group = parts[-1]
            value_token = parts[2] if len(parts) >= 4 else None
            if not pieces_text.isdigit():
                continue
            entries.append(
                GroundTruthEntry(
                    filename=filename,
                    group=group,
                    coin_count=int(pieces_text),
                    value_cents=_parse_value_cents(value_token),
                )
            )
        return entries


def _parse_value_cents(raw: str | None) -> int | None:
    if raw is None:
        return None
    cleaned = raw.strip().lower().replace(",", ".")
    if not cleaned or cleaned in {"nan", "na", "n/a", "-"}:
        return None
    try:
        value_eur = float(cleaned)
    except ValueError:
        return None
    if not (value_eur >= 0.0):
        return None
    return int(round(value_eur * 100.0))


RAW_ANNOTATIONS = """
exemple1.png 4 7.25 gp1
10.jpg 9 3.13 gp5
11.jpg 12 6,18 gp5
12.jpg 16 8,83 gp5
13.jpg 19 12,33 gp5
14.jpg 28 15.69 gp5
15.jpg 35 17.32 gp5
16.jpg 48 18.69 gp5
17.jpg 48 18.20 gp5
0.jpeg 2 2.2 gp5
1.jpeg 4 4.22 gp5
2.jpeg 3 3.2 gp5
3.jpeg 4 0.8 gp5
4.jpeg 3 3 gp5
5.jpeg 2 1.20 gp5
6.jpeg 11 10.26 gp5
7.jpeg 3 1.7 gp5
8.jpg 6 NAN gp5
9.jpg 8 3.88 gp5
18.png 7 4.31 gp1
19.png 4 1.60 gp1
20.png 8 4.81 gp1
21.png 6 3.76 gp1
22.png 5 2.25 gp1
23.png 8 4.34 gp1
24.png 3 2.55 gp1
25.png 10 4.40 gp1
26.jpg 8 3.51 gp1
27.jpg 9 0.88 gp1
28.jpg 3 0.21 gp1
29.jpg 5 0.36 gp1
30.jpg 7 3.72 gp1
31.jpg 4 1.7 gp1
3_1.jpg 8 5 grp3
3_2.jpg 16 4.8 grp3
3_3.jpg 8 5 grp3
3_4.jpg 10 04.03 grp3
3_5.jpg 25 12.5 grp3
3_6.jpg 8 16 grp3
3_7.jpg 8 16 grp3
3_8.jpg 50 5 grp3
3_9.jpg 24 24 grp3
3_10.jpg 35 3.5 grp3
18.jpg 8 02.01 grp5
19.jpg 10 3.19 grp5
20.jpg 12 4.17 grp5
21.jpg 8 4.22 grp5
22.jpg 12 6.19 grp5
23.jpg 20 8.88 grp5
24.jpg 26 10.05 grp5
1.jpg 2 1.50 grp4
2.jpg 4 2.27 grp4
3.jpg 5 3.27 grp4
4.jpg 7 1.88 grp4
5.jpg 8 4.38 grp4
6.jpg 7 2.37 grp4
7.jpg 8 3.88 grp4
8.jpg 8 3.88 grp4
9.jpg 4 2.65 grp4
10.jpg 7 5.12 grp4
60.jpg 13 6,33 gp6
61.jpg 11 5,53 gp6
62.jpg 9 6,86 gp6
63.jpg 9 5,34 gp6
64.jpg 12 7,07 gp6
65.jpg 13 2,63 gp6
66.jpg 7 0,77 gp6
67.jpg 10 3,31 gp6
68.jpg 11 5,41 gp6
69.jpg 9 7,4 gp6
gp7_01.webp 7 3,79 gp7
gp7_02.webp 12 1,85 gp7
gp7_03.webp 12 4,6 gp7
gp7_04.webp 13 4,65 gp7
gp7_05.webp 12 4,15 gp7
gp7_06.webp 12 4,74 gp7
gp7_07.webp 11 3,74 gp7
gp7_08.webp 10 4,19 gp7
gp7_09.webp 11 2,55 gp7
gp7_10.webp 9 4,46 gp7
gp7_11.webp 10 4,03 gp7
gp7_12.webp 14 4,95 gp7
IMG_1136.png 5 0,83 gp8
IMG_1137.png 10 2,16 gp8
IMG_1138.png 9 2,17 gp8
IMG_1139.png 4 1,21 gp8
IMG_1140.png 11 2,47 gp8
IMG_1141.png 7 1,36 gp8
IMG_1142.png 4 1,52 gp8
IMG_1143.png 17 1,4 gp8
IMG_1144.png 16 0,43 gp8
IMG_1145.png 5 3,86 gp8
1.jpeg 8 gp2
2.jpeg 2 3 gp2
3.jpeg 3 2,7 gp2
4.jpeg 8 3,86 gp2
5.jpeg 3 0,24 gp2
6.jpeg 9 3,98 gp2
7.jpeg 9 3.98 gp2
8.jpeg 3 3,5 gp2
9.jpeg 6 0,96 gp2
10.jpeg 6 0,96 gp2
11.jpeg 9 3,37 gp2
12.jpeg 2 3 gp2
13.jpeg 9 3,87 gp2
14.jpeg 4 2,45 gp2
15.jpeg 5 3,9 gp2
"""


# -----------------------------------------------------------------------------
# Inlined from src/evaluation.py
# -----------------------------------------------------------------------------




@dataclass(frozen=True)
class EvaluationItem:
    relative_path: Path
    group: str
    filename: str
    predicted: int
    expected: int
    predicted_value_cents: int
    expected_value_cents: int | None

    @property
    def diff(self) -> int:
        return self.predicted - self.expected

    @property
    def abs_diff(self) -> int:
        return abs(self.diff)

    @property
    def is_correct(self) -> bool:
        return self.diff == 0

    @property
    def has_value_ground_truth(self) -> bool:
        return self.expected_value_cents is not None

    @property
    def value_diff_cents(self) -> int | None:
        if self.expected_value_cents is None:
            return None
        return int(self.predicted_value_cents) - int(self.expected_value_cents)

    @property
    def value_abs_diff_cents(self) -> int | None:
        diff = self.value_diff_cents
        return None if diff is None else abs(diff)

    @property
    def value_is_correct(self) -> bool:
        diff = self.value_diff_cents
        return False if diff is None else diff == 0


class Evaluator:
    # Value evaluation policy:
    # - differences up to tolerance are treated as "acceptable" for accuracy
    # - larger differences are scored progressively down to 0 at soft-cap
    VALUE_TOLERANCE_CENTS = 100
    VALUE_SOFT_CAP_CENTS = 400

    def __init__(self):
        self._items: list[EvaluationItem] = []
        self._skipped_missing_ground_truth = 0
        self._skipped_filtered_group = 0

    @property
    def evaluated_count(self) -> int:
        return len(self._items)

    @property
    def skipped_missing_ground_truth(self) -> int:
        return self._skipped_missing_ground_truth

    @property
    def skipped_filtered_group(self) -> int:
        return self._skipped_filtered_group

    def add_match(
        self,
        relative_path: Path,
        group: str,
        predicted: int,
        ground_truth: GroundTruthEntry,
        predicted_value_cents: int = 0,
    ) -> EvaluationItem:
        item = EvaluationItem(
            relative_path=relative_path,
            group=group,
            filename=ground_truth.filename,
            predicted=int(predicted),
            expected=int(ground_truth.coin_count),
            predicted_value_cents=int(predicted_value_cents),
            expected_value_cents=None if ground_truth.value_cents is None else int(ground_truth.value_cents),
        )
        self._items.append(item)
        return item

    def add_missing_ground_truth(self) -> None:
        self._skipped_missing_ground_truth += 1

    def add_filtered_group(self) -> None:
        self._skipped_filtered_group += 1

    def _compute_metrics(self, items: list[EvaluationItem]) -> dict[str, float | int | None]:
        value_tolerance_cents = int(self.VALUE_TOLERANCE_CENTS)
        value_soft_cap_cents = int(max(self.VALUE_SOFT_CAP_CENTS, value_tolerance_cents + 1))

        def _value_quality_score(abs_diff_cents: int) -> float:
            # Full score inside tolerance, then linear decay to 0 at soft-cap.
            diff = float(max(0, int(abs_diff_cents)))
            tol = float(value_tolerance_cents)
            cap = float(value_soft_cap_cents)
            if diff <= tol:
                return 100.0
            if diff >= cap:
                return 0.0
            return float(100.0 * (1.0 - (diff - tol) / (cap - tol)))

        if not items:
            return {
                "evaluated": 0,
                "coin_correct": 0,
                "coin_accuracy": 0.0,
                "coin_mae": 0.0,
                "coin_total_abs_error": 0,
                "value_evaluated": 0,
                "value_correct": 0,
                "value_accuracy": 0.0,
                "value_correct_exact": 0,
                "value_accuracy_exact": 0.0,
                "value_mae_cents": 0.0,
                "value_mae_eur": 0.0,
                "value_total_abs_error_cents": 0,
                "value_tolerance_cents": value_tolerance_cents,
                "value_soft_cap_cents": value_soft_cap_cents,
                "value_error_score": 0.0,
                "coin_score": 0.0,
                "value_score": None,
                "general_score": 0.0,
                # Backward-compatible aliases.
                "correct": 0,
                "accuracy": 0.0,
                "mae": 0.0,
                "total_abs_error": 0,
            }

        evaluated = len(items)
        coin_correct = sum(1 for item in items if item.is_correct)
        coin_total_abs_error = sum(item.abs_diff for item in items)
        coin_accuracy = (coin_correct / evaluated) * 100.0
        coin_mae = coin_total_abs_error / evaluated

        value_items = [item for item in items if item.has_value_ground_truth]
        value_evaluated = len(value_items)
        value_correct_exact = sum(1 for item in value_items if item.value_is_correct)
        value_correct = sum(
            1
            for item in value_items
            if int(item.value_abs_diff_cents or 0) <= value_tolerance_cents
        )
        value_total_abs_error_cents = sum(int(item.value_abs_diff_cents or 0) for item in value_items)
        value_accuracy_exact = (value_correct_exact / value_evaluated) * 100.0 if value_evaluated > 0 else 0.0
        value_accuracy = (value_correct / value_evaluated) * 100.0 if value_evaluated > 0 else 0.0
        value_mae_cents = (value_total_abs_error_cents / value_evaluated) if value_evaluated > 0 else 0.0
        value_mae_eur = value_mae_cents / 100.0

        coin_error_score = 100.0 / (1.0 + coin_mae)
        coin_score = 0.60 * coin_accuracy + 0.40 * coin_error_score

        if value_evaluated > 0:
            quality_scores = [
                _value_quality_score(int(item.value_abs_diff_cents or 0))
                for item in value_items
            ]
            value_error_score = float(np.mean(quality_scores)) if quality_scores else 0.0
            value_score = 0.55 * value_accuracy + 0.45 * value_error_score
            general_score = 0.50 * coin_score + 0.50 * value_score
        else:
            value_error_score = 0.0
            value_score = None
            general_score = coin_score

        return {
            "evaluated": evaluated,
            "coin_correct": coin_correct,
            "coin_accuracy": coin_accuracy,
            "coin_mae": coin_mae,
            "coin_total_abs_error": coin_total_abs_error,
            "value_evaluated": value_evaluated,
            "value_correct": value_correct,
            "value_accuracy": value_accuracy,
            "value_correct_exact": value_correct_exact,
            "value_accuracy_exact": value_accuracy_exact,
            "value_mae_cents": value_mae_cents,
            "value_mae_eur": value_mae_eur,
            "value_total_abs_error_cents": value_total_abs_error_cents,
            "value_tolerance_cents": value_tolerance_cents,
            "value_soft_cap_cents": value_soft_cap_cents,
            "value_error_score": value_error_score,
            "coin_score": coin_score,
            "value_score": value_score,
            "general_score": general_score,
            # Backward-compatible aliases.
            "correct": coin_correct,
            "accuracy": coin_accuracy,
            "mae": coin_mae,
            "total_abs_error": coin_total_abs_error,
        }

    def summary(self) -> dict[str, float | int | None]:
        return self._compute_metrics(self._items)

    def summary_by_group(self) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[EvaluationItem]] = {}
        for item in self._items:
            grouped.setdefault(item.group, []).append(item)

        out: dict[str, dict[str, float | int | None]] = {}
        for group, items in grouped.items():
            out[group] = self._compute_metrics(items)
        return out


class Evaluation(Evaluator):
    """Pipeline stage class alias: ImagePreprocessing -> CoinDetector -> CoinValueEstimator -> Evaluation."""


# -----------------------------------------------------------------------------
# Inlined from src/io_utils.py
# -----------------------------------------------------------------------------




def read_bgr_or_raise(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path.resolve()}")
    return image_bgr


def letterbox_resize_to_canvas(image_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image has invalid dimensions")

    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas



# -----------------------------------------------------------------------------
# Inlined from src/preprocessing_ops.py
# -----------------------------------------------------------------------------





@dataclass(frozen=True)
class PreprocessingResult:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    color_balanced_bgr: np.ndarray
    color_balanced_rgb: np.ndarray
    hist_norm_bgr: np.ndarray
    hist_norm_rgb: np.ndarray
    clahe_bgr: np.ndarray
    clahe_rgb: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    color_norm_debug: dict[str, Any]
    hist_norm_debug: dict[str, Any]


class ImagePreprocessing:
    def __init__(
        self,
        clahe_enabled: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        color_normalization_enabled: bool = False,
        color_balance_strength: float = 0.35,
        color_balance_max_shift_ab: float = 6.0,
        color_balance_max_shift_l: float = 4.0,
        color_saturation_target: float = 108.0,
        color_saturation_strength: float = 0.14,
        histogram_normalization_enabled: bool = False,
        histogram_clip_limit: float = 2.2,
        histogram_tile_grid_size: tuple[int, int] = (8, 8),
        histogram_stretch_percentiles: tuple[float, float] = (1.0, 99.0),
        euro_reference_lab_bronze: tuple[float, float, float] = (200.0, 142.0, 145.0),
        euro_reference_lab_gold: tuple[float, float, float] = (210.0, 131.0, 143.0),
        euro_reference_lab_nickel: tuple[float, float, float] = (195.0, 129.0, 135.0),
        blur_mode: str = "gauss",
        gauss_ksize: int = 5,
        gauss_sigma: float = 2.0,
    ):
        self._clahe_enabled = bool(clahe_enabled)
        self._clahe_clip_limit = float(clahe_clip_limit)
        self._clahe_tile_grid_size = clahe_tile_grid_size
        self._color_normalization_enabled = bool(color_normalization_enabled)
        self._color_balance_strength = float(color_balance_strength)
        self._color_balance_max_shift_ab = float(color_balance_max_shift_ab)
        self._color_balance_max_shift_l = float(color_balance_max_shift_l)
        self._color_saturation_target = float(color_saturation_target)
        self._color_saturation_strength = float(color_saturation_strength)
        self._histogram_normalization_enabled = bool(histogram_normalization_enabled)
        self._histogram_clip_limit = float(histogram_clip_limit)
        self._histogram_tile_grid_size = histogram_tile_grid_size
        self._histogram_stretch_percentiles = histogram_stretch_percentiles
        self._euro_reference_lab_bronze = euro_reference_lab_bronze
        self._euro_reference_lab_gold = euro_reference_lab_gold
        self._euro_reference_lab_nickel = euro_reference_lab_nickel
        self._blur_mode = _normalize_blur_mode(blur_mode)
        self._gauss_ksize = int(gauss_ksize)
        self._gauss_sigma = float(gauss_sigma)

    @property
    def blur_step_name(self) -> str:
        return "Gaussian Blur" if self._blur_mode == "gauss" else "Median Blur"

    def process(self, image_bgr: np.ndarray) -> PreprocessingResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._color_normalization_enabled:
            color_balanced_bgr, color_norm_debug = normalize_color_to_euro_references(
                image_bgr=image_bgr,
                strength=self._color_balance_strength,
                max_shift_ab=self._color_balance_max_shift_ab,
                max_shift_l=self._color_balance_max_shift_l,
                saturation_target=self._color_saturation_target,
                saturation_strength=self._color_saturation_strength,
                bronze_ref_lab=self._euro_reference_lab_bronze,
                gold_ref_lab=self._euro_reference_lab_gold,
                nickel_ref_lab=self._euro_reference_lab_nickel,
            )
        else:
            color_balanced_bgr = image_bgr
            color_norm_debug = {"enabled": False}
        color_balanced_rgb = cv2.cvtColor(color_balanced_bgr, cv2.COLOR_BGR2RGB)

        if self._histogram_normalization_enabled:
            hist_norm_bgr, hist_norm_debug = normalize_luminance_histogram(
                image_bgr=color_balanced_bgr,
                clip_limit=self._histogram_clip_limit,
                tile_grid_size=self._histogram_tile_grid_size,
                stretch_percentiles=self._histogram_stretch_percentiles,
            )
        else:
            hist_norm_bgr = color_balanced_bgr
            hist_norm_debug = {"enabled": False}
        hist_norm_rgb = cv2.cvtColor(hist_norm_bgr, cv2.COLOR_BGR2RGB)

        if self._clahe_enabled:
            clahe_bgr, clahe_rgb = apply_clahe_on_l_channel(
                hist_norm_bgr,
                clip_limit=self._clahe_clip_limit,
                tile_grid_size=self._clahe_tile_grid_size,
            )
        else:
            clahe_bgr = hist_norm_bgr
            clahe_rgb = hist_norm_rgb

        gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        ksize = _normalize_odd_ksize(self._gauss_ksize)
        if self._blur_mode == "gauss":
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), self._gauss_sigma)
        else:
            blurred = gray if ksize <= 1 else cv2.medianBlur(gray, ksize)

        return PreprocessingResult(
            image_bgr=image_bgr,
            image_rgb=image_rgb,
            color_balanced_bgr=color_balanced_bgr,
            color_balanced_rgb=color_balanced_rgb,
            hist_norm_bgr=hist_norm_bgr,
            hist_norm_rgb=hist_norm_rgb,
            clahe_bgr=clahe_bgr,
            clahe_rgb=clahe_rgb,
            gray=gray,
            blurred=blurred,
            color_norm_debug=color_norm_debug,
            hist_norm_debug=hist_norm_debug,
        )


def _clamp_float(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def gray_world_white_balance(image_bgr: np.ndarray, gain_clip: tuple[float, float] = (0.72, 1.40)) -> np.ndarray:
    image_f = image_bgr.astype(np.float32)
    channel_means = np.mean(image_f.reshape(-1, 3), axis=0)
    mean_gray = float(np.mean(channel_means))
    gains = mean_gray / np.maximum(channel_means, 1e-6)
    gains = np.clip(gains, gain_clip[0], gain_clip[1]).astype(np.float32)
    balanced = np.clip(image_f * gains.reshape((1, 1, 3)), 0.0, 255.0).astype(np.uint8)
    return balanced


def _estimate_coin_like_masks(hsv_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = hsv_u8[:, :, 0]
    s = hsv_u8[:, :, 1]
    v = hsv_u8[:, :, 2]

    warm_mask = (
        (h >= 7)
        & (h <= 35)
        & (s >= 26)
        & (v >= 30)
        & (v <= 245)
    )
    neutral_mask = (
        (s <= 45)
        & (v >= 35)
        & (v <= 235)
    )
    return warm_mask, neutral_mask


def normalize_color_to_euro_references(
    image_bgr: np.ndarray,
    strength: float = 0.35,
    max_shift_ab: float = 6.0,
    max_shift_l: float = 4.0,
    saturation_target: float = 108.0,
    saturation_strength: float = 0.14,
    bronze_ref_lab: tuple[float, float, float] = (200.0, 142.0, 145.0),
    gold_ref_lab: tuple[float, float, float] = (210.0, 131.0, 143.0),
    nickel_ref_lab: tuple[float, float, float] = (195.0, 129.0, 135.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    wb_bgr = gray_world_white_balance(image_bgr)
    hsv = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    warm_mask, neutral_mask = _estimate_coin_like_masks(hsv)

    target_warm_lab = (
        0.50 * float(bronze_ref_lab[0]) + 0.50 * float(gold_ref_lab[0]),
        0.46 * float(bronze_ref_lab[1]) + 0.54 * float(gold_ref_lab[1]),
        0.44 * float(bronze_ref_lab[2]) + 0.56 * float(gold_ref_lab[2]),
    )

    shift_l = 0.0
    shift_a = 0.0
    shift_b = 0.0
    warm_count = int(np.sum(warm_mask))
    warm_mean_lab = None

    if warm_count >= 300:
        warm_pixels = lab[warm_mask]
        warm_mean = np.median(warm_pixels, axis=0)
        warm_mean_lab = [float(warm_mean[0]), float(warm_mean[1]), float(warm_mean[2])]
        shift_l = _clamp_float(
            (target_warm_lab[0] - warm_mean[0]) * 0.18 * strength,
            -max_shift_l,
            max_shift_l,
        )
        shift_a = _clamp_float(
            (target_warm_lab[1] - warm_mean[1]) * 0.45 * strength,
            -max_shift_ab,
            max_shift_ab,
        )
        shift_b = _clamp_float(
            (target_warm_lab[2] - warm_mean[2]) * 0.45 * strength,
            -max_shift_ab,
            max_shift_ab,
        )

    neutral_count = int(np.sum(neutral_mask))
    neutral_a_shift = 0.0
    neutral_b_shift = 0.0
    if neutral_count >= 400:
        neutral_pixels = lab[neutral_mask]
        neutral_mean = np.median(neutral_pixels, axis=0)
        neutral_a_shift = _clamp_float(
            (float(nickel_ref_lab[1]) - float(neutral_mean[1])) * 0.20 * strength,
            -3.0,
            3.0,
        )
        neutral_b_shift = _clamp_float(
            (float(nickel_ref_lab[2]) - float(neutral_mean[2])) * 0.20 * strength,
            -3.0,
            3.0,
        )

    lab[:, :, 0] = np.clip(lab[:, :, 0] + shift_l, 0.0, 255.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + shift_a + neutral_a_shift, 0.0, 255.0)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + shift_b + neutral_b_shift, 0.0, 255.0)

    balanced_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    balanced_hsv = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    sat_values = balanced_hsv[:, :, 1][warm_mask]
    if sat_values.size < 120:
        sat_values = balanced_hsv[:, :, 1].reshape(-1)
    sat_median = float(np.median(sat_values)) if sat_values.size > 0 else 0.0
    sat_scale = 1.0
    if sat_median > 1.0:
        sat_scale = (float(saturation_target) / sat_median) ** float(saturation_strength)
        sat_scale = _clamp_float(sat_scale, 0.93, 1.08)
        balanced_hsv[:, :, 1] = np.clip(balanced_hsv[:, :, 1] * sat_scale, 0.0, 255.0)

    balanced_bgr = cv2.cvtColor(balanced_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    debug_info: dict[str, Any] = {
        "enabled": True,
        "warm_pixel_count": warm_count,
        "neutral_pixel_count": neutral_count,
        "warm_mean_lab": warm_mean_lab,
        "target_warm_lab": [float(target_warm_lab[0]), float(target_warm_lab[1]), float(target_warm_lab[2])],
        "shift_l": float(shift_l),
        "shift_a": float(shift_a + neutral_a_shift),
        "shift_b": float(shift_b + neutral_b_shift),
        "reference_lab": {
            "bronze": [float(bronze_ref_lab[0]), float(bronze_ref_lab[1]), float(bronze_ref_lab[2])],
            "gold": [float(gold_ref_lab[0]), float(gold_ref_lab[1]), float(gold_ref_lab[2])],
            "nickel": [float(nickel_ref_lab[0]), float(nickel_ref_lab[1]), float(nickel_ref_lab[2])],
        },
        "sat_median_before": float(sat_median),
        "sat_scale": float(sat_scale),
    }
    return balanced_bgr, debug_info


def _percentile_stretch_u8(channel_u8: np.ndarray, p_low: float, p_high: float) -> tuple[np.ndarray, float, float]:
    low = float(np.percentile(channel_u8, p_low))
    high = float(np.percentile(channel_u8, p_high))
    if high <= low + 1.0:
        return channel_u8.copy(), low, high
    out = ((channel_u8.astype(np.float32) - low) * (255.0 / (high - low)))
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out, low, high


def normalize_luminance_histogram(
    image_bgr: np.ndarray,
    clip_limit: float = 2.2,
    tile_grid_size: tuple[int, int] = (8, 8),
    stretch_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    p_low, p_high = stretch_percentiles
    l_norm, lo, hi = _percentile_stretch_u8(l_clahe, p_low=float(p_low), p_high=float(p_high))
    out_bgr = cv2.cvtColor(cv2.merge((l_norm, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    debug_info: dict[str, Any] = {
        "enabled": True,
        "clip_limit": float(clip_limit),
        "tile_grid_size": [int(tile_grid_size[0]), int(tile_grid_size[1])],
        "stretch_percentiles": [float(p_low), float(p_high)],
        "l_percentile_low": float(lo),
        "l_percentile_high": float(hi),
    }
    return out_bgr, debug_info


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


def _normalize_blur_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"gauss", "gaussian"}:
        return "gauss"
    if normalized == "median":
        return "median"
    raise ValueError("Invalid blur_mode. Expected one of: 'gauss', 'gaussian', 'median'.")


def _normalize_odd_ksize(ksize: int) -> int:
    k = int(ksize)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1


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


# -----------------------------------------------------------------------------
# Inlined from src/hough_detection.py
# -----------------------------------------------------------------------------






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


# -----------------------------------------------------------------------------
# Inlined from src/value_estimator.py
# -----------------------------------------------------------------------------








# -----------------------------------------------------------------------------
# CoinValueEstimator internals (consolidated for pipeline-stage OOP structure)
# -----------------------------------------------------------------------------

def mean_hsv_from_pixels(hsv_pixels: np.ndarray) -> tuple[int, int, int]:
    if hsv_pixels is None or len(hsv_pixels) == 0:
        return 0, 0, 0

    h_vals = hsv_pixels[:, 0].astype(np.float32)
    s_vals = hsv_pixels[:, 1].astype(np.float32)
    v_vals = hsv_pixels[:, 2].astype(np.float32)

    theta = h_vals * (2.0 * np.pi / 180.0)
    weights = np.clip(s_vals, 1.0, 255.0)
    mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
    mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))

    hue_deg = np.degrees(np.arctan2(mean_sin, mean_cos))
    if hue_deg < 0.0:
        hue_deg += 360.0

    h_cv = int(np.clip(round(hue_deg / 2.0), 0, 179))
    s_cv = int(np.clip(round(np.median(s_vals)), 0, 255))
    v_cv = int(np.clip(round(np.median(v_vals)), 0, 255))
    return h_cv, s_cv, v_cv


def mean_lab_from_pixels(lab_pixels: np.ndarray) -> tuple[float, float, float]:
    if lab_pixels is None or len(lab_pixels) == 0:
        return 0.0, 0.0, 0.0

    l_vals = lab_pixels[:, 0].astype(np.float32)
    a_vals = lab_pixels[:, 1].astype(np.float32)
    b_vals = lab_pixels[:, 2].astype(np.float32)
    return float(np.mean(l_vals)), float(np.mean(a_vals)), float(np.mean(b_vals))


def lab_distance(lab_a: tuple[float, float, float], lab_b: tuple[float, float, float]) -> float:
    dl = float(lab_a[0]) - float(lab_b[0])
    da = float(lab_a[1]) - float(lab_b[1])
    db = float(lab_a[2]) - float(lab_b[2])
    return float(np.sqrt((0.55 * dl) * (0.55 * dl) + da * da + db * db))


def lab_warmth(lab_color: tuple[float, float, float]) -> float:
    a_shift = float(lab_color[1]) - 128.0
    b_shift = float(lab_color[2]) - 128.0
    return float(b_shift - 0.45 * max(a_shift, 0.0))


def hsv_similarity_score(hsv_a: tuple[int, int, int], hsv_b: tuple[int, int, int]) -> float:
    _, sa, _ = hsv_a
    _, sb, _ = hsv_b

    ds = abs(float(sa) - float(sb)) / 255.0
    sim = 1.0 - ds
    return float(np.clip(sim, 0.0, 1.0))


def hue_circular_delta_cv(h1: int, h2: int) -> float:
    d = abs(float(h1) - float(h2))
    return float(min(d, 180.0 - d))


def choose_dynamic_sat_delta_threshold(sat_deltas: list[float], default: float = 18.0) -> float:
    if sat_deltas is None:
        return float(default)

    vals = np.asarray(sat_deltas, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)

    vals = np.sort(vals)
    n = int(vals.size)
    if n < 4:
        return float(np.max(vals) + 1.0)

    gaps = vals[1:] - vals[:-1]
    if gaps.size == 0:
        return float(np.max(vals) + 1.0)

    median_gap = float(np.median(gaps))
    min_low = max(3, int(np.ceil(0.55 * n)))

    valid_idx = [i for i in range(len(gaps)) if (i + 1) >= min_low and (n - i - 1) >= 1]
    if len(valid_idx) == 0:
        return float(np.max(vals) + 1.0)

    best_i = max(valid_idx, key=lambda i: float(gaps[i]))
    best_gap = float(gaps[best_i])

    if best_gap < max(6.0, 2.2 * max(median_gap, 1.0)):
        return float(np.max(vals) + 1.0)

    low = vals[: best_i + 1]
    high = vals[best_i + 1 :]

    n_high = int(high.size)
    high_frac = float(n_high / n)
    if high_frac > 0.45:
        return float(np.max(vals) + 1.0)

    sep = float(np.mean(high) - np.mean(low))
    if sep < max(8.0, 1.3 * float(np.std(low) + 1.0)):
        return float(np.max(vals) + 1.0)

    thr = float(0.5 * (vals[best_i] + vals[best_i + 1]))
    p60 = float(np.percentile(vals, 60))
    p99 = float(np.percentile(vals, 99))
    return float(np.clip(thr, p60, p99))

def clamp_value(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def bronze_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    h_term = clamp_value((18.0 - float(h)) / 3.0, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 45.0) / 75.0, 0.0, 1.0)
    val_conf = clamp_value((float(v) - 35.0) / 90.0, 0.0, 1.0)
    conf = 0.65 * sat_conf + 0.35 * val_conf
    return float(np.clip(h_term * max(0.35, conf), 0.0, 1.0))


def gold_score_from_inner_hsv(h: int, s: int, v: int) -> float:
    hue_center = 22.0
    hue_width = 8.5
    h_term = 1.0 - clamp_value(abs(float(h) - hue_center) / hue_width, 0.0, 1.0)
    sat_conf = clamp_value((float(s) - 40.0) / 90.0, 0.0, 1.0)
    val_conf = clamp_value((float(v) - 55.0) / 110.0, 0.0, 1.0)
    conf = 0.60 * sat_conf + 0.40 * val_conf
    return float(np.clip(h_term * max(0.30, conf), 0.0, 1.0))


def bronze_score_from_inner_lab(l: float, a: float, b: float) -> float:
    a_shift = float(a) - 128.0
    b_shift = float(b) - 128.0
    red_term = clamp_value((a_shift - 4.0) / 22.0, 0.0, 1.0)
    warm_term = clamp_value((b_shift - 2.0) / 26.0, 0.0, 1.0)
    dark_term = clamp_value((150.0 - float(l)) / 80.0, 0.0, 1.0)
    return float(np.clip(0.55 * red_term + 0.25 * warm_term + 0.20 * dark_term, 0.0, 1.0))


def gold_score_from_inner_lab(l: float, a: float, b: float) -> float:
    a_shift = float(a) - 128.0
    b_shift = float(b) - 128.0
    yellow_term = clamp_value((b_shift - 10.0) / 24.0, 0.0, 1.0)
    bright_term = clamp_value((float(l) - 92.0) / 70.0, 0.0, 1.0)
    low_red_term = 1.0 - clamp_value((a_shift - 14.0) / 22.0, 0.0, 1.0)
    red_dominance = clamp_value((a_shift - (0.72 * b_shift + 1.5)) / 15.0, 0.0, 1.0)
    base_score = 0.55 * yellow_term + 0.25 * bright_term + 0.20 * low_red_term
    return float(np.clip(base_score * (1.0 - 0.45 * red_dominance), 0.0, 1.0))


def label_material_from_inner_hsv(h: int, s: int, v: int) -> str:
    if s < 45 or v < 35:
        return "borderline"
    if h <= 15:
        return "bronze"
    if h >= 18:
        return "gold"
    return "borderline"


def label_material_from_inner_lab(l: float, a: float, b: float) -> str:
    bronze_score = bronze_score_from_inner_lab(l, a, b)
    gold_score = gold_score_from_inner_lab(l, a, b)
    if abs(gold_score - bronze_score) < 0.10:
        return "borderline"
    return "gold" if gold_score > bronze_score else "bronze"


def classify_material_hybrid(
    h: int,
    s: int,
    v: int,
    l: float,
    a: float,
    b: float,
) -> dict[str, float | str]:
    bronze_hsv = bronze_score_from_inner_hsv(h, s, v)
    gold_hsv = gold_score_from_inner_hsv(h, s, v)
    bronze_lab = bronze_score_from_inner_lab(l, a, b)
    gold_lab = gold_score_from_inner_lab(l, a, b)

    hsv_weight = 0.62 if s >= 70 else 0.54
    lab_weight = 1.0 - hsv_weight

    bronze_score = hsv_weight * bronze_hsv + lab_weight * bronze_lab
    gold_score = hsv_weight * gold_hsv + lab_weight * gold_lab

    a_shift = float(a) - 128.0
    b_shift = float(b) - 128.0
    red_dominance = clamp_value((a_shift - (0.72 * b_shift + 1.5)) / 15.0, 0.0, 1.0)

    if h <= 15 and s >= 55:
        bronze_score = min(1.0, bronze_score + 0.16)
        gold_score = max(0.0, gold_score - 0.12)
    elif h >= 20 and s >= 55:
        gold_score = min(1.0, gold_score + 0.08)

    if red_dominance > 0.0:
        bronze_score = min(1.0, bronze_score + 0.12 * red_dominance)
        gold_score = max(0.0, gold_score - 0.16 * red_dominance)

    margin = abs(gold_score - bronze_score)
    max_score = max(gold_score, bronze_score)
    if max_score < 0.35 or margin < 0.12:
        label = "borderline"
    else:
        label = "gold" if gold_score > bronze_score else "bronze"

    return {
        "label": label,
        "bronze_score": float(np.clip(bronze_score, 0.0, 1.0)),
        "gold_score": float(np.clip(gold_score, 0.0, 1.0)),
        "bronze_score_hsv": float(np.clip(bronze_hsv, 0.0, 1.0)),
        "gold_score_hsv": float(np.clip(gold_hsv, 0.0, 1.0)),
        "bronze_score_lab": float(np.clip(bronze_lab, 0.0, 1.0)),
        "gold_score_lab": float(np.clip(gold_lab, 0.0, 1.0)),
        "margin": float(margin),
    }

def label_bimetal_euro_from_saturation(inner_s: int, border_s: int) -> str:
    if inner_s < border_s:
        return "1-euro-like"
    if inner_s > border_s:
        return "2-euro-like"
    return "bi-metal-euro-uncertain"


def label_bimetal_euro_from_mean_lab(
    inner_lab: tuple[float, float, float],
    border_lab: tuple[float, float, float],
) -> str:
    inner_warm = lab_warmth(inner_lab)
    border_warm = lab_warmth(border_lab)
    delta_warm = inner_warm - border_warm

    if delta_warm >= 2.0:
        return "2-euro-like"
    if delta_warm <= -2.0:
        return "1-euro-like"

    inner_chroma = float(
        np.hypot(float(inner_lab[1]) - 128.0, float(inner_lab[2]) - 128.0)
    )
    border_chroma = float(
        np.hypot(float(border_lab[1]) - 128.0, float(border_lab[2]) - 128.0)
    )
    if (inner_chroma - border_chroma) >= 4.0:
        return "2-euro-like"
    if (border_chroma - inner_chroma) >= 4.0:
        return "1-euro-like"
    return "bi-metal-euro-uncertain"


def coin_kmeans_radial_structure(hsv_pixels: np.ndarray, radial_norm: np.ndarray) -> dict[str, float | bool]:
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0

    w = np.clip(s, 0.2, 1.0)
    feats = np.stack([np.cos(h) * w, np.sin(h) * w, s, 0.35 * v], axis=1).astype(np.float32)

    if n > 4000:
        rng = np.random.default_rng(123)
        idx = rng.choice(n, size=4000, replace=False)
        feats_fit = feats[idx]
    else:
        feats_fit = feats

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.02)
    cv2.setRNGSeed(1234)
    _, _, centers = cv2.kmeans(feats_fit, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    d0 = np.sum((feats - centers[0]) ** 2, axis=1)
    d1 = np.sum((feats - centers[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.int32)

    cnt0 = int(np.sum(labels == 0))
    cnt1 = int(np.sum(labels == 1))
    balance = float(min(cnt0, cnt1) / max(n, 1))
    if balance < 0.08:
        return {
            "ok": False,
            "score": float(np.clip(balance / 0.08, 0.0, 1.0)),
            "radial_sep": 0.0,
            "agreement": 0.0,
            "balance": balance,
        }

    r0 = float(np.mean(radial_norm[labels == 0]))
    r1 = float(np.mean(radial_norm[labels == 1]))
    radial_sep = abs(r0 - r1)

    outer_id = 0 if r0 > r1 else 1
    inner_mean = min(r0, r1)
    outer_mean = max(r0, r1)
    radial_cut = 0.5 * (inner_mean + outer_mean)

    pred_outer = radial_norm >= radial_cut
    lbl_outer = labels == outer_id
    agreement = float(np.mean(pred_outer == lbl_outer))

    score = (
        0.45 * np.clip(radial_sep / 0.22, 0.0, 1.0)
        + 0.35 * np.clip((agreement - 0.5) / 0.4, 0.0, 1.0)
        + 0.20 * np.clip(balance / 0.22, 0.0, 1.0)
    )
    ok = (radial_sep >= 0.13) and (agreement >= 0.66) and (balance >= 0.10)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "radial_sep": float(radial_sep),
        "agreement": float(agreement),
        "balance": float(balance),
    }


def coin_radial_step_score(
    hsv_pixels: np.ndarray,
    radial_norm: np.ndarray,
    split_radius: float,
    bins: int = 8,
) -> dict[str, float | bool]:
    n = int(hsv_pixels.shape[0])
    if n < 120:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    h = hsv_pixels[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv_pixels[:, 1].astype(np.float32) / 255.0
    v = hsv_pixels[:, 2].astype(np.float32) / 255.0
    feats = np.stack([np.cos(h) * s, np.sin(h) * s, s, 0.20 * v], axis=1).astype(np.float32)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_vecs: list[np.ndarray | None] = []
    boundary_r: list[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (radial_norm >= lo) & (radial_norm <= hi)
        else:
            mask = (radial_norm >= lo) & (radial_norm < hi)
        if int(np.sum(mask)) < max(20, int(0.03 * n)):
            bin_vecs.append(None)
        else:
            bin_vecs.append(np.mean(feats[mask], axis=0))
        if i < bins - 1:
            boundary_r.append(float(edges[i + 1]))

    steps: list[float] = []
    for i in range(bins - 1):
        a = bin_vecs[i]
        b = bin_vecs[i + 1]
        if (a is None) or (b is None):
            steps.append(0.0)
        else:
            steps.append(float(np.linalg.norm(a - b)))

    if len(steps) == 0:
        return {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": float(split_radius)}

    idx = int(np.argmax(steps))
    max_step = float(steps[idx])
    step_radius = float(boundary_r[idx])
    dist_to_split = abs(step_radius - float(split_radius))

    strength = np.clip(max_step / 0.17, 0.0, 1.0)
    location = np.clip(1.0 - (dist_to_split / 0.28), 0.0, 1.0)
    score = float(0.65 * strength + 0.35 * location)
    ok = (max_step >= 0.10) and (dist_to_split <= 0.26)
    return {
        "ok": bool(ok),
        "score": float(np.clip(score, 0.0, 1.0)),
        "max_step": float(max_step),
        "step_radius": float(step_radius),
    }


def coin_edge_roughness_score(
    gray_u8: np.ndarray,
    x: int,
    y: int,
    r: int,
    n_bins: int = 72,
) -> dict[str, float]:
    if r < 8:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    h, w = gray_u8.shape[:2]
    x0 = max(0, int(x - 1.3 * r))
    x1 = min(w, int(x + 1.3 * r) + 1)
    y0 = max(0, int(y - 1.3 * r))
    y1 = min(h, int(y + 1.3 * r) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    roi = gray_u8[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, 60, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    xs = xs.astype(np.float32) + float(x0)
    ys = ys.astype(np.float32) + float(y0)
    dx = xs - float(x)
    dy = ys - float(y)
    dist = np.sqrt(dx * dx + dy * dy)

    mask = (dist >= 0.62 * float(r)) & (dist <= 1.20 * float(r))
    if int(np.sum(mask)) < 18:
        return {"roughness": 0.0, "coverage": 0.0, "score": 0.0}

    dist = dist[mask]
    ang = np.arctan2(dy[mask], dx[mask])
    ang[ang < 0.0] += 2.0 * np.pi

    bins = np.full((n_bins,), np.nan, dtype=np.float32)
    idx = np.floor((ang / (2.0 * np.pi)) * n_bins).astype(np.int32)
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        vals = dist[idx == b]
        if vals.size > 0:
            bins[b] = float(np.median(vals))

    valid = np.isfinite(bins)
    coverage = float(np.mean(valid))
    if int(np.sum(valid)) < max(14, int(0.30 * n_bins)):
        return {"roughness": 0.0, "coverage": coverage, "score": 0.0}

    vals = bins[valid].astype(np.float32)
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    roughness = float((q3 - q1) / (med + 1e-6))

    score = np.clip((roughness - 0.040) / 0.032, 0.0, 1.0)
    score *= np.clip((coverage - 0.35) / 0.45, 0.0, 1.0)
    return {
        "roughness": float(roughness),
        "coverage": float(coverage),
        "score": float(np.clip(score, 0.0, 1.0)),
    }

@dataclass(frozen=True)
class CoinAnalyzerConfig:
    border_ratio: float = 0.24
    sat_delta_threshold: float | None = None
    bimetal_mode: str = "mean-color"
    material_mode: str = "hybrid"


class CoinAnalyzer:
    def __init__(self, config: CoinAnalyzerConfig | None = None):
        self._cfg = config or CoinAnalyzerConfig()

    def analyze(self, image_bgr: np.ndarray, circles: np.ndarray | None) -> tuple[np.ndarray, list[dict]]:
        return draw_and_analyze_circle_inner_border_colors(
            image_bgr,
            circles,
            border_ratio=self._cfg.border_ratio,
            sat_delta_threshold=self._cfg.sat_delta_threshold,
            bimetal_mode=self._cfg.bimetal_mode,
            material_mode=self._cfg.material_mode,
        )

    @staticmethod
    def draw_circles_filled_with_average_hue(
        image_bgr: np.ndarray, circles: np.ndarray | None
    ) -> tuple[np.ndarray, int]:
        output_bgr = image_bgr.copy()
        if circles is None:
            return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), 0

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        circles_int = np.round(circles[0, :]).astype(int)
        h_img, w_img = image_bgr.shape[:2]

        for idx, (x, y, r) in enumerate(circles_int, start=1):
            if r <= 1:
                continue
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

            pix = hsv[mask > 0]
            if pix.size == 0:
                continue

            h_vals = pix[:, 0].astype(np.float32)
            s_vals = pix[:, 1].astype(np.float32)
            v_vals = pix[:, 2].astype(np.float32)

            theta = h_vals * (2.0 * np.pi / 180.0)
            weights = np.clip(s_vals, 1.0, 255.0)
            mean_sin = float(np.sum(np.sin(theta) * weights) / np.sum(weights))
            mean_cos = float(np.sum(np.cos(theta) * weights) / np.sum(weights))
            hue = np.degrees(np.arctan2(mean_sin, mean_cos))
            if hue < 0.0:
                hue += 360.0

            hue_cv = int(np.clip(round(hue / 2.0), 0, 179))
            sat_cv = int(np.clip(round(np.median(s_vals)), 35, 255))
            val_cv = int(np.clip(round(np.median(v_vals)), 35, 255))
            fill_hsv = np.uint8([[[hue_cv, sat_cv, val_cv]]])
            fill_bgr = cv2.cvtColor(fill_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            cv2.circle(output_bgr, (int(x), int(y)), int(r), tuple(int(c) for c in fill_bgr), -1)

            text = str(idx)
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output_bgr,
                text,
                (int(x) - 8, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), int(len(circles_int))


def radius_spread_metric(circles_int: np.ndarray | None) -> float:
    if circles_int is None or len(circles_int) == 0:
        return 1e9
    r = circles_int[:, 2].astype(np.float32)
    med = float(np.median(r))
    if med <= 1e-6:
        return 1e9
    q1, q3 = np.percentile(r, [25, 75])
    iqr = float(q3 - q1)
    return iqr / med


def draw_and_analyze_circle_inner_border_colors(
    image_bgr: np.ndarray,
    circles: np.ndarray | None,
    border_ratio: float = 0.24,
    sat_delta_threshold: float | None = None,
    bimetal_mode: str = "mean-color",
    material_mode: str = "hybrid",
) -> tuple[np.ndarray, list[dict]]:
    bimetal_mode = str(bimetal_mode).strip().lower()
    if bimetal_mode not in ("hybrid", "mean-color"):
        raise ValueError("bimetal_mode must be one of: 'hybrid', 'mean-color'")
    material_mode = str(material_mode).strip().lower()
    if material_mode not in ("hsv", "lab", "hybrid"):
        raise ValueError("material_mode must be one of: 'hsv', 'lab', 'hybrid'")

    output_bgr = image_bgr.copy()
    stats: list[dict] = []
    if circles is None:
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray_shape = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    circles_int = np.round(circles[0, :]).astype(int)
    h_img, w_img = image_bgr.shape[:2]

    rows: list[dict] = []
    color_deltas_hybrid: list[float] = []
    color_deltas_mean: list[float] = []

    for idx, (x, y, r) in enumerate(circles_int):
        x = int(x)
        y = int(y)
        r = int(max(1, r))
        inner_r = int(max(1, round(r * (1.0 - border_ratio))))
        split_radius = float(inner_r) / float(r)

        outer_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        inner_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), r, 255, -1)
        cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
        border_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        pix_full_hsv = hsv[outer_mask > 0]
        pix_inner_hsv = hsv[inner_mask > 0]
        pix_border_hsv = hsv[border_mask > 0]
        pix_full_lab = lab[outer_mask > 0]
        pix_inner_lab = lab[inner_mask > 0]
        pix_border_lab = lab[border_mask > 0]

        full_hsv = mean_hsv_from_pixels(pix_full_hsv)
        inner_hsv = mean_hsv_from_pixels(pix_inner_hsv)
        border_hsv = mean_hsv_from_pixels(pix_border_hsv)
        full_lab = mean_lab_from_pixels(pix_full_lab)
        inner_lab = mean_lab_from_pixels(pix_inner_lab)
        border_lab = mean_lab_from_pixels(pix_border_lab)

        similarity = hsv_similarity_score(inner_hsv, border_hsv)
        sat_delta = abs(float(inner_hsv[1]) - float(border_hsv[1]))
        hue_delta = hue_circular_delta_cv(int(inner_hsv[0]), int(border_hsv[0]))
        val_delta = abs(float(inner_hsv[2]) - float(border_hsv[2]))

        sat_mean = 0.5 * (float(inner_hsv[1]) + float(border_hsv[1]))
        sat_conf = float(np.clip(sat_mean / 70.0, 0.0, 1.0))
        hue_term = hue_delta * sat_conf
        value_term = 0.15 * val_delta
        color_delta = float(sat_delta + hue_term + value_term)

        lab_delta = lab_distance(inner_lab, border_lab)
        mean_warm_delta = abs(lab_warmth(inner_lab) - lab_warmth(border_lab))
        mean_color_delta = float(lab_delta + 0.18 * mean_warm_delta)

        color_deltas_hybrid.append(color_delta)
        color_deltas_mean.append(mean_color_delta)

        ys, xs = np.where(outer_mask > 0)
        if len(xs) > 0:
            radial_norm = (
                np.sqrt((xs.astype(np.float32) - float(x)) ** 2 + (ys.astype(np.float32) - float(y)) ** 2)
                / float(r)
            )
            radial_norm = np.clip(radial_norm, 0.0, 1.0)
            coin_hsv = hsv[ys, xs]
            radial_kmeans = coin_kmeans_radial_structure(coin_hsv, radial_norm)
            radial_step = coin_radial_step_score(coin_hsv, radial_norm, split_radius=split_radius, bins=8)
        else:
            radial_kmeans = {"ok": False, "score": 0.0, "radial_sep": 0.0, "agreement": 0.0, "balance": 0.0}
            radial_step = {"ok": False, "score": 0.0, "max_step": 0.0, "step_radius": split_radius}

        edge_shape = coin_edge_roughness_score(gray_shape, x=x, y=y, r=r, n_bins=72)
        rows.append(
            {
                "id": idx,
                "x": x,
                "y": y,
                "r": r,
                "inner_r": inner_r,
                "full_hsv": full_hsv,
                "inner_hsv": inner_hsv,
                "border_hsv": border_hsv,
                "full_lab": full_lab,
                "inner_lab": inner_lab,
                "border_lab": border_lab,
                "similarity": float(similarity),
                "sat_delta": float(sat_delta),
                "hue_delta": float(hue_delta),
                "val_delta": float(val_delta),
                "sat_conf": float(sat_conf),
                "hue_term": float(hue_term),
                "value_term": float(value_term),
                "color_delta": float(color_delta),
                "lab_delta": float(lab_delta),
                "mean_warm_delta": float(mean_warm_delta),
                "mean_color_delta": float(mean_color_delta),
                "kmeans_radial_ok": bool(radial_kmeans["ok"]),
                "kmeans_radial_score": float(radial_kmeans["score"]),
                "kmeans_radial_sep": float(radial_kmeans["radial_sep"]),
                "kmeans_radial_agreement": float(radial_kmeans["agreement"]),
                "kmeans_radial_balance": float(radial_kmeans["balance"]),
                "step_ok": bool(radial_step["ok"]),
                "step_score": float(radial_step["score"]),
                "step_max": float(radial_step["max_step"]),
                "step_radius": float(radial_step["step_radius"]),
                "edge_roughness": float(edge_shape["roughness"]),
                "edge_coverage": float(edge_shape["coverage"]),
                "gold_flower_score": float(edge_shape["score"]),
            }
        )

    outlier_threshold_hybrid = choose_dynamic_sat_delta_threshold(color_deltas_hybrid, default=18.0)
    outlier_threshold_mean = choose_dynamic_sat_delta_threshold(color_deltas_mean, default=14.0)
    if sat_delta_threshold is not None:
        if bimetal_mode == "hybrid":
            outlier_threshold_hybrid = float(sat_delta_threshold)
        else:
            outlier_threshold_mean = float(sat_delta_threshold)

    hybrid_abs_mid = 16.0
    hybrid_abs_hi = 22.0
    hybrid_abs_vhi = 30.0
    hybrid_abs_lo = 10.0
    mean_abs_mid = 11.5
    mean_abs_hi = 15.5
    mean_abs_vhi = 22.0
    mean_abs_lo = 8.0

    for row in rows:
        x = row["x"]
        y = row["y"]
        r = row["r"]
        inner_r = row["inner_r"]
        full_hsv = row["full_hsv"]
        inner_hsv = row["inner_hsv"]
        border_hsv = row["border_hsv"]
        inner_lab = row["inner_lab"]
        border_lab = row["border_lab"]

        color_delta = float(row["color_delta"])
        sat_delta = float(row["sat_delta"])
        hue_delta = float(row["hue_delta"])
        lab_delta = float(row["lab_delta"])
        mean_warm_delta = float(row["mean_warm_delta"])
        mean_color_delta = float(row["mean_color_delta"])
        bronze_veto_applied = False
        outlier_vote_for_veto = False
        kmeans_support_for_veto = False

        if bimetal_mode == "hybrid":
            outlier_threshold = outlier_threshold_hybrid
            abs_mid = hybrid_abs_mid
            abs_hi = hybrid_abs_hi
            abs_vhi = hybrid_abs_vhi
            abs_lo = hybrid_abs_lo

            outlier_vote = color_delta >= outlier_threshold
            mid_delta = color_delta >= abs_mid
            strong_delta = color_delta >= abs_hi
            very_strong_delta = color_delta >= abs_vhi

            kmeans_ok = bool(row["kmeans_radial_ok"])
            step_ok = bool(row["step_ok"])
            kmeans_strong = bool(kmeans_ok and (float(row["kmeans_radial_score"]) >= 0.82))
            step_strong = bool(step_ok and (float(row["step_score"]) >= 0.70))
            kmeans_support = bool(
                kmeans_ok
                and (float(row["kmeans_radial_agreement"]) >= 0.64)
                and (float(row["kmeans_radial_balance"]) >= 0.12)
            )
            step_support = bool(step_ok and (float(row["step_score"]) >= 0.75))
            structure_votes = int(kmeans_support) + int(step_support)
            structure_strong_votes = int(kmeans_strong) + int(step_strong)
            outlier_vote_for_veto = bool(outlier_vote)
            kmeans_support_for_veto = bool(kmeans_support)

            strong_color_evidence = bool(
                (sat_delta >= 22.0) or (hue_delta >= 10.0 and sat_delta >= 10.0) or (color_delta >= abs_vhi)
            )
            evidence = int(strong_delta) + int(outlier_vote) + structure_votes + int(strong_color_evidence)

            if strong_color_evidence and very_strong_delta and (outlier_vote or kmeans_support):
                detector_type = "bi-metal-like"
            elif strong_color_evidence and strong_delta and (
                (outlier_vote and structure_votes >= 1)
                or kmeans_support
            ):
                detector_type = "bi-metal-like"
            elif strong_color_evidence and mid_delta and outlier_vote and (kmeans_support and step_support):
                detector_type = "bi-metal-like"
            elif (not outlier_vote) and (color_delta < 20.0) and (sat_delta <= 18.0) and (hue_delta <= 8.0):
                detector_type = "one-color-like"
            elif (color_delta <= 14.0) and (sat_delta <= 12.0) and (hue_delta <= 10.0):
                detector_type = "one-color-like"
            elif (color_delta <= abs_lo) and (structure_votes <= 1):
                detector_type = "one-color-like"
            elif (not strong_color_evidence) and (not outlier_vote) and (color_delta <= (abs_hi + 2.0)):
                detector_type = "one-color-like"
            elif (
                (not outlier_vote)
                and (not kmeans_support)
                and strong_color_evidence
                and (int(inner_hsv[0]) <= 15)
                and (int(inner_hsv[1]) >= 120)
            ):
                detector_type = "one-color-like"
            else:
                detector_type = "uncertain"
        else:
            outlier_threshold = outlier_threshold_mean
            abs_mid = mean_abs_mid
            abs_hi = mean_abs_hi
            abs_vhi = mean_abs_vhi
            abs_lo = mean_abs_lo

            outlier_vote = mean_color_delta >= outlier_threshold
            strong_delta = mean_color_delta >= abs_hi
            very_strong_delta = mean_color_delta >= abs_vhi

            strong_color_evidence = bool(
                strong_delta or (lab_delta >= 14.0) or (mean_warm_delta >= 5.0 and mean_color_delta >= abs_mid)
            )
            evidence = int(strong_delta) + int(outlier_vote) + int(lab_delta >= 14.0) + int(mean_warm_delta >= 5.0)
            outlier_vote_for_veto = bool(outlier_vote)

            if strong_color_evidence and (very_strong_delta or (strong_delta and outlier_vote)):
                detector_type = "bi-metal-like"
            elif (mean_color_delta <= abs_lo) and (mean_warm_delta <= 3.2) and (sat_delta <= 16.0):
                detector_type = "one-color-like"
            elif (not strong_color_evidence) and (not outlier_vote) and (mean_color_delta <= (abs_hi + 1.5)):
                detector_type = "one-color-like"
            else:
                detector_type = "uncertain"

        if detector_type == "bi-metal-like" and bimetal_mode == "hybrid":
            inner_h = int(inner_hsv[0])
            inner_s = int(inner_hsv[1])
            inner_v = int(inner_hsv[2])
            inner_l = float(inner_lab[0])
            inner_a = float(inner_lab[1])
            inner_b = float(inner_lab[2])
            hybrid_material = classify_material_hybrid(inner_h, inner_s, inner_v, inner_l, inner_a, inner_b)
            bronze_margin = float(hybrid_material["bronze_score"]) - float(hybrid_material["gold_score"])

            if (
                (not outlier_vote_for_veto)
                and (not kmeans_support_for_veto)
                and str(hybrid_material["label"]) == "bronze"
                and bronze_margin >= 0.18
            ):
                detector_type = "one-color-like"
                bronze_veto_applied = True

        if detector_type == "bi-metal-like":
            if bimetal_mode == "mean-color":
                bimetal_euro_label = label_bimetal_euro_from_mean_lab(inner_lab, border_lab)
            else:
                bimetal_euro_label = label_bimetal_euro_from_saturation(int(inner_hsv[1]), int(border_hsv[1]))
            final_label = bimetal_euro_label
            bronze_score = 0.0
            bronze_score_hsv = 0.0
            bronze_score_lab = 0.0
            bronze_score_hybrid = 0.0
            gold_score_hsv = 0.0
            gold_score_lab = 0.0
            gold_score_hybrid = 0.0
            material_label_hsv = "n/a"
            material_label_lab = "n/a"
            material_label_hybrid = "n/a"
            material_label = "n/a"
        else:
            bimetal_euro_label = "n/a"
            inner_h = int(inner_hsv[0])
            inner_s = int(inner_hsv[1])
            inner_v = int(inner_hsv[2])
            inner_l = float(inner_lab[0])
            inner_a = float(inner_lab[1])
            inner_b = float(inner_lab[2])

            bronze_score_hsv = bronze_score_from_inner_hsv(inner_h, inner_s, inner_v)
            gold_score_hsv = gold_score_from_inner_hsv(inner_h, inner_s, inner_v)
            bronze_score_lab = bronze_score_from_inner_lab(inner_l, inner_a, inner_b)
            gold_score_lab = gold_score_from_inner_lab(inner_l, inner_a, inner_b)
            hybrid_material = classify_material_hybrid(inner_h, inner_s, inner_v, inner_l, inner_a, inner_b)
            bronze_score_hybrid = float(hybrid_material["bronze_score"])
            gold_score_hybrid = float(hybrid_material["gold_score"])
            material_label_hsv = label_material_from_inner_hsv(inner_h, inner_s, inner_v)
            material_label_lab = label_material_from_inner_lab(inner_l, inner_a, inner_b)
            material_label_hybrid = str(hybrid_material["label"])
            if material_mode == "hsv":
                bronze_score = bronze_score_hsv
                material_label = material_label_hsv
            elif material_mode == "lab":
                bronze_score = bronze_score_lab
                material_label = material_label_lab
            else:
                bronze_score = bronze_score_hybrid
                material_label = material_label_hybrid

            if detector_type == "uncertain":
                final_label = "uncertain"
            elif material_label == "borderline":
                final_label = "one-color-like/borderline"
            else:
                final_label = f"one-color-like/{material_label}"

        full_bgr = cv2.cvtColor(np.uint8([[[*full_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        inner_bgr = cv2.cvtColor(np.uint8([[[*inner_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]
        border_bgr = cv2.cvtColor(np.uint8([[[*border_hsv]]]), cv2.COLOR_HSV2BGR)[0, 0]

        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), r, tuple(int(c) for c in border_bgr), -1)
            cv2.circle(output_bgr, (x, y), inner_r, tuple(int(c) for c in inner_bgr), -1)
        else:
            cv2.circle(output_bgr, (x, y), r, tuple(int(c) for c in full_bgr), -1)

        cv2.circle(output_bgr, (x, y), r, (0, 0, 0), 2)
        if detector_type == "bi-metal-like":
            cv2.circle(output_bgr, (x, y), inner_r, (0, 0, 0), 2)

        text = str(row["id"] + 1)
        if detector_type == "bi-metal-like":
            if bimetal_euro_label == "2-euro-like":
                short_label = "(2)"
            elif bimetal_euro_label == "1-euro-like":
                short_label = "(1)"
            else:
                short_label = "(?)"
        else:
            if material_label == "gold":
                short_label = "G"
            elif material_label == "bronze":
                short_label = "B"
            else:
                short_label = "?"

        text_fg = (0, 220, 255) if detector_type == "uncertain" else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_scale = 0.55
        lbl_scale = 0.45
        num_size, _ = cv2.getTextSize(text, font, num_scale, 1)
        lbl_size, _ = cv2.getTextSize(short_label, font, lbl_scale, 1)
        num_org = (int(x) - (num_size[0] // 2), int(y) - 2)
        lbl_org = (int(x) - (lbl_size[0] // 2), int(y) + 14)
        cv2.putText(output_bgr, text, num_org, font, num_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, text, num_org, font, num_scale, text_fg, 1, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_bgr, short_label, lbl_org, font, lbl_scale, text_fg, 1, cv2.LINE_AA)

        row["color_delta_threshold"] = float(outlier_threshold)
        row["abs_threshold_mid"] = float(abs_mid)
        row["abs_threshold_hi"] = float(abs_hi)
        row["abs_threshold_vhi"] = float(abs_vhi)
        row["decision_evidence"] = int(evidence)
        row["bronze_score"] = float(bronze_score)
        row["bronze_score_hsv"] = float(bronze_score_hsv)
        row["bronze_score_lab"] = float(bronze_score_lab)
        row["bronze_score_hybrid"] = float(bronze_score_hybrid)
        row["gold_score_hsv"] = float(gold_score_hsv)
        row["gold_score_lab"] = float(gold_score_lab)
        row["gold_score_hybrid"] = float(gold_score_hybrid)
        row["material_label_hsv"] = material_label_hsv
        row["material_label_lab"] = material_label_lab
        row["material_label_hybrid"] = material_label_hybrid
        row["material_label"] = material_label
        row["detector_type"] = detector_type
        row["bronze_veto"] = bool(bronze_veto_applied)
        row["bimetal_euro_label"] = bimetal_euro_label
        row["bimetal_mode"] = bimetal_mode
        row["material_mode"] = material_mode
        row["short_label"] = short_label
        row["label"] = final_label
        row["type"] = detector_type
        stats.append(row)

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), stats

@dataclass
class ValueEstimationResult:
    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int


@dataclass
class CoinValueEstimationOutput:
    split_rgb: np.ndarray
    split_stats: list[dict]
    value_labeled_rgb: np.ndarray
    predictions: dict[int, dict]
    scale_info: dict
    family_models: dict
    counts: dict[str, int]
    total_cents: int


def _coin_marker_token(index: int) -> str:
    """Convert zero-based index into Excel-like marker tokens: A, B, ..., Z, AA, AB..."""
    n = max(0, int(index))
    out: list[str] = []
    while True:
        n, rem = divmod(n, 26)
        out.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(out))


class ValueEstimator:
    EURO_DIAMETER_MM: dict[str, float] = {
        "1c": 16.25,
        "2c": 18.75,
        "5c": 21.25,
        "10c": 19.75,
        "20c": 22.25,
        "50c": 24.25,
        "1e": 23.25,
        "2e": 25.75,
    }

    FAMILY_TO_DENOMS: dict[str, list[str]] = {
        "bronze": ["1c", "2c", "5c"],
        "gold": ["10c", "20c", "50c"],
        "bimetal": ["1e", "2e"],
        "unknown": ["1c", "2c", "5c", "10c", "20c", "50c"],
    }

    DENOM_TEXT: dict[str, str] = {
        "1c": "1c",
        "2c": "2c",
        "5c": "5c",
        "10c": "10c",
        "20c": "20c",
        "50c": "50c",
        "1e": "1EUR",
        "2e": "2EUR",
    }

    DENOM_TO_CENTS: dict[str, int] = {
        "1c": 1,
        "2c": 2,
        "5c": 5,
        "10c": 10,
        "20c": 20,
        "50c": 50,
        "1e": 100,
        "2e": 200,
    }

    DENOM_PRINT_ORDER: list[str] = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

    @classmethod
    def estimate_from_stats(cls, rows: list[dict]) -> ValueEstimationResult:
        predictions, scale_info, family_models = cls._estimate_coin_values_from_stats(rows)
        counts, total_cents = cls.summarize_prediction_totals(predictions)
        return ValueEstimationResult(
            predictions=predictions,
            scale_info=scale_info,
            family_models=family_models,
            counts=counts,
            total_cents=total_cents,
        )

    @classmethod
    def coin_family_from_row(cls, row: dict) -> str:
        if row.get("type") == "bi-metal-like":
            return "bimetal"

        material = row.get("material_label")
        if material in ("bronze", "gold"):
            return material

        inner_hsv = row.get("inner_hsv")
        if isinstance(inner_hsv, (tuple, list)) and len(inner_hsv) >= 2:
            h = int(inner_hsv[0])
            s = int(inner_hsv[1])
            if s >= 45:
                if h <= 15:
                    return "bronze"
                if h >= 16:
                    return "gold"

        return "unknown"

    @classmethod
    def summarize_prediction_totals(cls, predictions: dict[int, dict]) -> tuple[dict[str, int], int]:
        counts = {d: 0 for d in cls.DENOM_PRINT_ORDER}
        total_cents = 0
        for pred in predictions.values():
            denom = pred.get("best_denom")
            if denom not in cls.DENOM_TO_CENTS:
                continue
            counts[denom] += 1
            total_cents += int(cls.DENOM_TO_CENTS[denom])
        return counts, int(total_cents)

    @classmethod
    def prob_string(cls, prob_map: dict[str, float]) -> str:
        ordered = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        return ", ".join(f"{cls.DENOM_TEXT[k]}={100.0 * v:.1f}%" for k, v in ordered)

    @classmethod
    def draw_coin_value_labels(
        cls,
        image_bgr: np.ndarray,
        rows: list[dict],
        predictions: dict[int, dict],
    ) -> np.ndarray:
        out = image_bgr.copy()
        family_colors = {
            "bronze": (40, 120, 230),
            "gold": (0, 200, 255),
            "bimetal": (255, 180, 40),
            "unknown": (180, 180, 180),
        }

        for row in rows:
            coin_id = int(row["id"])
            pred = predictions.get(coin_id)
            if pred is None:
                continue

            x = int(row["x"])
            y = int(row["y"])
            r = int(row["r"])
            family = pred["family"]
            color = family_colors.get(family, (180, 180, 180))

            cv2.circle(out, (x, y), r, color, 2)
            cv2.circle(out, (x, y), 2, (0, 0, 0), 3)
            marker = _coin_marker_token(coin_id)
            marker_radius = max(8, min(16, int(0.34 * max(1, r))))
            cv2.circle(out, (x, y), marker_radius, (15, 15, 15), -1)
            cv2.circle(out, (x, y), marker_radius, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52 if len(marker) == 1 else 0.42
            thickness = 1
            text_size, _ = cv2.getTextSize(marker, font, scale, thickness)
            tx = int(x) - (text_size[0] // 2)
            ty = int(y) + (text_size[1] // 2)
            cv2.putText(out, marker, (tx, ty), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, marker, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    @classmethod
    def _kmeans_1d_sorted(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        n = int(vals.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        k = int(max(1, min(k, n)))
        if k == 1:
            center = np.array([float(np.mean(vals[:, 0]))], dtype=np.float32)
            labels = np.zeros((n,), dtype=np.int32)
            return labels, center

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.02)
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(vals, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        centers = centers[:, 0]
        order = np.argsort(centers)
        centers_sorted = centers[order]
        remap = np.zeros((k,), dtype=np.int32)
        for rank, old_idx in enumerate(order):
            remap[int(old_idx)] = int(rank)

        labels_flat = labels[:, 0].astype(np.int32)
        labels_sorted = remap[labels_flat]
        return labels_sorted, centers_sorted.astype(np.float32)

    @classmethod
    def _kmeans_1d_with_stats(cls, values: list[float], k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        labels, centers = cls._kmeans_1d_sorted(values, k)
        vals = np.asarray(values, dtype=np.float64)

        if vals.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)
        if centers.size == 0:
            return labels, centers, 0.0, np.zeros((0,), dtype=np.int32)

        labels_i = labels.astype(np.int32)
        pred = centers[labels_i].astype(np.float64)
        inertia = float(np.sum((vals - pred) ** 2))
        counts = np.bincount(labels_i, minlength=int(centers.size)).astype(np.int32)
        return labels_i, centers.astype(np.float32), inertia, counts

    @classmethod
    def _choose_k_for_family(cls, values: list[float], max_k: int, family: str) -> tuple[int, dict]:
        vals = np.asarray(values, dtype=np.float64)
        n = int(vals.size)
        if n <= 1:
            return 1, {"reason": "n<=1", "evaluated": []}

        uniq = int(np.unique(np.round(vals, 2)).size)
        max_k = int(max(1, min(max_k, n, uniq)))
        if max_k <= 1:
            return 1, {"reason": "max_k<=1", "evaluated": []}

        min_count_abs = 1 if n < 6 else 2
        min_count_frac = 0.14 if family in ("gold", "bronze") else 0.10

        rows = []
        for k in range(1, max_k + 1):
            _, centers, inertia, counts = cls._kmeans_1d_with_stats(vals.tolist(), k)
            sigma2 = max(float(inertia) / max(n, 1), 1e-6)
            dof = max(1, 2 * k - 1)
            bic = float(n * np.log(sigma2) + dof * np.log(max(n, 2)))

            min_count = int(np.min(counts)) if counts.size > 0 else 0
            min_needed = max(int(min_count_abs), int(np.ceil(min_count_frac * n)))
            tiny = max(0, min_needed - min_count)
            if tiny > 0:
                bic += float(4.0 * tiny)

            rows.append(
                {
                    "k": int(k),
                    "bic": float(bic),
                    "inertia": float(inertia),
                    "min_count": int(min_count),
                    "counts": [int(c) for c in counts.tolist()],
                }
            )

        best = min(rows, key=lambda r: r["bic"])
        best_k = int(best["k"])
        if best_k > 1:
            prev = [r for r in rows if r["k"] == (best_k - 1)][0]
            rel_gain = float((prev["inertia"] - best["inertia"]) / (prev["inertia"] + 1e-9))
            min_gain = {"gold": 0.12, "bronze": 0.10, "bimetal": 0.08, "unknown": 0.14}.get(family, 0.10)
            if rel_gain < min_gain:
                best_k = int(best_k - 1)

        return best_k, {"reason": "bic+min_cluster", "evaluated": rows, "chosen_k": int(best_k)}

    @classmethod
    def _best_denom_subset_for_centers(
        cls,
        sorted_centers_px: np.ndarray,
        candidate_denoms: list[str],
        px_per_mm: float | None,
    ) -> list[str]:
        k = int(len(sorted_centers_px))
        if k <= 0:
            return []
        if k >= len(candidate_denoms):
            return candidate_denoms[:]

        centers = np.asarray(sorted_centers_px, dtype=np.float64)
        best_score = 1e18
        best: list[str] | None = None

        for comb in combinations(candidate_denoms, k):
            half_mm = np.array([0.5 * cls.EURO_DIAMETER_MM[d] for d in comb], dtype=np.float64)
            if px_per_mm is not None and px_per_mm > 0:
                expected_px = half_mm * float(px_per_mm)
                denom = np.mean(expected_px ** 2) + 1e-9
                score = float(np.mean((centers - expected_px) ** 2) / denom)
            else:
                c = centers / (np.median(centers) + 1e-9)
                e = half_mm / (np.median(half_mm) + 1e-9)
                score = float(np.mean((c - e) ** 2))
            if score < best_score:
                best_score = score
                best = list(comb)

        return best if best is not None else candidate_denoms[:k]

    @classmethod
    def _estimate_scale_from_all_radii_voting(cls, rows: list[dict]) -> tuple[float | None, dict]:
        scale_candidates: list[float] = []
        for row in rows:
            r_px = float(max(1, int(row.get("r", 1))))
            family = cls.coin_family_from_row(row)
            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            for denom in denom_pool:
                d_mm = cls.EURO_DIAMETER_MM[denom]
                scale_candidates.append((2.0 * r_px) / d_mm)

        if len(scale_candidates) == 0:
            return None, {"support": 0, "total": 0}

        arr = np.asarray(scale_candidates, dtype=np.float64)
        tol = 0.18
        diffs = np.abs(arr[:, None] - arr[None, :])
        support_counts = np.sum(diffs <= tol, axis=1)
        best_idx = int(np.argmax(support_counts))
        support_mask = diffs[best_idx] <= tol
        robust_scale = float(np.median(arr[support_mask]))
        return robust_scale, {"support": int(np.sum(support_mask)), "total": int(arr.size), "window_tol": float(tol)}

    @classmethod
    def _estimate_px_per_mm(cls, rows: list[dict]) -> tuple[float | None, dict]:
        ref_scales: list[float] = []
        for row in rows:
            if row.get("type") != "bi-metal-like":
                continue
            label = row.get("bimetal_euro_label", "")
            if label == "1-euro-like":
                denom = "1e"
            elif label == "2-euro-like":
                denom = "2e"
            else:
                continue
            r_px = float(max(1, int(row.get("r", 1))))
            ref_scales.append((2.0 * r_px) / cls.EURO_DIAMETER_MM[denom])

        fallback_scale, fallback_dbg = cls._estimate_scale_from_all_radii_voting(rows)
        if len(ref_scales) >= 2:
            return float(np.median(np.asarray(ref_scales, dtype=np.float64))), {
                "method": "bimetal_reference",
                "count": int(len(ref_scales)),
                "raw_scales": [float(x) for x in ref_scales],
            }
        if len(ref_scales) == 1:
            ref = float(ref_scales[0])
            info = {"method": "single_bimetal_reference", "count": 1, "raw_scales": [float(ref)]}
            if fallback_scale is not None:
                info["fallback_scale"] = float(fallback_scale)
                info.update(fallback_dbg)
            return ref, info
        if fallback_scale is not None:
            return float(fallback_scale), {"method": "all_radii_voting", **fallback_dbg}
        return None, {"method": "none", "count": 0}

    @classmethod
    def _build_family_radius_models(cls, rows: list[dict], px_per_mm: float | None) -> dict:
        models = {}
        for family, denom_pool in cls.FAMILY_TO_DENOMS.items():
            if family == "bimetal":
                continue
            fam_rows = [row for row in rows if cls.coin_family_from_row(row) == family]
            if len(fam_rows) == 0:
                continue

            radii = [float(row["r"]) for row in fam_rows]
            max_k = min(len(denom_pool), len(radii))
            k, k_dbg = cls._choose_k_for_family(radii, max_k=max_k, family=family)
            labels, centers, _, counts = cls._kmeans_1d_with_stats(radii, k)
            subset = cls._best_denom_subset_for_centers(centers, denom_pool, px_per_mm=px_per_mm)

            cluster_to_denom = {int(i): subset[i] for i in range(len(subset))}
            coin_to_cluster = {int(fam_rows[i]["id"]): int(labels[i]) for i in range(len(fam_rows))}
            models[family] = {
                "k": int(k),
                "centers_px": [float(c) for c in centers.tolist()],
                "cluster_counts": [int(c) for c in counts.tolist()],
                "subset_denoms": subset,
                "cluster_to_denom": cluster_to_denom,
                "coin_to_cluster": coin_to_cluster,
                "k_debug": k_dbg,
            }
        return models

    @classmethod
    def _normalize_scores(cls, score_map: dict[str, float]) -> dict[str, float]:
        vals = np.array([max(1e-12, float(v)) for v in score_map.values()], dtype=np.float64)
        total = float(np.sum(vals))
        if total <= 0:
            n = max(1, len(score_map))
            return {k: 1.0 / n for k in score_map.keys()}
        return {k: float(max(1e-12, score_map[k]) / total) for k in score_map.keys()}

    @classmethod
    def _estimate_coin_values_from_stats(cls, rows: list[dict]) -> tuple[dict[int, dict], dict, dict]:
        px_per_mm, scale_info = cls._estimate_px_per_mm(rows)
        family_models = cls._build_family_radius_models(rows, px_per_mm=px_per_mm)
        predictions: dict[int, dict] = {}

        for row in rows:
            coin_id = int(row["id"])
            family = cls.coin_family_from_row(row)

            if family == "bimetal":
                bm = row.get("bimetal_euro_label", "")
                if bm == "1-euro-like":
                    probs = {"1e": 1.0, "2e": 0.0}
                    best_denom = "1e"
                elif bm == "2-euro-like":
                    probs = {"1e": 0.0, "2e": 1.0}
                    best_denom = "2e"
                else:
                    probs = {"1e": 0.5, "2e": 0.5}
                    best_denom = "1e"
                predictions[coin_id] = {
                    "coin_id": int(coin_id),
                    "family": family,
                    "cluster_idx": None,
                    "estimated_diameter_mm": None,
                    "probs": probs,
                    "best_denom": best_denom,
                    "best_label": cls.DENOM_TEXT[best_denom],
                    "best_prob": float(probs[best_denom]),
                }
                continue

            denom_pool = cls.FAMILY_TO_DENOMS.get(family, cls.FAMILY_TO_DENOMS["unknown"])
            model = family_models.get(family, {})
            coin_to_cluster = model.get("coin_to_cluster", {})
            cluster_to_denom = model.get("cluster_to_denom", {})
            cluster_idx = coin_to_cluster.get(coin_id)

            centers_px = np.asarray(model.get("centers_px", []), dtype=np.float64)
            denom_to_center = {}
            for c_idx, d_name in cluster_to_denom.items():
                ci = int(c_idx)
                if 0 <= ci < int(centers_px.size):
                    denom_to_center[d_name] = float(centers_px[ci])

            r_px = float(row["r"])
            d_est_mm = None if px_per_mm is None else (2.0 * r_px / float(px_per_mm))

            raw_scores = {}
            for denom in denom_pool:
                d_ref = cls.EURO_DIAMETER_MM[denom]
                if d_est_mm is None:
                    size_like = 1.0
                else:
                    sigma_map = {"gold": 0.48, "bronze": 0.62, "unknown": 0.85}
                    sigma_mm = float(sigma_map.get(family, 0.80))
                    z = (float(d_est_mm) - float(d_ref)) / sigma_mm
                    size_like = float(np.exp(-0.5 * z * z))

                k_model = int(model.get("k", 0))
                full_k = int(len(denom_pool))
                if cluster_idx is None:
                    cluster_prior = 1.0
                else:
                    mapped = cluster_to_denom.get(int(cluster_idx))
                    if mapped == denom:
                        cluster_prior = 1.0
                    elif k_model >= full_k:
                        strict_map = {"gold": 0.08, "bronze": 0.10, "unknown": 0.30}
                        cluster_prior = float(strict_map.get(family, 0.25))
                    elif k_model >= 2:
                        partial_map = {"gold": 0.46, "bronze": 0.38, "unknown": 0.55}
                        cluster_prior = float(partial_map.get(family, 0.45))
                    else:
                        cluster_prior = 0.75

                if denom in denom_to_center:
                    c_px = float(denom_to_center[denom])
                    sigma_px = max(0.80, 0.055 * c_px)
                    zc = (r_px - c_px) / sigma_px
                    center_prior = float(np.exp(-0.5 * zc * zc))
                elif k_model >= full_k:
                    strict_miss = {"gold": 0.12, "bronze": 0.14, "unknown": 0.35}
                    center_prior = float(strict_miss.get(family, 0.25))
                elif k_model >= 2:
                    partial_miss = {"gold": 0.70, "bronze": 0.66, "unknown": 0.80}
                    center_prior = float(partial_miss.get(family, 0.65))
                else:
                    center_prior = 1.0

                shape_prior = 1.0
                raw_scores[denom] = float(max(1e-9, size_like * cluster_prior * center_prior * shape_prior))

            probs = cls._normalize_scores(raw_scores)
            best_denom = max(probs.keys(), key=lambda d: probs[d])
            predictions[coin_id] = {
                "coin_id": int(coin_id),
                "family": family,
                "cluster_idx": None if cluster_idx is None else int(cluster_idx),
                "estimated_diameter_mm": None if d_est_mm is None else float(d_est_mm),
                "probs": probs,
                "best_denom": best_denom,
                "best_label": cls.DENOM_TEXT[best_denom],
                "best_prob": float(probs[best_denom]),
            }

        scale_info = {**scale_info, "px_per_mm": None if px_per_mm is None else float(px_per_mm)}
        return predictions, scale_info, family_models


class CoinValueEstimator:
    def __init__(
        self,
        border_ratio: float = 0.24,
        sat_delta_threshold: float | None = None,
        bimetal_mode: str = "mean-color",
        material_mode: str = "hybrid",
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
        split_rgb, split_stats = self._analyzer.analyze(image_bgr, circles)

        if len(split_stats) > 0:
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


# -----------------------------------------------------------------------------
# Inlined from src/processor_circles.py
# -----------------------------------------------------------------------------





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
        self._clahe_enabled = bool(config.clahe_enabled)
        preset = config.get_preset(self._preset_name)

        self._preprocessing = ImagePreprocessing(
            clahe_enabled=self._clahe_enabled,
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_tile_grid_size=config.clahe_tile_grid_size,
            color_normalization_enabled=config.color_normalization_enabled,
            color_balance_strength=config.color_balance_strength,
            color_balance_max_shift_ab=config.color_balance_max_shift_ab,
            color_balance_max_shift_l=config.color_balance_max_shift_l,
            color_saturation_target=config.color_saturation_target,
            color_saturation_strength=config.color_saturation_strength,
            histogram_normalization_enabled=config.histogram_normalization_enabled,
            histogram_clip_limit=config.histogram_clip_limit,
            histogram_tile_grid_size=config.histogram_tile_grid_size,
            histogram_stretch_percentiles=config.histogram_stretch_percentiles,
            euro_reference_lab_bronze=config.euro_reference_lab_bronze,
            euro_reference_lab_gold=config.euro_reference_lab_gold,
            euro_reference_lab_nickel=config.euro_reference_lab_nickel,
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
        detection_gray = cv2.cvtColor(prep.image_bgr, cv2.COLOR_BGR2GRAY)
        ksize = _normalize_odd_ksize(self._cfg.gauss_ksize)
        if self._cfg.blur_mode == "gauss":
            detection_blurred = cv2.GaussianBlur(detection_gray, (ksize, ksize), self._cfg.gauss_sigma)
        else:
            detection_blurred = detection_gray if ksize <= 1 else cv2.medianBlur(detection_gray, ksize)

        detection = self._detector.detect(detection_gray, detection_blurred, prep.image_rgb)
        valuation_input_bgr = prep.hist_norm_bgr if self._cfg.histogram_normalization_enabled else prep.color_balanced_bgr
        valuation = self._value_estimator.estimate(valuation_input_bgr, detection.circles)

        value_counts = valuation.counts if len(valuation.counts) > 0 else {d: 0 for d in ValueEstimator.DENOM_PRINT_ORDER}
        steps = [
            PipelineStep("Original (Letterbox 640x480)", prep.image_rgb, "rgb"),
        ]
        if self._cfg.color_normalization_enabled:
            steps.append(PipelineStep("Color Normalized (Euro Etalon)", prep.color_balanced_rgb, "rgb"))
        if self._cfg.histogram_normalization_enabled:
            steps.append(PipelineStep("Histogram Normalized (L-channel)", prep.hist_norm_rgb, "rgb"))
        if self._clahe_enabled:
            steps.append(PipelineStep("CLAHE (L channel)", prep.clahe_rgb, "rgb"))
        steps.extend(
            [
                PipelineStep("Grayscale", prep.gray, "gray"),
                PipelineStep(self._preprocessing.blur_step_name, prep.blurred, "gray"),
                PipelineStep("Hough Circles", detection.circles_overlay, "rgb"),
                PipelineStep("Inner/Border Mean Color Analysis", valuation.split_rgb, "rgb"),
                PipelineStep("Coin Value Estimation", valuation.value_labeled_rgb, "rgb"),
            ]
        )
        debug_info = {
            "preset": self._preset_name,
            "plateau_debug": detection.sweep_debug,
            "sweep_results": detection.sweep_results,
            "clahe_enabled": self._clahe_enabled,
            "blur_mode": self._cfg.blur_mode,
            "color_normalization_enabled": self._cfg.color_normalization_enabled,
            "histogram_normalization_enabled": self._cfg.histogram_normalization_enabled,
            "color_norm_debug": prep.color_norm_debug,
            "hist_norm_debug": prep.hist_norm_debug,
            "valuation_input_stage": (
                "hist_norm_bgr" if self._cfg.histogram_normalization_enabled else "color_balanced_bgr"
            ),
            "split_stats": valuation.split_stats,
            "value_predictions": valuation.predictions,
            "value_scale_info": valuation.scale_info,
            "family_models": valuation.family_models,
            "value_counts": value_counts,
            "total_cents": int(valuation.total_cents),
        }
        if self._clahe_enabled:
            debug_info["clahe_bgr"] = prep.clahe_bgr

        return PipelineResult(
            source_path=source_path,
            steps=steps,
            circle_count=detection.circle_count,
            hough_params=detection.hough_params,
            debug_info=debug_info,
        )


# -----------------------------------------------------------------------------
# One-file runner / viewer
# -----------------------------------------------------------------------------

# Initialize the rich console globally for the script
console = Console()


def _is_non_interactive_backend(backend_name: str) -> bool:
    backend = str(backend_name).strip().lower()
    if backend.startswith("module://matplotlib_inline"):
        return True

    non_interactive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    return backend in non_interactive_backends


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
        }
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return repr(value)


def build_debug_dump_payload(
    result: PipelineResult,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> dict[str, Any]:
    steps = result.steps[-1:] if final_only else result.steps
    step_count = len(steps)
    clamped_step_index = 0
    step_name = "n/a"
    step_shape = None
    if step_count > 0:
        clamped_step_index = max(0, min(int(step_index), step_count - 1))
        step = steps[clamped_step_index]
        step_name = step.name
        step_shape = [int(dim) for dim in step.image.shape]

    info = result.debug_info if isinstance(result.debug_info, dict) else {}
    relative_path = str(info.get("relative_path", result.source_path.name))
    predictions = info.get("value_predictions", {})
    coin_rows: list[dict[str, Any]] = []
    if isinstance(predictions, dict):
        coin_ids: list[int] = []
        for raw_key in predictions.keys():
            try:
                coin_ids.append(int(raw_key))
            except (TypeError, ValueError):
                continue
        for coin_id in sorted(coin_ids):
            pred = predictions.get(coin_id, {})
            if pred == {} and coin_id in predictions:
                pred = predictions[coin_id]
            elif pred == {} and str(coin_id) in predictions:
                pred = predictions[str(coin_id)]
            if not isinstance(pred, dict):
                continue
            coin_rows.append(
                {
                    "marker": _coin_marker_token(coin_id),
                    "coin_id": int(coin_id),
                    "best_label": str(pred.get("best_label", "?")),
                    "best_denom": str(pred.get("best_denom", "?")),
                    "best_prob": float(pred.get("best_prob", 0.0)),
                    "family": str(pred.get("family", "unknown")),
                }
            )

    payload: dict[str, Any] = {
        "source_path": str(result.source_path),
        "relative_path": relative_path,
        "viewer": {
            "final_only": bool(final_only),
            "step_index": int(clamped_step_index),
            "step_count": int(step_count),
            "step_name": step_name,
            "step_image_shape": step_shape,
        },
        "metrics": {
            "status": str(info.get("status", "n/a")),
            "predicted_coin_count": int(info.get("predicted_coin_count", result.circle_count)),
            "true_coin_count": _to_serializable(info.get("true_coin_count")),
            "coin_diff": _to_serializable(info.get("coin_diff")),
            "predicted_value_cents": int(info.get("predicted_value_cents", info.get("total_cents", 0))),
            "true_value_cents": _to_serializable(info.get("true_value_cents")),
            "value_diff_cents": _to_serializable(info.get("value_diff_cents")),
        },
        "hough_params": _to_serializable(result.hough_params),
        "coin_predictions": coin_rows,
        "raw_debug_info": _to_serializable(info),
    }
    if panel_text is not None:
        payload["panel_text"] = panel_text
    return payload


def export_result_debug(
    result: PipelineResult,
    export_root: Path,
    step_index: int = 0,
    final_only: bool = False,
    panel_text: str | None = None,
) -> tuple[Path, Path]:
    payload = build_debug_dump_payload(
        result=result,
        step_index=step_index,
        final_only=final_only,
        panel_text=panel_text,
    )

    relative_path = Path(str(payload.get("relative_path", result.source_path.name)))
    out_subdir = Path(export_root) / relative_path.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    step_no = int(payload["viewer"]["step_index"]) + 1
    tag = "final" if final_only else f"s{step_no:02d}"
    json_path = out_subdir / f"{relative_path.stem}_debug_{tag}.json"
    text_path = out_subdir / f"{relative_path.stem}_debug_{tag}.txt"

    json_blob = json.dumps(payload, indent=2, ensure_ascii=True)
    json_path.write_text(json_blob, encoding="utf-8")

    panel_blob = panel_text if panel_text is not None else "(panel text not captured)"
    text_blob = (
        "DEBUG PANEL SNAPSHOT\n"
        "====================\n"
        f"{panel_blob}\n\n"
        "FULL DEBUG PAYLOAD (JSON)\n"
        "=========================\n"
        f"{json_blob}\n"
    )
    text_path.write_text(text_blob, encoding="utf-8")
    return json_path, text_path


class OneFileViewer:
    def __init__(
        self,
        results: list[PipelineResult],
        cols: int = 3,
        final_only: bool = False,
        debug_export_dir: Path | None = None,
    ):
        self._results = results
        self._cols = max(1, cols)
        self._final_only = bool(final_only)
        self._idx = 0
        self._step_idx = 0
        self._fig = None
        self._image_ax = None
        self._info_ax = None
        self._debug_export_dir = (
            Path(debug_export_dir) if debug_export_dir is not None else Path.cwd() / "debug_exports"
        )

    def show(self) -> None:
        if not self._results:
            console.print("[yellow][WARN] No pipeline results to display.[/yellow]")
            return

        backend = str(plt.get_backend())
        if _is_non_interactive_backend(backend):
            console.print(
                f"[yellow][WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                "viewer window cannot open.[/yellow]"
            )
            return
        if backend.strip().lower() == "webagg":
            console.print(
                "[cyan][INFO] Using WebAgg backend. Open the local URL printed by Matplotlib "
                "to view the interactive window in your browser.[/cyan]"
            )

        try:
            self._fig = plt.figure(figsize=(16, 9))
            grid = self._fig.add_gridspec(
                1,
                2,
                width_ratios=(4.8, 1.9),
                left=0.025,
                right=0.985,
                top=0.92,
                bottom=0.03,
                wspace=0.04,
            )
            self._image_ax = self._fig.add_subplot(grid[0, 0])
            self._info_ax = self._fig.add_subplot(grid[0, 1])
        except Exception as exc:
            console.print(
                f"[yellow][WARN] Unable to open interactive Matplotlib window using "
                f"'{plt.get_backend()}': {exc}[/yellow]"
            )
            return
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()
        plt.show()

    def _on_key(self, event) -> None:
        if event.key in ("right", "d", "n", " "):
            self._idx = (self._idx + 1) % len(self._results)
            self._render()
        elif event.key in ("left", "a", "p"):
            self._idx = (self._idx - 1) % len(self._results)
            self._render()
        elif event.key in ("up", "w"):
            self._step_idx += 1
            self._render()
        elif event.key in ("down", "s"):
            self._step_idx -= 1
            self._render()
        elif event.key in ("f",):
            self._final_only = not self._final_only
            self._step_idx = 0
            self._render()
        elif event.key in ("c", "x"):
            self._export_current_debug()
        elif event.key in ("q", "escape"):
            plt.close(self._fig)

    def _render(self) -> None:
        result = self._results[self._idx]
        steps = result.steps[-1:] if self._final_only else result.steps
        if not steps:
            return

        self._step_idx = max(0, min(self._step_idx, len(steps) - 1))
        step = steps[self._step_idx]
        total_cents = int(result.debug_info.get("total_cents", 0))
        title_step = f"step {self._step_idx + 1}/{len(steps)}"
        mode_label = "FINAL ONLY" if self._final_only else "FULL PIPELINE"
        self._fig.suptitle(
            f"[{self._idx + 1}/{len(self._results)}] {result.source_path.name} | "
            f"coins={result.circle_count} | value={_format_total_cents(total_cents)} | "
            f"{title_step} | {mode_label} | "
            "img: right/left | step: up/down | toggle-final: f | export: c/x | quit: q/esc",
            fontsize=11,
        )

        self._image_ax.clear()
        self._image_ax.axis("off")
        if step.cmap == "gray":
            self._image_ax.imshow(step.image, cmap="gray")
        else:
            self._image_ax.imshow(step.image)
        self._image_ax.set_title(step.name, fontsize=12, pad=8)

        panel_text = self._build_info_panel_text(result, step, len(steps))
        self._info_ax.clear()
        self._info_ax.axis("off")
        self._info_ax.text(
            0.03,
            0.98,
            panel_text,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            linespacing=1.35,
            color="white",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#111827",
                "edgecolor": "#374151",
                "linewidth": 1.2,
                "alpha": 0.97,
            },
            transform=self._info_ax.transAxes,
        )
        self._fig.canvas.draw_idle()

    def _export_current_debug(self) -> None:
        if not self._results:
            return
        result = self._results[self._idx]
        steps = result.steps[-1:] if self._final_only else result.steps
        if not steps:
            console.print("[yellow][WARN] No pipeline steps available to export debug info.[/yellow]")
            return
        self._step_idx = max(0, min(self._step_idx, len(steps) - 1))
        step = steps[self._step_idx]
        panel_text = self._build_info_panel_text(result, step, len(steps))
        try:
            json_path, text_path = export_result_debug(
                result=result,
                export_root=self._debug_export_dir,
                step_index=self._step_idx,
                final_only=self._final_only,
                panel_text=panel_text,
            )
        except Exception as exc:
            console.print(f"[yellow][WARN] Failed to export debug info: {exc}[/yellow]")
            return

        console.print(f"[green][INFO] Debug exported:[/green] {json_path}")
        console.print(f"[green][INFO] Text snapshot:[/green] {text_path}")

    def _build_info_panel_text(self, result: PipelineResult, step: PipelineStep, step_count: int) -> str:
        info = result.debug_info

        relative_path = str(info.get("relative_path", result.source_path.name))
        group_name = str(info.get("group", "n/a"))
        status = str(info.get("status", "n/a"))

        pred_coins = int(info.get("predicted_coin_count", result.circle_count))
        true_coins = info.get("true_coin_count")
        coin_diff = info.get("coin_diff")

        pred_value = int(info.get("predicted_value_cents", info.get("total_cents", 0)))
        true_value = info.get("true_value_cents")
        value_diff = info.get("value_diff_cents")

        has_gt = bool(info.get("has_ground_truth", False))
        # Keep explicit formatting here for readability in the side panel.
        if has_gt and true_coins is not None and coin_diff is not None:
            coins_line = f"{pred_coins} vs {int(true_coins)} ({int(coin_diff):+d})"
        else:
            coins_line = f"{pred_coins} vs n/a"

        if has_gt and true_value is not None and value_diff is not None:
            values_line = (
                f"{_format_total_cents(pred_value)} vs "
                f"{_format_total_cents(int(true_value))} ({_format_signed_cents(int(value_diff))})"
            )
        else:
            values_line = f"{_format_total_cents(pred_value)} vs n/a"

        counts = info.get("value_counts", {})
        if isinstance(counts, dict):
            breakdown = ", ".join(
                f"{ValueEstimator.DENOM_TEXT[d]}:{int(counts.get(d, 0))}"
                for d in ValueEstimator.DENOM_PRINT_ORDER
                if int(counts.get(d, 0)) > 0
            )
        else:
            breakdown = ""
        breakdown = breakdown if breakdown else "none"

        split_stats = info.get("split_stats", [])
        split_by_id: dict[int, dict] = {}
        if isinstance(split_stats, list):
            for row in split_stats:
                if isinstance(row, dict) and "id" in row:
                    split_by_id[int(row["id"])] = row

        prediction_lines: list[str] = []
        predictions = info.get("value_predictions", {})
        if isinstance(predictions, dict):
            coin_ids: list[int] = []
            for raw_key in predictions.keys():
                try:
                    coin_ids.append(int(raw_key))
                except (TypeError, ValueError):
                    continue
            sorted_ids = sorted(coin_ids)
            for coin_id in sorted_ids[:14]:
                pred = predictions.get(coin_id, {})
                if pred == {} and coin_id in predictions:
                    pred = predictions[coin_id]
                elif pred == {} and str(coin_id) in predictions:
                    pred = predictions[str(coin_id)]
                if not isinstance(pred, dict):
                    continue
                marker = _coin_marker_token(coin_id)
                best_label = str(pred.get("best_label", "?"))
                best_prob = int(round(100.0 * float(pred.get("best_prob", 0.0))))
                family = str(pred.get("family", "unknown"))
                split_row = split_by_id.get(coin_id, {})
                visual_type = str(split_row.get("short_label", "?"))
                prediction_lines.append(
                    f"  {marker:>2} -> {best_label:<4} {best_prob:>3}%  fam={family:<7} vis={visual_type}"
                )
            if len(sorted_ids) > 14:
                prediction_lines.append(f"  ... ({len(sorted_ids) - 14} more)")

        coin_map_block = "\n".join(prediction_lines) if prediction_lines else "  none"

        hough = result.hough_params
        hough_line = (
            f"dp={hough.get('dp', 'n/a')}  minDist={hough.get('minDist', 'n/a')}\n"
            f"param1={hough.get('param1', 'n/a')}  param2={hough.get('param2', 'n/a')}\n"
            f"minR={hough.get('minRadius', 'n/a')}  maxR={hough.get('maxRadius', 'n/a')}"
        )

        color_norm_debug = info.get("color_norm_debug", {})
        hist_norm_debug = info.get("hist_norm_debug", {})
        if isinstance(color_norm_debug, dict) and bool(color_norm_debug.get("enabled", False)):
            color_norm_line = (
                f"sat_scale={float(color_norm_debug.get('sat_scale', 1.0)):.3f}  "
                f"shift_a={float(color_norm_debug.get('shift_a', 0.0)):+.2f}  "
                f"shift_b={float(color_norm_debug.get('shift_b', 0.0)):+.2f}"
            )
        else:
            color_norm_line = "disabled"

        if isinstance(hist_norm_debug, dict) and bool(hist_norm_debug.get("enabled", False)):
            p = hist_norm_debug.get("stretch_percentiles", [1.0, 99.0])
            lo = float(hist_norm_debug.get("l_percentile_low", 0.0))
            hi = float(hist_norm_debug.get("l_percentile_high", 255.0))
            hist_norm_line = f"p=[{float(p[0]):.1f},{float(p[1]):.1f}]  L=[{lo:.1f},{hi:.1f}]"
        else:
            hist_norm_line = "disabled"

        text = (
            "DEBUG PANEL\n"
            "================================\n"
            f"file        : {relative_path}\n"
            f"group       : {group_name}\n"
            f"status      : {status}\n"
            f"backend     : {plt.get_backend()}\n"
            "\n"
            f"step        : {self._step_idx + 1}/{step_count}\n"
            f"step name   : {step.name}\n"
            f"image size  : {step.image.shape[1]} x {step.image.shape[0]}\n"
            "\n"
            f"coins (P/T) : {coins_line}\n"
            f"value (P/T) : {values_line}\n"
            "\n"
            f"value split : {breakdown}\n"
            "\n"
            "coin map (matches in-image markers)\n"
            f"{coin_map_block}\n"
            "\n"
            "hough params\n"
            f"{hough_line}\n"
            "\n"
            "normalization\n"
            f"  color_norm : {color_norm_line}\n"
            f"  hist_norm  : {hist_norm_line}\n"
            "\n"
            "keys\n"
            "  left/right : previous/next image\n"
            "  up/down    : previous/next step\n"
            "  f          : toggle final-only/full\n"
            "  c or x     : export full debug to files\n"
            "  q or esc   : quit\n"
        )
        return text


class OneFileRunner:
    def run(self) -> None:
        args = self._build_parser().parse_args()
        cols = max(1, args.cols)
        eval_groups = _parse_eval_groups(args.eval_groups)
        debug_export_dir = None
        if args.debug_export_dir:
            debug_export_dir = Path(args.debug_export_dir).expanduser().resolve()

        console.print(f"[cyan][INFO] Matplotlib backend:[/cyan] {plt.get_backend()}")
        config = PipelineConfig()
        if args.dataset_dir is not None:
            config = replace(config, dataset_dir=Path(args.dataset_dir).expanduser().resolve())
        if args.color_norm is not None:
            config = replace(config, color_normalization_enabled=bool(args.color_norm))
        if args.hist_norm is not None:
            config = replace(config, histogram_normalization_enabled=bool(args.hist_norm))

        preset_name = args.preset or config.active_preset
        if preset_name not in HOUGH_PRESETS:
            available = ", ".join(sorted(HOUGH_PRESETS))
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

        dataset = ImageDataset(config.dataset_dir, config.valid_extensions)
        images = dataset.list_images(limit=args.limit)
        if not images:
            console.print(f"[yellow][WARN] No images found under: {config.dataset_dir}[/yellow]")
            return

        processor = CirclePipelineProcessor(config, preset_name=preset_name)
        evaluator = Evaluation()
        ground_truth = GroundTruthRepository()
        results: list[PipelineResult] = []

        console.print(f"[cyan][INFO] Processing {len(images)} image(s) from:[/cyan] {config.dataset_dir}")
        if eval_groups is None:
            console.print("[cyan][INFO] Evaluation groups:[/cyan] all")
        else:
            console.print(f"[cyan][INFO] Evaluation groups:[/cyan] {', '.join(sorted(eval_groups))}")
        console.print(
            f"[cyan][INFO] Color norm:[/cyan] {config.color_normalization_enabled} | "
            f"[cyan][INFO] Hist norm:[/cyan] {config.histogram_normalization_enabled}"
        )
        print()

        # Set up the main results table
        main_table = Table(show_header=True, header_style="bold magenta", expand=True)
        main_table.add_column("FILE", style="dim", width=25)
        main_table.add_column("GROUP", justify="center")
        main_table.add_column("C_PRED", justify="right")
        main_table.add_column("C_TRUE", justify="right")
        main_table.add_column("C_DIFF", justify="right")
        main_table.add_column("V_PRED", justify="right", style="cyan")
        main_table.add_column("V_TRUE", justify="right", style="cyan")
        main_table.add_column("V_DIFF", justify="right")
        main_table.add_column("STATUS", justify="center", style="bold")

        # Process images with a spinner so the terminal doesn't look frozen
        with console.status("[bold green]Processing images and calculating metrics...") as status:
            for item in images:
                try:
                    result = processor.process_path(item.path)
                except Exception as exc:
                    console.print(f"[bold red][ERR ][/bold red] {item.relative_path}: {exc}")
                    continue

                results.append(result)
                group_name = _group_from_relative_path(item.relative_path)
                pred_value_cents = int(result.debug_info.get("total_cents", 0))
                result.debug_info.update(
                    {
                        "relative_path": str(item.relative_path),
                        "group": group_name,
                        "predicted_coin_count": int(result.circle_count),
                        "predicted_value_cents": pred_value_cents,
                        "has_ground_truth": False,
                        "true_coin_count": None,
                        "true_value_cents": None,
                        "coin_diff": None,
                        "value_diff_cents": None,
                        "status": "PREDICTED_ONLY",
                    }
                )

                # Handling Filtered Groups
                if eval_groups is not None and group_name not in eval_groups:
                    evaluator.add_filtered_group()
                    result.debug_info["status"] = "SKIP_GROUP"
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_GROUP[/yellow]"
                    )
                    if args.save_dir:
                        self._save_result(result, item.relative_path, Path(args.save_dir), cols, config.viewer_final_only)
                    if debug_export_dir is not None:
                        try:
                            export_result_debug(
                                result=result,
                                export_root=debug_export_dir,
                                step_index=max(0, len(result.steps) - 1),
                                final_only=True,
                                panel_text=None,
                            )
                        except Exception as exc:
                            console.print(
                                f"[yellow][WARN] Could not auto-export debug for {item.relative_path}: {exc}[/yellow]"
                            )
                    continue

                gt_entry = ground_truth.find(item.relative_path.name, group_name)
                
                # Handling Missing Ground Truth
                if gt_entry is None:
                    evaluator.add_missing_ground_truth()
                    result.debug_info["status"] = "SKIP_NO_GT"
                    main_table.add_row(
                        str(item.relative_path), group_name, "-", "-", "-", "-", "-", "-", "[yellow]SKIP_NO_GT[/yellow]"
                    )
                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")
                
                # Normal Evaluation Process
                else:
                    eval_item = evaluator.add_match(
                        relative_path=item.relative_path,
                        group=group_name,
                        predicted=result.circle_count,
                        ground_truth=gt_entry,
                        predicted_value_cents=pred_value_cents,
                    )
                    
                    status_col = "[green]OK[/green]" if eval_item.is_correct else "[red]ERR[/red]"
                    
                    c_diff_val = int(eval_item.diff)
                    c_diff_txt = str(c_diff_val) if c_diff_val == 0 else f"[red]{c_diff_val}[/red]"

                    pred_value_txt = _format_total_cents(int(eval_item.predicted_value_cents))
                    
                    if eval_item.expected_value_cents is None:
                        true_value_txt = "n/a"
                        diff_txt = "n/a"
                    else:
                        true_value_txt = _format_total_cents(int(eval_item.expected_value_cents))
                        v_diff_val = int(eval_item.value_diff_cents or 0)
                        raw_diff_str = _format_signed_cents(v_diff_val)
                        diff_txt = raw_diff_str if v_diff_val == 0 else f"[red]{raw_diff_str}[/red]"

                    result.debug_info.update(
                        {
                            "has_ground_truth": True,
                            "true_coin_count": int(eval_item.expected),
                            "true_value_cents": (
                                None
                                if eval_item.expected_value_cents is None
                                else int(eval_item.expected_value_cents)
                            ),
                            "coin_diff": int(eval_item.diff),
                            "value_diff_cents": (
                                None if eval_item.value_diff_cents is None else int(eval_item.value_diff_cents)
                            ),
                            "status": "OK" if eval_item.is_correct else "ERR",
                        }
                    )

                    # Add main row
                    main_table.add_row(
                        str(item.relative_path), group_name, str(eval_item.predicted), str(eval_item.expected), 
                        c_diff_txt, pred_value_txt, true_value_txt, diff_txt, status_col
                    )
                    
                    # Add breakdown underneath
                    breakdown_txt = self._get_value_breakdown_str(result)
                    main_table.add_row("", "", "", "", "", Text(breakdown_txt, style="dim italic"), "", "", "")

                if args.save_dir:
                    self._save_result(result, item.relative_path, Path(args.save_dir), cols, config.viewer_final_only)

                if debug_export_dir is not None:
                    try:
                        export_result_debug(
                            result=result,
                            export_root=debug_export_dir,
                            step_index=max(0, len(result.steps) - 1),
                            final_only=True,
                            panel_text=None,
                        )
                    except Exception as exc:
                        console.print(
                            f"[yellow][WARN] Could not auto-export debug for {item.relative_path}: {exc}[/yellow]"
                        )

        # Print the fully assembled table
        console.print(main_table)

        summary = evaluator.summary()
        by_group = evaluator.summary_by_group()
        value_tolerance_cents = int(summary.get("value_tolerance_cents", 100))
        value_tolerance_label = _format_total_cents(value_tolerance_cents)
        
        # Print Summary Metrics
        console.print("\n[bold cyan]--- OVERALL METRICS ---[/bold cyan]")
        console.print(f"[INFO] Completed detections: [bold]{len(results)}/{len(images)}[/bold]")
        console.print(f"[INFO] Evaluated (found in GT): [bold]{int(summary['evaluated'])}[/bold]")
        console.print(f"[INFO] Skipped (filtered group): {evaluator.skipped_filtered_group}")
        console.print(f"[INFO] Skipped (missing GT): {evaluator.skipped_missing_ground_truth}")
        
        console.print("\n[bold]Coin Metrics:[/bold]")
        console.print(
            f"  accuracy=[green]{float(summary['coin_accuracy']):.2f}%[/green] | "
            f"mae={float(summary['coin_mae']):.2f} coins/image | "
            f"correct={int(summary['coin_correct'])}/{int(summary['evaluated'])}"
        )
        
        console.print("[bold]Value Metrics:[/bold]")
        console.print(
            f"  accuracy(<= {value_tolerance_label})=[green]{float(summary['value_accuracy']):.2f}%[/green] | "
            f"exact={float(summary.get('value_accuracy_exact', 0.0)):.2f}% | "
            f"mae={float(summary['value_mae_eur']):.2f} EUR/image "
            f"({float(summary['value_mae_cents']):.1f} cents) | "
            f"correct={int(summary['value_correct'])}/{int(summary['value_evaluated'])} | "
            f"quality={float(summary.get('value_error_score', 0.0)):.2f}"
        )
        
        console.print("[bold]Combined Score:[/bold]")
        console.print(
            f"  coin_score={float(summary['coin_score']):.2f} | "
            f"value_score={_fmt_optional_score(summary.get('value_score'))} | "
            f"general_score=[bold magenta]{float(summary['general_score']):.2f}[/bold magenta]\n"
        )

        # Print Group Summary Table
        if by_group:
            group_table = Table(show_header=True, header_style="bold cyan", title="Summary By Group")
            group_table.add_column("GROUP")
            group_table.add_column("EVAL", justify="right")
            group_table.add_column("COIN ACC", justify="right")
            group_table.add_column("COIN MAE", justify="right")
            group_table.add_column(f"VAL ACC <= {value_tolerance_label}", justify="right")
            group_table.add_column("VAL MAE (EUR)", justify="right")
            group_table.add_column("GENERAL", justify="right", style="bold magenta")

            for group in sorted(by_group):
                row = by_group[group]
                group_table.add_row(
                    group,
                    str(int(row['evaluated'])),
                    f"{float(row['coin_accuracy']):.2f}%",
                    f"{float(row['coin_mae']):.2f}",
                    f"{float(row['value_accuracy']):.2f}%",
                    f"{float(row['value_mae_eur']):.2f}",
                    f"{float(row['general_score']):.2f}"
                )
            console.print(group_table)
            print()

        if args.no_view:
            return
        OneFileViewer(
            results,
            cols=cols,
            final_only=config.viewer_final_only,
            debug_export_dir=debug_export_dir,
        ).show()

    @staticmethod
    def _save_result(
        result: PipelineResult,
        relative_path: Path,
        save_dir: Path,
        cols: int,
        final_only: bool,
    ) -> None:
        out_subdir = save_dir / relative_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / f"{relative_path.stem}_pipeline.png"
        save_pipeline_figure(result, out_file, cols=cols, final_only=final_only)

    @staticmethod
    def _get_value_breakdown_str(result: PipelineResult) -> str:
        """Helper to return the formatted breakdown string instead of printing directly."""
        total_cents = int(result.debug_info.get("total_cents", 0))
        counts = result.debug_info.get("value_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        non_zero = [
            f"{ValueEstimator.DENOM_TEXT[d]}:{int(counts.get(d, 0))}"
            for d in ValueEstimator.DENOM_PRINT_ORDER
            if int(counts.get(d, 0)) > 0
        ]
        detail = ", ".join(non_zero) if non_zero else "none"
        return f"VALUE      {_format_total_cents(total_cents):<12} | {detail}"

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "One-file script version of the notebook pipeline with project-style dataset loading and evaluation."
            )
        )
        parser.add_argument(
            "--dataset-dir",
            type=str,
            default=None,
            help="Dataset root folder (default: data/images).",
        )
        parser.add_argument(
            "--preset",
            type=str,
            default=None,
            help=f"Hough preset name (default: {PipelineConfig().active_preset}).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of images for quick debugging.",
        )
        parser.add_argument(
            "--cols",
            type=int,
            default=3,
            help="Number of columns in pipeline grid view.",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default=None,
            help="Optional output folder to save per-image pipeline grids.",
        )
        parser.add_argument(
            "--no-view",
            action="store_true",
            help="Process/evaluate without opening interactive viewer.",
        )
        parser.add_argument(
            "--eval-groups",
            nargs="*",
            default=None,
            help="Evaluate only these groups (e.g. --eval-groups gp1 gp2 or --eval-groups gp1,gp2).",
        )
        parser.add_argument(
            "--color-norm",
            dest="color_norm",
            action="store_true",
            default=None,
            help="Enable euro-reference color normalization before value estimation.",
        )
        parser.add_argument(
            "--no-color-norm",
            dest="color_norm",
            action="store_false",
            help="Disable euro-reference color normalization.",
        )
        parser.add_argument(
            "--hist-norm",
            dest="hist_norm",
            action="store_true",
            default=None,
            help="Enable luminance histogram normalization (CLAHE + percentile stretch).",
        )
        parser.add_argument(
            "--no-hist-norm",
            dest="hist_norm",
            action="store_false",
            help="Disable luminance histogram normalization.",
        )
        parser.add_argument(
            "--debug-export-dir",
            type=str,
            default=None,
            help=(
                "Optional folder to auto-export per-image debug dumps (.json/.txt). "
                "Inside viewer, press c/x to export the currently displayed debug info."
            ),
        )
        return parser


def save_pipeline_figure(
    result: PipelineResult,
    output_path: Path,
    cols: int = 3,
    final_only: bool = False,
) -> None:
    cols = max(1, int(cols))
    rows = 1 if final_only else max(1, ceil(max(1, len(result.steps)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows))
    axes_matrix = _normalize_axes(axes, rows, cols)
    steps = result.steps[-1:] if final_only else result.steps

    for flat_idx in range(rows * cols):
        row, col = divmod(flat_idx, cols)
        ax = axes_matrix[row, col]
        ax.axis("off")
        if flat_idx >= len(steps):
            continue
        step = steps[flat_idx]
        if step.cmap == "gray":
            ax.imshow(step.image, cmap="gray")
        else:
            ax.imshow(step.image)
        ax.set_title(step.name if not final_only else f"Final: {step.name}", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _normalize_axes(axes, rows: int, cols: int) -> np.ndarray:
    if rows == 1 and cols == 1:
        return np.array([[axes]])
    if rows == 1:
        return np.array([axes])
    if cols == 1:
        return np.array([[ax] for ax in axes])
    return axes


def _group_from_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) <= 1:
        return ""
    return normalize_group_name(parts[0])


def _parse_eval_groups(raw_groups: list[str] | None) -> set[str] | None:
    if raw_groups is None:
        return None
    groups: set[str] = set()
    for token in raw_groups:
        for chunk in token.split(","):
            group = normalize_group_name(chunk.strip())
            if group:
                groups.add(group)
    return groups if groups else None


def _format_total_cents(total_cents: int) -> str:
    euros = total_cents // 100
    cents = total_cents % 100
    return f"{euros} EUR {cents:02d} c"


def _format_signed_cents(diff_cents: int) -> str:
    sign = "+" if diff_cents >= 0 else "-"
    euros_abs = abs(int(diff_cents)) // 100
    cents_abs = abs(int(diff_cents)) % 100
    return f"{sign}{euros_abs} EUR {cents_abs:02d} c"


def _fmt_optional_score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    OneFileRunner().run()
