from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable parameters for quality-gated classical OpenCV coin pipeline (v2)."""

    TARGET_WIDTH: int = 800

    # Step 0: Scene quality gate.
    # - off: no gate, no warnings
    # - warn: process image and emit warnings
    # - reject: skip counting/value for poor-quality captures
    SCENE_QUALITY_MODE: str = "warn"
    SCENE_BORDER_FRACTION: float = 0.08
    MAX_BORDER_STD: float = 24.0
    MAX_BORDER_EDGE_DENSITY: float = 0.085
    MIN_LAPLACIAN_VAR: float = 30.0
    EXPOSURE_DARK_THRESHOLD: int = 8
    EXPOSURE_BRIGHT_THRESHOLD: int = 247
    MAX_DARK_RATIO: float = 0.22
    MAX_BRIGHT_RATIO: float = 0.22
    SPECULAR_LOW_SAT_THRESHOLD: int = 38
    MAX_SPECULAR_RATIO: float = 0.015

    # Step 1: Illumination normalization.
    CLAHE_CLIP_LIMIT: float = 2.2
    CLAHE_TILE_GRID: int = 8
    DENOISE_METHOD: str = "median"
    BILATERAL_DIAMETER: int = 7
    BILATERAL_SIGMA_COLOR: int = 55
    BILATERAL_SIGMA_SPACE: int = 55
    MEDIAN_BLUR_KERNEL_SIZE: int = 5
    HOUGH_PREP_BLUR_KERNEL_SIZE: int = 15

    # Step 2: Foreground mask from border background color model.
    BG_DIST_WEIGHT_L: float = 0.35
    BG_DIST_WEIGHT_A: float = 1.0
    BG_DIST_WEIGHT_B: float = 1.0
    MASK_OTSU_OFFSET: int = -5
    MASK_MIN_DISTANCE_THRESHOLD: int = 18
    MASK_OPEN_KERNEL: int = 3
    MASK_CLOSE_KERNEL: int = 7
    MASK_ERODE_ITERS: int = 1
    MASK_DILATE_ITERS: int = 1
    MASK_MIN_COMPONENT_AREA: int = 300
    MASK_MAX_COMPONENT_AREA_RATIO: float = 0.26

    # Step 3: Marker-based watershed split for touching coins.
    WATERSHED_OPEN_KERNEL: int = 3
    WATERSHED_CLOSE_KERNEL: int = 5
    WATERSHED_DT_FG_RATIO: float = 0.52
    WATERSHED_MIN_SEED_AREA: int = 140

    # Step 4: Geometric cleanup (contour + circle/ellipse).
    GEOM_MIN_RADIUS: int = 9
    GEOM_MAX_RADIUS: int = 180
    GEOM_MIN_CIRCULARITY: float = 0.68
    GEOM_MAX_ASPECT_RATIO: float = 1.30
    GEOM_MIN_FILL_RATIO: float = 0.58
    GEOM_MAX_FILL_RATIO: float = 1.18
    GEOM_MIN_AREA: int = 250
    GEOM_MAX_AREA_RATIO: float = 0.20
    GEOM_DUPLICATE_CENTER_SCALE: float = 0.42
    GEOM_DUPLICATE_RADIUS_SCALE: float = 0.36
    EDGE_STACK_ASPECT_RATIO: float = 1.85

    # Optional Hough fallback (also used by interactive slider browser).
    HOUGH_FALLBACK_ENABLED: bool = False
    HOUGH_DP: float = 1.2
    HOUGH_MIN_DIST: int = 70
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 45
    HOUGH_MIN_RADIUS: int = 10
    HOUGH_MAX_RADIUS: int = 150
    HOUGH_MAX_ADDED_CIRCLES: int = 4
    HOUGH_REQUIRED_ANGULAR_COVERAGE: float = 0.20
    HOUGH_REQUIRED_CONTRAST: float = 0.040

    # Hough ensemble parameters (compatible with `src.processor_circles.CircleDetector`).
    HOUGH_LOOSE_DP_SCALE: float = 1.0
    HOUGH_LOOSE_MIN_DIST_SCALE: float = 0.90
    HOUGH_LOOSE_PARAM2_SCALE: float = 0.85
    CONTOUR_MIN_CIRCULARITY: float = 0.84
    CONTOUR_MIN_RADIUS_SCALE: float = 0.85
    CONTOUR_MAX_RADIUS_SCALE: float = 1.20
    CONTOUR_MIN_FILL_RATIO: float = 0.72
    MERGE_CENTER_DIST_SCALE: float = 0.50
    MERGE_RADIUS_DIFF_SCALE: float = 0.50
    CIRCLE_MIN_ANGULAR_COVERAGE: float = 0.20
    CIRCLE_MIN_CONTRAST: float = 0.055
    CIRCLE_MIN_SUPPORT_SCORE: float = 0.34
    LOOSE_MIN_SUPPORT_SCORE: float = 0.62

    # Hybrid arbitration: when to trust mask/watershed over Hough ensemble.
    HYBRID_POLICY_ENABLED: bool = True
    HYBRID_MASK_OVER_HOUGH_MIN_GAP: int = 15
    HYBRID_MASK_OVER_HOUGH_MIN_HOUGH_COUNT: int = 20
    HYBRID_MASK_OVER_HOUGH_MIN_MASK_COUNT: int = 2
    HYBRID_MASK_OVER_HOUGH_MAX_MASK_COUNT: int = 3
    HYBRID_MASK_OVER_HOUGH_MIN_BORDER_STD: float = 35.0
    HYBRID_MASK_OVER_HOUGH_MIN_BORDER_EDGE: float = 0.06
    HYBRID_MASK_ZERO_ALLOWED_MIN_EDGE_DENSITY: float = 0.22

    # Adaptive strict/relaxed profile.
    ADAPTIVE_PROFILE_ENABLED: bool = True
    ADAPTIVE_TEXTURE_EDGE_DENSITY: float = 0.095
    ADAPTIVE_STRICT_MASK_OFFSET: int = 10
    ADAPTIVE_RELAXED_MASK_OFFSET: int = -4
    ADAPTIVE_STRICT_PARAM2_SCALE: float = 1.00
    ADAPTIVE_STRICT_MIN_DIST_SCALE: float = 1.00
    ADAPTIVE_RELAXED_PARAM2_SCALE: float = 0.95
    ADAPTIVE_RELAXED_MIN_DIST_SCALE: float = 0.95

    # Step 5: Denomination acceptance threshold.
    MAX_LABEL_REL_ERROR: float = 0.12

    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for batch evaluation and interactive visualization."""

    IMAGE_DIRECTORY: str = field(default_factory=lambda: _find_image_directory())
    BROWSE_TUNE: bool = True
    SAVE_STEPS: bool = False
    OUT_DIR: str = "./pipeline_v2_viz"


def _find_image_directory() -> str:
    """Locate `data/images` automatically from current source location."""
    current = Path(__file__).resolve()
    project_name = "image_projet_money"

    for parent in current.parents:
        data_images = parent / "data" / "images"
        if parent.name.lower() == project_name and data_images.exists():
            return str(data_images)

    for parent in current.parents:
        data_images = parent / "data" / "images"
        if data_images.exists():
            return str(data_images)

    return str((current.parent / ".." / "data" / "images").resolve())
