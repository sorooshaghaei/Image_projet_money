from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable parameters for circle detection and post-filtering."""

    TARGET_WIDTH: int = 800
    BLUR_KERNEL_SIZE: int = 15

    HOUGH_DP: float = 1.2
    HOUGH_MIN_DIST: int = 70
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 45
    HOUGH_MIN_RADIUS: int = 10
    HOUGH_MAX_RADIUS: int = 150
    HOUGH_LOOSE_DP_SCALE: float = 1.0
    HOUGH_LOOSE_MIN_DIST_SCALE: float = 0.9
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
    # Maximum relative diameter-fit error allowed to keep a denomination label.
    MAX_LABEL_REL_ERROR: float = 0.12

    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for batch evaluation and interactive visualization."""

    IMAGE_DIRECTORY: str = field(default_factory=lambda: _find_image_directory())
    BROWSE_TUNE: bool = True
    SAVE_STEPS: bool = False
    OUT_DIR: str = "./pipeline_viz"


def _find_image_directory() -> str:
    """
    Locate `data/images` automatically from the current source location.

    Why auto-discovery:
    - Makes the project runnable from IDE, terminal, or notebooks without manual path edits.
    """
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
