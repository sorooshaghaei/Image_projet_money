from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DetectionConfig:
    TARGET_WIDTH: int = 800
    BLUR_KERNEL_SIZE: int = 15

    HOUGH_DP: float = 1.2
    HOUGH_MIN_DIST: int = 70
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 45
    HOUGH_MIN_RADIUS: int = 10
    HOUGH_MAX_RADIUS: int = 150

    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class RuntimeConfig:
    IMAGE_DIRECTORY: str = field(default_factory=lambda: _find_image_directory())
    BROWSE_TUNE: bool = True
    SAVE_STEPS: bool = False
    OUT_DIR: str = "./pipeline_viz"


def _find_image_directory() -> str:
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
