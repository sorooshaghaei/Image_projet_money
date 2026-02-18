from dataclasses import dataclass
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
    IMAGE_DIRECTORY: str = "/Users/sigmoid/Desktop/Coding/GitHub/S2/Image_projet_money/data/images"
    BROWSE_TUNE: bool = True
    SAVE_STEPS: bool = False
    OUT_DIR: str = "./pipeline_viz"
