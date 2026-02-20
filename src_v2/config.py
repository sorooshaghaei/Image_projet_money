from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class HoughSettings:
    """Parameters for circle proposals in textured scenes."""

    dp: float = 1.2
    min_dist: int = 58
    param1: int = 120
    param2: int = 44
    min_radius: int = 10
    max_radius: int = 160


@dataclass(frozen=True)
class ContourSettings:
    """Parameters for contour-based detection on clean backgrounds."""

    blur_kernel: int = 5
    morph_kernel: int = 3
    min_area: int = 320
    min_circularity: float = 0.72
    max_aspect_ratio: float = 1.45


@dataclass(frozen=True)
class WatershedSettings:
    """Parameters for separating touching or overlapping coins."""

    open_kernel: int = 3
    close_kernel: int = 5
    fg_ratio: float = 0.46
    min_seed_area: int = 120
    min_region_area: int = 450


@dataclass(frozen=True)
class PolicySettings:
    """Scene policy for difficulty labels and algorithm routing."""

    easy_threshold: float = 0.23
    medium_threshold: float = 0.43
    overlap_distance_scale: float = 0.88
    contour_merge_circularity: float = 0.72
    contour_merge_area_ratio: float = 0.05
    overlap_merge_score: float = 0.20
    target_width: int = 1100


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime defaults for CLI entrypoints."""

    image_directory: str = field(default_factory=lambda: _find_image_directory())
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    report_csv_path: str = "report/runtime_v2_policy_trace.csv"



def _find_image_directory() -> str:
    """Locate `data/images` relative to this package."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data" / "images"
        if candidate.exists():
            return str(candidate)
    return str((current.parent / ".." / "data" / "images").resolve())
