from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HoughPreset:
    dp: float
    min_dist: int
    param2: int


HOUGH_PRESETS: dict[str, HoughPreset] = {
    "test1": HoughPreset(dp=1.15, min_dist=30, param2=55),
    "test2": HoughPreset(dp=1.15, min_dist=20, param2=57),
}


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "images"


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

    blur_mode: str = "gauss" #or median
    gauss_ksize: int = 5
    gauss_sigma: float = 2.0

    active_preset: str = "test1"

    auto_param1_blur_ksize: int = gauss_ksize
    auto_param1_percentile: float = 65.0
    auto_param1_scale: float = 1.0
    auto_param1_clamp: tuple[int, int] = (30, 220)

    max_radius: int = 100
    min_radius_sweep_start: int = 10
    min_radius_sweep_end: int = 140
    min_radius_sweep_step: int = 2

    circle_outline_color: tuple[int, int, int] = (0, 255, 0)
    circle_outline_thickness: int = 2
    center_color: tuple[int, int, int] = (255, 0, 0)
    center_radius: int = 2
    center_thickness: int = 3

    analysis_border_ratio: float = 0.24
    analysis_sat_delta_threshold: float | None = None
    analysis_bimetal_mode: str = "mean-color"
    analysis_material_mode: str = "lab"

    def get_preset(self, preset_name: str | None = None) -> HoughPreset:
        key = preset_name or self.active_preset
        if key not in HOUGH_PRESETS:
            available = ", ".join(sorted(HOUGH_PRESETS))
            raise ValueError(f"Unknown preset '{key}'. Available presets: {available}")
        return HOUGH_PRESETS[key]
