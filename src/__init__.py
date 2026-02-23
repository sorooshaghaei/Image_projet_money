"""Public API for the modular coin-analysis pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "HoughPreset",
    "HOUGH_PRESETS",
    "PipelineConfig",
    "default_dataset_dir",
    "DatasetImage",
    "ImageDataset",
    "GroundTruthEntry",
    "GroundTruthRepository",
    "AnalysisResult",
    "PipelineStep",
    "Analyzer",
    "AppRunner",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "HoughPreset": ("src.config", "HoughPreset"),
    "HOUGH_PRESETS": ("src.config", "HOUGH_PRESETS"),
    "PipelineConfig": ("src.config", "PipelineConfig"),
    "default_dataset_dir": ("src.config", "default_dataset_dir"),
    "DatasetImage": ("src.dataset", "DatasetImage"),
    "ImageDataset": ("src.dataset", "ImageDataset"),
    "GroundTruthEntry": ("src.dataset", "GroundTruthEntry"),
    "GroundTruthRepository": ("src.dataset", "GroundTruthRepository"),
    "AnalysisResult": ("src.models", "AnalysisResult"),
    "PipelineStep": ("src.models", "PipelineStep"),
    "Analyzer": ("src.analyzer", "Analyzer"),
    "AppRunner": ("src.runner", "AppRunner"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from src.analyzer import Analyzer
    from src.config import HOUGH_PRESETS, HoughPreset, PipelineConfig, default_dataset_dir
    from src.dataset import DatasetImage, GroundTruthEntry, GroundTruthRepository, ImageDataset
    from src.models import AnalysisResult, PipelineStep
    from src.runner import AppRunner
