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
    "HoughPreset": ("src.pipeline.config", "HoughPreset"),
    "HOUGH_PRESETS": ("src.pipeline.config", "HOUGH_PRESETS"),
    "PipelineConfig": ("src.pipeline.config", "PipelineConfig"),
    "default_dataset_dir": ("src.pipeline.config", "default_dataset_dir"),
    "DatasetImage": ("src.data.dataset", "DatasetImage"),
    "ImageDataset": ("src.data.dataset", "ImageDataset"),
    "GroundTruthEntry": ("src.data.ground_truth", "GroundTruthEntry"),
    "GroundTruthRepository": ("src.data.ground_truth", "GroundTruthRepository"),
    "AnalysisResult": ("src.pipeline.models", "AnalysisResult"),
    "PipelineStep": ("src.pipeline.models", "PipelineStep"),
    "Analyzer": ("src.pipeline.orchestrator", "Analyzer"),
    "AppRunner": ("src.app.cli", "AppRunner"),
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
    from src.app.cli import AppRunner
    from src.data.dataset import DatasetImage, ImageDataset
    from src.data.ground_truth import GroundTruthEntry, GroundTruthRepository
    from src.pipeline.config import HOUGH_PRESETS, HoughPreset, PipelineConfig, default_dataset_dir
    from src.pipeline.models import AnalysisResult, PipelineStep
    from src.pipeline.orchestrator import Analyzer
