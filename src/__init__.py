"""
Package public API.

This module exposes the main classes so users can do:
    from src import CoinProcessor, PipelineVisualizer, DetectionConfig, ...

To keep imports fast and to reduce circular-import issues, we use lazy imports
(PEP 562: __getattr__) so heavy modules are imported only when needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

# Public symbols (what `from src import *` exports)
__all__ = [
    "DetectionConfig",
    "RuntimeConfig",
    "DatasetRepository",
    "ImagePathResolver",
    "PipelineResult",
    "PipelineStep",
    "CoinProcessor",
    "RunStats",
    "PipelineApp",
    "AppRunner",
    "PipelineVisualizer",
    "HoughTuningBrowser",
]

# Map: public name -> (relative module, attribute name)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DetectionConfig": (".config", "DetectionConfig"),
    "RuntimeConfig": (".config", "RuntimeConfig"),
    "DatasetRepository": (".dataset", "DatasetRepository"),
    "ImagePathResolver": (".io_utils", "ImagePathResolver"),
    "PipelineResult": (".models", "PipelineResult"),
    "PipelineStep": (".models", "PipelineStep"),
    "CoinProcessor": (".processor", "CoinProcessor"),
    "RunStats": (".runner", "RunStats"),
    "PipelineApp": (".runner", "PipelineApp"),
    "AppRunner": (".runner", "AppRunner"),
    "PipelineVisualizer": (".visualization", "PipelineVisualizer"),
    "HoughTuningBrowser": (".visualization", "HoughTuningBrowser"),
}

if TYPE_CHECKING:
    # Only for type checkers / IDE autocomplete (no runtime imports)
    from .config import DetectionConfig, RuntimeConfig
    from .dataset import DatasetRepository
    from .io_utils import ImagePathResolver
    from .models import PipelineResult, PipelineStep
    from .processor import CoinProcessor
    from .runner import AppRunner, PipelineApp, RunStats
    from .visualization import HoughTuningBrowser, PipelineVisualizer


def __getattr__(name: str) -> Any:
    """
    Lazy attribute loader.

    Called when `name` is not found in globals().
    This lets `from src import CoinProcessor` work without importing
    heavy submodules at package import time.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)

    # Cache it so future access is fast and doesn't re-import.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Improve autocomplete: include lazily-exported names in dir()."""
    return sorted(list(globals().keys()) + __all__)
