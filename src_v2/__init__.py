from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "HybridCoinAnalyzer",
    "HoughSettings",
    "ContourSettings",
    "WatershedSettings",
    "PolicySettings",
    "RuntimeConfig",
    "ExperimentRunner",
    "AppRunner",
    "HybridVisualizer",
]

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HybridCoinAnalyzer": (".analyzer", "HybridCoinAnalyzer"),
    "HoughSettings": (".config", "HoughSettings"),
    "ContourSettings": (".config", "ContourSettings"),
    "WatershedSettings": (".config", "WatershedSettings"),
    "PolicySettings": (".config", "PolicySettings"),
    "RuntimeConfig": (".config", "RuntimeConfig"),
    "ExperimentRunner": (".runner", "ExperimentRunner"),
    "AppRunner": (".runner", "AppRunner"),
    "HybridVisualizer": (".visualizer", "HybridVisualizer"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
