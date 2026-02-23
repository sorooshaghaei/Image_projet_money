"""Dataset/repository abstractions for image listing and ground-truth lookup."""

from __future__ import annotations

from onefiler import (
    DatasetImage,
    GroundTruthEntry,
    GroundTruthRepository,
    ImageDataset,
    normalize_group_name,
)

__all__ = [
    "DatasetImage",
    "ImageDataset",
    "normalize_group_name",
    "GroundTruthEntry",
    "GroundTruthRepository",
]
