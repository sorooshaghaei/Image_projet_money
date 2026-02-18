from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PipelineStep:
    name: str
    image: np.ndarray
    cmap: str


@dataclass
class PipelineResult:
    steps: List[PipelineStep]
    coin_count: int
    is_inverted: bool
    source_filename: str
