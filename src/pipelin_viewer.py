from dataclasses import dataclass
from typing import Callable, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class Step:
    name: str
    fn: Callable[[Any], Any]            # takes "state" and returns output
    kind: str = "image"                 # "image" | "mask" | "plot" (extend later)
    cmap: Optional[str] = None          # "gray" for grayscale, etc.

class ImagePipeline:
    def __init__(self):
        self.steps: list[Step] = []

    def add(self, name: str, fn: Callable[[Any], Any], kind="image", cmap=None):
        self.steps.append(Step(name=name, fn=fn, kind=kind, cmap=cmap))
        return self

    def run(self, img_rgb: np.ndarray):
        """Return list of (title, output, step_meta). Keep it simple at first."""
        state = img_rgb
        outputs = []
        for step in self.steps:
            state = step.fn(state)
            outputs.append((step.name, state, step))
        return outputs

class PipelineViewer:
    def __init__(self, images_rgb: list[np.ndarray], paths: list[Path], pipeline: ImagePipeline):
        self.images = images_rgb
        self.paths = paths
        self.pipeline = pipeline
        self.i = 0

    def __len__(self):
        return len(self.images)

    def show(self, index: Optional[int] = None, cols: int = 4, figsize_per_cell=(4, 4)):
        if index is not None:
            self.i = int(np.clip(index, 0, len(self.images) - 1))

        img = self.images[self.i]
        name = Path(self.paths[self.i]).name if self.paths else f"img_{self.i}"

        results = self.pipeline.run(img)

        n = len(results)
        cols = max(1, cols)
        rows = int(np.ceil(n / cols))

        fig_w = figsize_per_cell[0] * cols
        fig_h = figsize_per_cell[1] * rows
        plt.figure(figsize=(fig_w, fig_h))
        plt.suptitle(f"[{self.i+1}/{len(self.images)}] {name}", fontsize=14)

        for k, (title, out, step) in enumerate(results):
            ax = plt.subplot(rows, cols, k + 1)
            self._imshow(ax, out, title, step)

        plt.tight_layout()
        plt.show()

    def next(self):
        self.i = (self.i + 1) % len(self.images)
        self.show()

    def prev(self):
        self.i = (self.i - 1) % len(self.images)
        self.show()

    def _imshow(self, ax, out, title, step: Step):
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        arr = self._to_display_array(out)

        if arr.ndim == 2:
            ax.imshow(arr, cmap=step.cmap or "gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr)

    def _to_display_array(self, x):
        """
        Normalise for display only:
        - float images expected in [0,1]
        - uint8 in [0,255]
        """
        x = np.asarray(x)

        # If grayscale uint8 -> convert to float [0,1] for stable display
        if x.dtype == np.uint8:
            if x.ndim == 2:
                return x.astype(np.float32) / 255.0
            if x.ndim == 3:
                return x  # RGB uint8 OK for matplotlib

        # float image: clip to [0,1] for display safety
        if np.issubdtype(x.dtype, np.floating):
            return np.clip(x, 0.0, 1.0)

        # fallback
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            x = (x - mn) / (mx - mn)
        return x
