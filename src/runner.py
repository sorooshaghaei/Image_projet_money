from dataclasses import dataclass
from typing import List

import cv2

from .config import DetectionConfig, RuntimeConfig
from .dataset import DatasetRepository
from .io_utils import ImagePathResolver
from .models import PipelineResult
from .processor import CoinProcessor
from .visualization import HoughTuningBrowser, PipelineVisualizer


@dataclass
class RunStats:
    processed: int = 0
    correct: int = 0
    total_abs_error: int = 0

    def update(self, diff: int):
        self.processed += 1
        self.total_abs_error += abs(diff)
        if diff == 0:
            self.correct += 1

    def print_summary(self):
        if self.processed <= 0:
            print("[WARN] No images processed. Check IMAGE_DIRECTORY path.")
            return

        accuracy = (self.correct / self.processed) * 100.0
        mae = self.total_abs_error / self.processed
        print("=" * 85)
        print(f"Total Images:     {self.processed}")
        print(f"Perfect Matches:  {self.correct}")
        print(f"Accuracy:         {accuracy:.2f}%")
        print(f"Mean Abs Error:   {mae:.2f} coins/image")
        print("=" * 85)


class PipelineApp:
    def __init__(self, runtime: RuntimeConfig):
        self._runtime = runtime
        self._detector_cfg = DetectionConfig()
        self._processor = CoinProcessor(self._detector_cfg)
        self._dataset_repo = DatasetRepository()
        self._path_resolver = ImagePathResolver(runtime.IMAGE_DIRECTORY)
        self._visualizer = PipelineVisualizer()

    def run(self):
        df = self._dataset_repo.to_dataframe()
        print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

        stats = RunStats()
        results_all: List[PipelineResult] = []

        self._print_header()

        for _, row in df.iterrows():
            filename = row["image"]
            true_count = int(row["pieces"])
            group = row["group"]

            image_path = self._path_resolver.resolve(filename, group)
            if not image_path:
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERR ] Unreadable: {filename}")
                continue

            result = self._processor.execute(img, filename)
            if not result:
                continue

            pred = int(result.coin_count)
            diff = pred - true_count
            stats.update(diff)

            status = "PERFECT" if diff == 0 else "ERROR"
            print(f"{filename:<25} | {group:<5} | {pred:<6} | {true_count:<6} | {diff:<6} | {status:<10}")

            results_all.append(result)

            if self._runtime.SAVE_STEPS:
                saved = self._visualizer.save_pipeline_steps(result, self._runtime.OUT_DIR, cols=4)
                if saved:
                    print(f"[SAVED] {saved}")

        stats.print_summary()

        if self._runtime.BROWSE_TUNE:
            HoughTuningBrowser(self._processor, results_all, cols=4).show()

    def _print_header(self):
        print("\n" + "=" * 85)
        print(f"{'FILENAME':<25} | {'GRP':<5} | {'PRED':<6} | {'TRUE':<6} | {'DIFF':<6} | {'STATUS':<10}")
        print("=" * 85)


class AppRunner:
    def main(self):
        PipelineApp(RuntimeConfig()).run()


if __name__ == "__main__":
    AppRunner().main()
