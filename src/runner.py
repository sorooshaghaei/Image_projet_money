from typing import List

import cv2
import pandas as pd

from .config import DetectionConfig, RuntimeConfig
from .dataset import DATA_ROWS
from .io_utils import get_image_path
from .models import PipelineResult
from .processor import CoinProcessor
from .visualization import browse_and_tune, save_pipeline_steps


def run(runtime: RuntimeConfig = RuntimeConfig()):
    config = DetectionConfig()
    processor = CoinProcessor(config)

    df = pd.DataFrame(DATA_ROWS, columns=["image", "pieces", "value_eur", "group"])
    print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

    correct = 0
    total_processed = 0
    total_abs_error = 0

    results_all: List[PipelineResult] = []

    print("\n" + "=" * 85)
    print(f"{'FILENAME':<25} | {'GRP':<5} | {'PRED':<6} | {'TRUE':<6} | {'DIFF':<6} | {'STATUS':<10}")
    print("=" * 85)

    for _, row in df.iterrows():
        filename = row["image"]
        true_count = int(row["pieces"])
        group = row["group"]

        image_path = get_image_path(runtime.IMAGE_DIRECTORY, filename, group)
        if not image_path:
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERR ] Unreadable: {filename}")
            continue

        result = processor.execute(img, filename)
        if not result:
            continue

        pred = int(result.coin_count)
        diff = pred - true_count
        total_abs_error += abs(diff)
        total_processed += 1

        status = "PERFECT" if diff == 0 else "ERROR"
        if diff == 0:
            correct += 1

        print(f"{filename:<25} | {group:<5} | {pred:<6} | {true_count:<6} | {diff:<6} | {status:<10}")

        results_all.append(result)

        if runtime.SAVE_STEPS:
            saved = save_pipeline_steps(result, runtime.OUT_DIR, cols=4)
            if saved:
                print(f"[SAVED] {saved}")

    if total_processed > 0:
        acc = (correct / total_processed) * 100.0
        mae = total_abs_error / total_processed
        print("=" * 85)
        print(f"Total Images:     {total_processed}")
        print(f"Perfect Matches:  {correct}")
        print(f"Accuracy:         {acc:.2f}%")
        print(f"Mean Abs Error:   {mae:.2f} coins/image")
        print("=" * 85)
    else:
        print("[WARN] No images processed. Check IMAGE_DIRECTORY path.")

    if runtime.BROWSE_TUNE:
        browse_and_tune(processor, results_all, cols=4)


def main():
    run(RuntimeConfig())


if __name__ == "__main__":
    main()
