from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import cv2

from .analyzer import HybridCoinAnalyzer
from .config import RuntimeConfig
from .io_utils import ensure_parent_dir, list_image_paths, short_path
from .models import AnalysisResult


class ExperimentRunner:
    """CLI pipeline for scene-aware method routing and trace reporting."""

    _TABLE_RULE = "-" * 126

    def __init__(self, runtime: RuntimeConfig):
        self._runtime = runtime
        self._analyzer = HybridCoinAnalyzer()

    def run(
        self,
        *,
        path: str,
        short_root: str,
        mode: str,
        max_images: int,
        csv_path: str,
        visualize: bool,
    ) -> List[AnalysisResult]:
        image_paths = list_image_paths(path, self._runtime.valid_extensions)
        if max_images > 0:
            image_paths = image_paths[:max_images]

        if not image_paths:
            print(f"[WARN] no images found under: {path}")
            return []

        short_root_path = Path(short_root)
        print(f"[INFO] mode={mode} | path={Path(path).as_posix()} | images={len(image_paths)}")
        self._print_header()

        results: List[AnalysisResult] = []
        for idx, img_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"{idx:>3}  {img_path.name:<34} {'-':<10} {'-':<16} {'-':>7} {'-':>8} {'-':>8} {'IOERR':<8}")
                continue

            rel = short_path(img_path, short_root_path)
            result = self._analyzer.analyze(
                image,
                source_path=img_path.as_posix(),
                short_path=rel,
                mode=mode,
            )
            results.append(result)

            m = result.metrics
            print(
                f"{idx:>3}  {self._truncate(rel, 34):<34} "
                f"{m.background_label:<10} "
                f"{result.selected_method:<16} "
                f"{len(result.circles):>7} "
                f"{m.texture_score:>8.3f} "
                f"{m.contour_merge_score:>8.3f} "
                f"{'YES' if m.likely_overlap else 'NO':<8}"
            )

        self._print_summary(results)

        if csv_path:
            ensure_parent_dir(csv_path)
            self._write_csv(csv_path, results)
            print(f"[INFO] trace csv written to {Path(csv_path).as_posix()}")

        if visualize and results:
            from .visualizer import HybridVisualizer

            HybridVisualizer(
                analyzer=self._analyzer,
                image_paths=[Path(r.source_path) for r in results],
                short_root=short_root_path,
                start_mode=mode,
            ).show()

        return results

    def _print_header(self) -> None:
        print(self._TABLE_RULE)
        print(
            f"{'#':>3}  {'IMAGE':<34} {'BG':<10} {'METHOD':<16} "
            f"{'COINS':>7} {'TEX':>8} {'MERGE':>8} {'OVERLAP':<8}"
        )
        print(self._TABLE_RULE)

    def _print_summary(self, results: Sequence[AnalysisResult]) -> None:
        if not results:
            return

        total = len(results)
        easy = sum(1 for r in results if r.metrics.background_label == "easy")
        medium = sum(1 for r in results if r.metrics.background_label == "medium")
        difficult = sum(1 for r in results if r.metrics.background_label == "difficult")

        by_method = {
            "contours": sum(1 for r in results if r.selected_method == "contours"),
            "hough": sum(1 for r in results if r.selected_method == "hough"),
            "watershed": sum(1 for r in results if r.selected_method == "watershed"),
            "hough+watershed": sum(1 for r in results if r.selected_method == "hough+watershed"),
        }

        print(self._TABLE_RULE)
        print(f"[SUMMARY] total={total} | easy={easy} medium={medium} difficult={difficult}")
        print(
            "[SUMMARY] methods="
            f"contours:{by_method['contours']} "
            f"hough:{by_method['hough']} "
            f"watershed:{by_method['watershed']} "
            f"hough+watershed:{by_method['hough+watershed']}"
        )

    def _write_csv(self, csv_path: str, results: Sequence[AnalysisResult]) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "source_path",
                    "short_path",
                    "background_label",
                    "selected_method",
                    "coin_count",
                    "border_cv",
                    "edge_density",
                    "texture_score",
                    "contour_merge_score",
                    "hough_overlap_pairs",
                    "likely_overlap",
                ]
            )
            for result in results:
                m = result.metrics
                writer.writerow(
                    [
                        result.source_path,
                        result.short_path,
                        m.background_label,
                        result.selected_method,
                        len(result.circles),
                        f"{m.border_cv:.6f}",
                        f"{m.edge_density:.6f}",
                        f"{m.texture_score:.6f}",
                        f"{m.contour_merge_score:.6f}",
                        int(m.hough_overlap_pairs),
                        int(m.likely_overlap),
                    ]
                )

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return f"...{text[-(width - 3):]}"


class AppRunner:
    """Thin app entrypoint used by `main_v2.py`."""

    def main(self) -> None:
        runtime = RuntimeConfig()
        args = _build_parser(runtime).parse_args()

        ExperimentRunner(runtime).run(
            path=args.path,
            short_root=args.short_root,
            mode=args.mode,
            max_images=args.max_images,
            csv_path=args.csv,
            visualize=args.visualize,
        )


def _build_parser(runtime: RuntimeConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid coin analysis: background label (easy/medium/difficult) + auto method routing "
            "(contours/hough/watershed/hough+watershed)."
        )
    )
    parser.add_argument("--path", default=runtime.image_directory, help="image file or directory")
    parser.add_argument(
        "--short-root",
        default=runtime.image_directory,
        help="root used to print short relative image paths",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "contours", "hough", "watershed", "hybrid", "hough+watershed"],
        help="force one method or keep auto policy",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--csv", default=runtime.report_csv_path, help="path to save per-image trace csv")
    parser.add_argument("--visualize", action="store_true", help="open interactive visualizer")
    return parser


if __name__ == "__main__":
    AppRunner().main()
