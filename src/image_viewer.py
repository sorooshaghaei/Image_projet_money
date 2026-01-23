#!/usr/bin/env python3
# src/image_viewer.py
#
# Minimal image viewer for project layout:
#   project_root/
#     data/images/
#     src/
#
# Run from anywhere:
#   python3 src/image_viewer.py
#
# Keys:
#   n / SPACE : next image
#   p         : previous image
#   r         : reload current image (if file changed)
#   q / ESC   : quit
#
from pathlib import Path
import cv2 as cv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "data" / "images"

WIN = "viewer"


def iter_images_recursive(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def draw_header(img, text: str):
    # Simple readable header (black outline + white text)
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv.LINE_AA)


def main():
    if not IN_DIR.exists():
        print(f"Input dir not found: {IN_DIR}")
        return

    paths = iter_images_recursive(IN_DIR)
    if not paths:
        print(f"No images found under: {IN_DIR}")
        return

    cv.namedWindow(WIN, cv.WINDOW_NORMAL)

    idx = 0
    cached = None
    cached_path = None

    while True:
        p = paths[idx]

        if cached is None or cached_path != p:
            img = cv.imread(str(p), cv.IMREAD_COLOR)
            cached = img
            cached_path = p
        else:
            img = cached

        if img is None:
            print(f"[skip] failed to read: {p}")
            idx = (idx + 1) % len(paths)
            cached = None
            cached_path = None
            continue

        vis = img.copy()
        header = f"[{idx+1}/{len(paths)}] {p.relative_to(PROJECT_ROOT)}"
        draw_header(vis, header)

        cv.imshow(WIN, vis)
        key = cv.waitKey(0) & 0xFF

        if key in (27, ord("q")):          # ESC / q
            break
        elif key in (ord("n"), ord(" ")):  # next
            idx = (idx + 1) % len(paths)
            cached = None
            cached_path = None
        elif key == ord("p"):              # prev
            idx = (idx - 1) % len(paths)
            cached = None
            cached_path = None
        elif key == ord("r"):              # reload current
            cached = None
            cached_path = None
        else:
            # any other key -> next (optional convenience)
            idx = (idx + 1) % len(paths)
            cached = None
            cached_path = None

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
