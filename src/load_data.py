# load_data.py
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BASE_DIR = Path(__file__).resolve().parent  # folder containing this file


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def list_images(image_dir="data/images", recursive=True):
    image_dir = resolve_path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    pattern = "**/*" if recursive else "*"
    paths = [p for p in image_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return paths


def load_image_rgb(path, background=(255, 255, 255)):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:  # grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 3:  # BGR -> RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[2] == 4:  # BGRA -> composite -> RGB
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3].astype(np.float32) / 255.0)[..., None]  # (H,W,1)

        bg_rgb = np.array(background, dtype=np.float32)
        bg_bgr = bg_rgb[::-1]  # RGB -> BGR

        comp_bgr = bgr * alpha + bg_bgr * (1.0 - alpha)
        comp_bgr = np.clip(comp_bgr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

    return None


def resize_max(img_rgb, max_size=None):
    if max_size is None:
        return img_rgb
    h, w = img_rgb.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img_rgb
    scale = max_size / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_dataset(image_dir="../data/images", recursive=True, max_size=None, alpha_bg=(255, 255, 255)):
    paths = list_images(image_dir, recursive=recursive)

    images_rgb, ok_paths = [], []
    bad = 0
    for p in paths:
        img = load_image_rgb(p, background=alpha_bg)
        if img is None:
            bad += 1
            continue
        images_rgb.append(resize_max(img, max_size=max_size))
        ok_paths.append(p)

    print(f"[INFO] loaded {len(images_rgb)} / {len(paths)} images from {resolve_path(image_dir)}")
    if bad:
        print(f"[WARN] {bad} images could not be read.")
    return images_rgb, ok_paths


def save_debug_grid(images_rgb, paths=None, out_path="debug_grid.png", cols=4, max_items=16, title=None):
    out_path = resolve_path(out_path)

    n = min(len(images_rgb), max_items)
    if n == 0:
        raise ValueError("No images to display.")

    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images_rgb[i])
        ax.axis("off")
        if paths is not None:
            ax.set_title(Path(paths[i]).name, fontsize=8)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved debug grid -> {out_path}")


if __name__ == "__main__":
    imgs, ps = load_dataset("../data/images", recursive=True, max_size=800, alpha_bg=(255, 255, 255))
    # uncomment to generate preview sheet next to load_data.py
    # save_debug_grid(imgs, ps, out_path="debug_grid.png", cols=4, max_items=16, title="Loaded images")
