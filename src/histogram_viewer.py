from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# =========================
# Image Repository
# =========================
class ImageRepository:
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root: Path):
        self.root = root
        self.images = self._scan()

        if not self.images:
            raise RuntimeError(f"No images found in {root}")

    def _scan(self):
        return sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self):
        return len(self.images)

    def get(self, index: int) -> Path:
        return self.images[index % len(self.images)]


# =========================
# Image Viewer
# =========================
class ImageViewer:
    def __init__(self, repo: ImageRepository):
        self.repo = repo
        self.index = 0

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.2)

        self._setup_buttons()
        self.update()

        plt.show()

    # ---------- UI ----------
    def _setup_buttons(self):
        ax_prev = plt.axes([0.25, 0.05, 0.2, 0.075])
        ax_next = plt.axes([0.55, 0.05, 0.2, 0.075])

        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

    # ---------- Navigation ----------
    def next_image(self, _):
        self.index = (self.index + 1) % len(self.repo)
        self.update()

    def prev_image(self, _):
        self.index = (self.index - 1) % len(self.repo)
        self.update()

    # ---------- Rendering ----------
    def update(self):
        path = self.repo.get(self.index)
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        for ax in self.axs.flat:
            ax.clear()

        # Image
        self.axs[0, 0].imshow(img_rgb)
        self.axs[0, 0].set_title(path.name)
        self.axs[0, 0].axis("off")

        # RGB Histogram
        for i, c in enumerate(("b", "g", "r")):
            hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
            self.axs[0, 1].plot(hist, color=c)

        self.axs[0, 1].set_title("RGB Histogram")

        # Grayscale
        self.axs[1, 0].imshow(img_gray, cmap="gray")
        self.axs[1, 0].set_title("Grayscale")
        self.axs[1, 0].axis("off")

        # Grayscale histogram
        gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        self.axs[1, 1].plot(gray_hist, color="black")
        self.axs[1, 1].set_title("Grayscale Histogram")

        self.fig.canvas.draw_idle()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    IN_DIR = PROJECT_ROOT / "data" / "images"

    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    repo = ImageRepository(IN_DIR)
    viewer = ImageViewer(repo)
