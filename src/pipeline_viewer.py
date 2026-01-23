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
        self.images = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.images:
            raise RuntimeError("No images found")

    def get(self, idx):
        return self.images[idx % len(self.images)]


# =========================
# Pipeline Page
# =========================
class Page:
    title = "Page"

    def process(self, img):
        return img

    def render(self, axs, input_img, output_img):
        raise NotImplementedError


# =========================
# Pages
# =========================
class OverviewPage(Page):
    title = "Overview"

    def render(self, axs, original, final):
        axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Original")

        axs[0, 1].imshow(final, cmap="gray")
        axs[0, 1].set_title("Final Result")

        axs[1, 0].axis("off")
        axs[1, 1].axis("off")


class GrayPage(Page):
    title = "Grayscale"

    def process(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def render(self, axs, inp, out):
        axs[0, 0].imshow(inp[..., ::-1])
        axs[0, 0].set_title("Input")

        axs[0, 1].imshow(out, cmap="gray")
        axs[0, 1].set_title("Grayscale")


class BlurPage(Page):
    title = "Blur"

    def process(self, img):
        return cv2.GaussianBlur(img, (7, 7), 1.5)

    def render(self, axs, inp, out):
        axs[0, 0].imshow(inp, cmap="gray")
        axs[0, 0].set_title("Input")

        axs[0, 1].imshow(out, cmap="gray")
        axs[0, 1].set_title("Blurred")


class EdgePage(Page):
    title = "Edges"

    def process(self, img):
        return cv2.Canny(img, 80, 150)

    def render(self, axs, inp, out):
        axs[0, 0].imshow(inp, cmap="gray")
        axs[0, 0].set_title("Input")

        axs[0, 1].imshow(out, cmap="gray")
        axs[0, 1].set_title("Edges")


# =========================
# Page Manager
# =========================
class PageManager:
    def __init__(self):
        self.pages = []
        self.index = 0

    def add(self, page):
        self.pages.append(page)

    def next(self):
        self.index = (self.index + 1) % len(self.pages)

    def prev(self):
        self.index = (self.index - 1) % len(self.pages)

    def current(self):
        return self.pages[self.index]


# =========================
# Application
# =========================
class ImageApp:
    def __init__(self, repo):
        self.repo = repo
        self.image_index = 0

        self.pages = PageManager()
        self.pages.add(OverviewPage())
        self.pages.add(GrayPage())
        self.pages.add(BlurPage())
        self.pages.add(EdgePage())

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.22)

        self._init_ui()
        self.page_text = self.fig.text(0.5, 0.02, "", ha="center")

        self.update()
        plt.show()

    # ---------- UI ----------
    def _init_ui(self):
        ax_prev = plt.axes([0.15, 0.08, 0.15, 0.06])
        ax_next = plt.axes([0.32, 0.08, 0.15, 0.06])
        ax_page_prev = plt.axes([0.55, 0.08, 0.15, 0.06])
        ax_page_next = plt.axes([0.72, 0.08, 0.15, 0.06])

        self.btn_prev = Button(ax_prev, "Image ←")
        self.btn_next = Button(ax_next, "Image →")
        self.btn_page_prev = Button(ax_page_prev, "Step ←")
        self.btn_page_next = Button(ax_page_next, "Step →")

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_page_prev.on_clicked(self.prev_page)
        self.btn_page_next.on_clicked(self.next_page)

    # ---------- Navigation ----------
    def next_image(self, _):
        self.image_index += 1
        self.update()

    def prev_image(self, _):
        self.image_index -= 1
        self.update()

    def next_page(self, _):
        self.pages.next()
        self.update()

    def prev_page(self, _):
        self.pages.prev()
        self.update()

    # ---------- Rendering ----------
    def update(self):
        original = cv2.imread(str(self.repo.get(self.image_index)))

        # Compute pipeline
        pipeline = [original]
        for page in self.pages.pages[1:]:
            pipeline.append(page.process(pipeline[-1]))

        for ax in self.axs.flat:
            ax.clear()

        current = self.pages.current()

        if isinstance(current, OverviewPage):
            current.render(self.axs, original, pipeline[-1])
        else:
            idx = self.pages.index - 1
            current.render(self.axs, pipeline[idx], pipeline[idx + 1])

        self.fig.suptitle(
            f"{current.title} — {self.repo.get(self.image_index).name}",
            fontsize=12
        )

        self.page_text.set_text(
            f"Step {self.pages.index + 1} / {len(self.pages.pages)}"
        )

        self.fig.canvas.draw_idle()


# =========================
# Entry
# =========================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    IMG_DIR = ROOT / "data" / "images"

    app = ImageApp(ImageRepository(IMG_DIR))
