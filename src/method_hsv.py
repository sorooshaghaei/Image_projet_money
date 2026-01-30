import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# FIND PROJECT ROOT
# =========================
cwd = Path.cwd()

while not (cwd / "data").exists():
    if cwd.parent == cwd:
        raise RuntimeError("Could not find project root with /data")
    cwd = cwd.parent

IMAGE_DIR = cwd / "data" / "images"

# =========================
# LOAD IMAGES
# =========================
image_paths = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise RuntimeError("No images found")

print(f"[INFO] Found {len(image_paths)} images")

index = 0

# =========================
# PROCESSING PIPELINE
# =========================
def process_image(img):
    img = cv2.bilateralFilter(img, 7, 50, 100)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    _, _, B = cv2.split(lab)

    # Gold mask (HSV)
    gold_mask = cv2.inRange(
        hsv,
        np.array([15, 60, 70]),
        np.array([40, 255, 255])
    )

    # Bronze mask (LAB)
    bronze_mask = cv2.inRange(B, 135, 190)

    # Combine
    mask = cv2.bitwise_or(gold_mask, bronze_mask)

    # Cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result, mask


# =========================
# DISPLAY FUNCTION
# =========================
def show_image():
    img = cv2.imread(str(image_paths[index]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, mask = process_image(img)

    plt.clf()

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("Detected Coins")
    plt.axis("off")

    plt.suptitle(f"{index+1}/{len(image_paths)} — {image_paths[index].name}")
    plt.tight_layout()
    plt.draw()


# =========================
# KEYBOARD CONTROL
# =========================
def on_key(event):
    global index

    if event.key in ["right", "d"]:
        index = (index + 1) % len(image_paths)
    elif event.key in ["left", "a"]:
        index = (index - 1) % len(image_paths)
    elif event.key == "q":
        plt.close()
        return

    show_image()


# =========================
# MAIN
# =========================
plt.figure(figsize=(10, 6))
plt.connect("key_press_event", on_key)

show_image()
plt.show()
