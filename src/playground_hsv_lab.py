"""
Minimal Tkinter + OpenCV mask playground (HSV or Lab)

- No cv2.imshow / no HighGUI trackbars
- Uses opencv-python-headless + Pillow for rendering
- Scans data/images/ recursively
- Lets you choose HSV or Lab and tune 6 thresholds
- Shows: Original | Mask | Result

Install:
  pip uninstall -y opencv-python opencv-contrib-python
  pip install -U opencv-python-headless pillow numpy

Run:
  python src/tk_mask_playground.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# ----------------------------- small utils -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def natural_numeric_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"\d+", p.stem)
    if not m:
        return (10**12, p.name.lower())
    return (int(m.group()), p.name.lower())


def to_tk_image(img: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    if img.ndim == 2:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale != 1.0:
        rgb = cv.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

    return ImageTk.PhotoImage(Image.fromarray(rgb))


def list_images(image_dir: Path) -> List[Path]:
    pats = ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP")
    paths: List[Path] = []
    for pat in pats:
        paths.extend(image_dir.rglob(pat))
    return sorted(paths, key=natural_numeric_key)


# ----------------------------- processing -----------------------------
def build_mask(bgr: np.ndarray, cfg: Dict[str, int | str | bool]) -> np.ndarray:
    # Optional blur helps reduce speckle in mask
    if cfg["blur"] > 0:
        k = int(cfg["blur"]) | 1
        k = clamp(k, 1, 31) | 1
        bgr2 = cv.GaussianBlur(bgr, (k, k), 0)
    else:
        bgr2 = bgr

    if cfg["space"] == "HSV":
        x = cv.cvtColor(bgr2, cv.COLOR_BGR2HSV)
        # OpenCV HSV: H in [0..179], S,V in [0..255]
        lo = np.array([cfg["c1_min"], cfg["c2_min"], cfg["c3_min"]], np.uint8)
        hi = np.array([cfg["c1_max"], cfg["c2_max"], cfg["c3_max"]], np.uint8)
        mask = cv.inRange(x, lo, hi)

    else:  # Lab
        x = cv.cvtColor(bgr2, cv.COLOR_BGR2LAB)
        # OpenCV Lab: L,a,b in [0..255] (offset/scaled)
        lo = np.array([cfg["c1_min"], cfg["c2_min"], cfg["c3_min"]], np.uint8)
        hi = np.array([cfg["c1_max"], cfg["c2_max"], cfg["c3_max"]], np.uint8)
        mask = cv.inRange(x, lo, hi)

    if cfg["invert"]:
        mask = cv.bitwise_not(mask)

    # Simple morphology cleanup (optional)
    k = int(cfg["morph_k"]) | 1
    k = clamp(k, 1, 51) | 1
    it = clamp(int(cfg["morph_it"]), 0, 10)

    if it > 0 and k >= 1 and cfg["morph_op"] != "off":
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        op = cfg["morph_op"]
        if op == "open":
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=it)
        elif op == "close":
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=it)
        elif op == "erode":
            mask = cv.erode(mask, kernel, iterations=it)
        elif op == "dilate":
            mask = cv.dilate(mask, kernel, iterations=it)

    return mask


def apply_mask(bgr: np.ndarray, mask_0_255: np.ndarray) -> np.ndarray:
    return cv.bitwise_and(bgr, bgr, mask=mask_0_255)


# ----------------------------- Tk app -----------------------------
class TkMaskPlayground(tk.Tk):
    def __init__(self, root: Path):
        super().__init__()
        self.title("Mask Playground (Tkinter)")

        self.root_path = root
        self.image_dir = root / "data" / "images"
        self.paths = list_images(self.image_dir)
        if not self.paths:
            raise SystemExit(f"No images under: {self.image_dir}")

        self.idx = 0
        self.img0 = self._load_current()

        # layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.ctrl = ttk.Frame(self, padding=10)
        self.ctrl.grid(row=0, column=0, sticky="nsw")
        self.ctrl.columnconfigure(0, weight=1)

        self.view = ttk.Frame(self, padding=10)
        self.view.grid(row=0, column=1, sticky="nsew")
        self.view.columnconfigure((0, 1, 2), weight=1)
        self.view.rowconfigure(1, weight=1)

        # status
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.ctrl, textvariable=self.status_var, wraplength=320).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )

        # vars
        self.vars: Dict[str, tk.Variable] = {}
        self._build_controls()
        self._build_tiles()

        # keybinds
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<Right>", lambda _e: self.next_image())
        self.bind("<Left>", lambda _e: self.prev_image())
        self.bind("p", lambda _e: self.print_cfg())

        # render loop
        self.max_tile_w = 520
        self.max_tile_h = 340
        self._tk_imgs: Dict[str, ImageTk.PhotoImage] = {}
        self._updating = True
        self.after(0, self.update_loop)

        self._set_status()

    # --------- ui helpers ---------
    def _add_int(self, parent, key: str, label: str, lo: int, hi: int, row: int, default: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        v = tk.IntVar(value=default)
        self.vars[key] = v
        s = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal")
        s.set(default)
        s.grid(row=row + 1, column=0, sticky="ew", pady=(2, 4))

        def on_scale(val):
            v.set(int(float(val)))

        s.configure(command=on_scale)
        ttk.Label(parent, textvariable=v).grid(row=row + 2, column=0, sticky="w", pady=(0, 8))

    def _add_bool(self, parent, key: str, text: str, row: int, default: int = 0) -> None:
        v = tk.IntVar(value=default)
        self.vars[key] = v
        ttk.Checkbutton(parent, text=text, variable=v).grid(row=row, column=0, sticky="w", pady=(0, 8))

    def _add_combo(self, parent, key: str, label: str, values: List[str], row: int, default: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        v = tk.StringVar(value=default)
        self.vars[key] = v
        cb = ttk.Combobox(parent, values=values, textvariable=v, state="readonly")
        cb.grid(row=row + 1, column=0, sticky="ew", pady=(2, 8))

    # --------- build ui ---------
    def _build_controls(self) -> None:
        nb = ttk.Notebook(self.ctrl)
        nb.grid(row=1, column=0, sticky="nsew")

        tab = ttk.Frame(nb, padding=8)
        tab.columnconfigure(0, weight=1)
        nb.add(tab, text="Mask")

        r = 0
        self._add_combo(tab, "space", "Colorspace", ["HSV", "Lab"], r, default="HSV"); r += 2

        # HSV note: H [0..179]
        self._add_int(tab, "c1_min", "C1 min (H or L)", 0, 255, r, default=0); r += 3
        self._add_int(tab, "c1_max", "C1 max (H or L)", 0, 255, r, default=255); r += 3
        self._add_int(tab, "c2_min", "C2 min (S or a)", 0, 255, r, default=0); r += 3
        self._add_int(tab, "c2_max", "C2 max (S or a)", 0, 255, r, default=255); r += 3
        self._add_int(tab, "c3_min", "C3 min (V or b)", 0, 255, r, default=0); r += 3
        self._add_int(tab, "c3_max", "C3 max (V or b)", 0, 255, r, default=255); r += 3

        self._add_bool(tab, "invert", "Invert mask", r, default=0); r += 1
        self._add_int(tab, "blur", "Pre-blur k (0=off, odd 1..31)", 0, 31, r, default=5); r += 3

        self._add_combo(tab, "morph_op", "Morph", ["off", "open", "close", "erode", "dilate"], r, default="off"); r += 2
        self._add_int(tab, "morph_k", "Morph k (odd 1..51)", 1, 51, r, default=5); r += 3
        self._add_int(tab, "morph_it", "Morph it (0..10)", 0, 10, r, default=1); r += 3

        # buttons
        btns = ttk.Frame(self.ctrl, padding=(0, 10, 0, 0))
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure((0, 1), weight=1)
        ttk.Button(btns, text="Prev (←)", command=self.prev_image).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Next (→)", command=self.next_image).grid(row=0, column=1, sticky="ew")

        ttk.Button(self.ctrl, text="Quit (Esc)", command=self.destroy).grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _build_tiles(self) -> None:
        ttk.Label(self.view, text="Original").grid(row=0, column=0, sticky="w")
        ttk.Label(self.view, text="Mask").grid(row=0, column=1, sticky="w")
        ttk.Label(self.view, text="Result").grid(row=0, column=2, sticky="w")

        self.lbl_orig = ttk.Label(self.view)
        self.lbl_mask = ttk.Label(self.view)
        self.lbl_res = ttk.Label(self.view)

        self.lbl_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.lbl_mask.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        self.lbl_res.grid(row=1, column=2, sticky="nsew")

    # --------- io / status ---------
    def _load_current(self) -> np.ndarray:
        p = self.paths[self.idx]
        img = cv.imread(str(p))
        if img is None:
            raise SystemExit(f"Failed to load: {p}")
        return img

    def _set_status(self) -> None:
        p = self.paths[self.idx]
        self.status_var.set(
            f"Image {self.idx+1}/{len(self.paths)}: {p.name}\n"
            f"Keys: Esc quit | ←/→ prev/next | p print cfg"
        )

    def next_image(self) -> None:
        self.idx = (self.idx + 1) % len(self.paths)
        self.img0 = self._load_current()
        self._set_status()

    def prev_image(self) -> None:
        self.idx = (self.idx - 1) % len(self.paths)
        self.img0 = self._load_current()
        self._set_status()

    # --------- cfg / loop ---------
    def read_cfg(self) -> Dict[str, int | str | bool]:
        space = str(self.vars["space"].get())

        # For HSV, H is 0..179 in OpenCV; clamp if user slides to 255.
        c1_min = clamp(int(self.vars["c1_min"].get()), 0, 255)
        c1_max = clamp(int(self.vars["c1_max"].get()), 0, 255)
        if space == "HSV":
            c1_min = clamp(c1_min, 0, 179)
            c1_max = clamp(c1_max, 0, 179)

        c2_min = clamp(int(self.vars["c2_min"].get()), 0, 255)
        c2_max = clamp(int(self.vars["c2_max"].get()), 0, 255)
        c3_min = clamp(int(self.vars["c3_min"].get()), 0, 255)
        c3_max = clamp(int(self.vars["c3_max"].get()), 0, 255)

        # enforce min<=max
        c1_min, c1_max = min(c1_min, c1_max), max(c1_min, c1_max)
        c2_min, c2_max = min(c2_min, c2_max), max(c2_min, c2_max)
        c3_min, c3_max = min(c3_min, c3_max), max(c3_min, c3_max)

        return {
            "space": space,
            "c1_min": c1_min,
            "c1_max": c1_max,
            "c2_min": c2_min,
            "c2_max": c2_max,
            "c3_min": c3_min,
            "c3_max": c3_max,
            "invert": int(self.vars["invert"].get()) == 1,
            "blur": clamp(int(self.vars["blur"].get()), 0, 31),
            "morph_op": str(self.vars["morph_op"].get()),
            "morph_k": clamp(int(self.vars["morph_k"].get()), 1, 51),
            "morph_it": clamp(int(self.vars["morph_it"].get()), 0, 10),
        }

    def print_cfg(self) -> None:
        cfg = self.read_cfg()
        p = self.paths[self.idx]
        print("\n" + "=" * 60)
        print(f"Image: {p.name}")
        for k in sorted(cfg.keys()):
            print(f"{k}: {cfg[k]}")
        print("=" * 60 + "\n")

    def update_loop(self) -> None:
        if not self._updating:
            return

        cfg = self.read_cfg()
        mask = build_mask(self.img0, cfg)
        res = apply_mask(self.img0, mask)

        tk_orig = to_tk_image(self.img0, self.max_tile_w, self.max_tile_h)
        tk_mask = to_tk_image(mask, self.max_tile_w, self.max_tile_h)
        tk_res = to_tk_image(res, self.max_tile_w, self.max_tile_h)

        self._tk_imgs["orig"] = tk_orig
        self._tk_imgs["mask"] = tk_mask
        self._tk_imgs["res"] = tk_res

        self.lbl_orig.configure(image=tk_orig)
        self.lbl_mask.configure(image=tk_mask)
        self.lbl_res.configure(image=tk_res)

        self.after(30, self.update_loop)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    app = TkMaskPlayground(root)
    app.mainloop()


if __name__ == "__main__":
    main()
