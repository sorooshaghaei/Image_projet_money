"""
Tkinter + OpenCV Image Processing Playground (no HighGUI trackbars)

- Uses Tkinter widgets (sliders/checkboxes/combos) instead of cv2.createTrackbar
- Renders multiple stages in a Tk grid using PIL.ImageTk
- Works with opencv-python-headless (recommended on Fedora/Wayland)

Install (inside venv):
  pip uninstall -y opencv-python opencv-contrib-python
  pip install -U opencv-python-headless pillow numpy

Run:
  python src/tk_playground.py

Images:
  data/images/ (scans recursively)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# ----------------------------- utils -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def natural_numeric_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"\d+", p.stem)
    if not m:
        return (10**12, p.name.lower())
    return (int(m.group()), p.name.lower())


def ensure_u8_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def connected_components_filter(binary_0_255: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary_0_255
    n, labels, stats, _ = cv.connectedComponentsWithStats(binary_0_255, connectivity=8)
    out = np.zeros_like(binary_0_255)
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def fill_holes(binary_0_255: np.ndarray) -> np.ndarray:
    h, w = binary_0_255.shape[:2]
    mask = binary_0_255.copy()
    ff = mask.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(ff, flood, seedPoint=(0, 0), newVal=255)
    holes = cv.bitwise_not(ff)
    return cv.bitwise_or(mask, holes)


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return bgr
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv.LUT(bgr, table)


def to_tk_image(img: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    """
    Convert numpy image (BGR or GRAY) to Tk PhotoImage, scaled to fit in max_w x max_h.
    """
    if img.ndim == 2:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale != 1.0:
        rgb = cv.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


# ----------------------------- repo -----------------------------
class ImageRepository:
    def __init__(
        self,
        image_dir: Path,
        patterns: Tuple[str, ...] = (
            "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP"
        ),
    ):
        self.image_dir = image_dir
        self.patterns = patterns

    def list_images(self) -> List[Path]:
        paths: List[Path] = []
        for pat in self.patterns:
            paths.extend(self.image_dir.rglob(pat))
        return sorted(paths, key=natural_numeric_key)


# ----------------------------- defaults -----------------------------
@dataclass
class Defaults:
    # view
    resize_pct: int = 100         # 10..200
    rotate_deg: int = 0           # 0..360
    brightness: int = 0           # -100..100
    contrast_x100: int = 100      # 0..300 means alpha=contrast_x100/100
    gamma_x100: int = 100         # 10..300 means gamma=gamma_x100/100

    # invert auto
    auto_invert: int = 0          # 0/1
    invert_thresh: int = 110      # 0..255

    # quant
    quant_enable: int = 0
    quant_mode: int = 0           # 0=BGR 1=HSV
    quant_levels: int = 8         # 2..64

    # denoise
    denoise_mode: int = 0         # 0=off 1=fastNlMeans
    nlm_h: int = 10               # 1..50
    nlm_template: int = 7         # 3..21 odd
    nlm_search: int = 21          # 7..51 odd

    # blur
    blur_mode: int = 3            # 0=off 1=Gaussian 2=Median 3=Bilateral
    gauss_k: int = 5              # odd 1..31
    gauss_sigma_x10: int = 14     # sigma=gauss_sigma_x10/10
    median_k: int = 5             # odd 1..31
    bil_d: int = 9
    bil_sc: int = 75
    bil_ss: int = 75

    # edges
    edge_mode: int = 1            # 0=off 1=Canny 2=SobelMag 3=Laplacian
    canny_low: int = 100
    canny_high: int = 200
    sobel_ksize: int = 3          # 1,3,5,7
    lap_ksize: int = 3

    # threshold (binary)
    thr_mode: int = 0             # 0=off 1=fixed 2=Otsu 3=AdaptiveMean 4=AdaptiveGauss
    thr_val: int = 128
    thr_inv: int = 0
    adp_block: int = 21           # odd >=3
    adp_C: int = 5                # -20..20

    # morphology
    morph_enable: int = 1
    morph_src: int = 1            # 0=edges 1=threshold 2=HSVmask 3=fused
    morph_op: int = 3             # 0=erode 1=dilate 2=open 3=close 4=gradient 5=tophat 6=blackhat
    morph_shape: int = 1          # 0=rect 1=ellipse 2=cross
    morph_k: int = 5              # 1..51 odd recommended
    morph_it: int = 1             # 1..10

    # HSV
    hsv_enable: int = 1
    h_min: int = 0
    h_max: int = 255
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255

    # fusion + post
    fuse_mode: int = 0            # 0=OR 1=AND 2=HSV only 3=Binary only
    cc_min_area: int = 0          # 0..5000
    fill_holes_enable: int = 0


# ----------------------------- processing core (unchanged logic) -----------------------------
def process(img_bgr: np.ndarray, cfg: dict) -> dict:
    # --- View transforms ---
    bgr = img_bgr.copy()

    # resize
    if cfg["resize_pct"] != 100:
        scale = cfg["resize_pct"] / 100.0
        bgr = cv.resize(
            bgr, None, fx=scale, fy=scale,
            interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
        )

    # rotate
    if cfg["rotate_deg"] % 360 != 0:
        h, w = bgr.shape[:2]
        M = cv.getRotationMatrix2D((w / 2, h / 2), cfg["rotate_deg"], 1.0)
        bgr = cv.warpAffine(bgr, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    # brightness/contrast: out = alpha*img + beta
    alpha = cfg["contrast"]
    beta = cfg["brightness"]
    bgr = cv.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    # gamma
    bgr = apply_gamma(bgr, cfg["gamma"])

    # --- auto invert ---
    gray0 = ensure_u8_gray(bgr)
    mean_val = float(gray0.mean())
    inverted = False
    if cfg["auto_invert"] and mean_val < cfg["invert_thresh"]:
        bgr = cv.bitwise_not(bgr)
        gray0 = ensure_u8_gray(bgr)
        inverted = True

    # --- quantization ---
    if cfg["quant_enable"]:
        levels = cfg["quant_levels"]
        step = max(1, 256 // levels)
        if cfg["quant_mode"] == 0:
            bgr = (bgr // step) * step
        else:
            hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] // step) * step
            hsv[:, :, 1] = (hsv[:, :, 1] // step) * step
            hsv[:, :, 2] = (hsv[:, :, 2] // step) * step
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # --- denoise ---
    if cfg["denoise_mode"] == 1:
        bgr = cv.fastNlMeansDenoisingColored(
            bgr, None,
            h=cfg["nlm_h"], hColor=cfg["nlm_h"],
            templateWindowSize=cfg["nlm_template"],
            searchWindowSize=cfg["nlm_search"],
        )

    # --- blur (works on gray for edge/threshold) ---
    gray = ensure_u8_gray(bgr)
    prep = gray.copy()

    if cfg["blur_mode"] == 1:  # Gaussian
        prep = cv.GaussianBlur(prep, (cfg["gauss_k"], cfg["gauss_k"]), cfg["gauss_sigma"])
    elif cfg["blur_mode"] == 2:  # Median
        prep = cv.medianBlur(prep, cfg["median_k"])
    elif cfg["blur_mode"] == 3:  # Bilateral
        prep = cv.bilateralFilter(prep, cfg["bil_d"], cfg["bil_sc"], cfg["bil_ss"])

    # --- edges ---
    edges = np.zeros_like(prep)
    if cfg["edge_mode"] == 1:
        edges = cv.Canny(prep, cfg["canny_low"], cfg["canny_high"])
    elif cfg["edge_mode"] == 2:
        k = cfg["sobel_ksize"]
        k = k if k in (1, 3, 5, 7) else 3
        gx = cv.Sobel(prep, cv.CV_32F, 1, 0, ksize=k)
        gy = cv.Sobel(prep, cv.CV_32F, 0, 1, ksize=k)
        mag = cv.magnitude(gx, gy)
        edges = cv.convertScaleAbs(mag)
        _, edges = cv.threshold(edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    elif cfg["edge_mode"] == 3:
        k = cfg["lap_ksize"]
        k = k if k in (1, 3, 5, 7) else 3
        lap = cv.Laplacian(prep, cv.CV_32F, ksize=k)
        edges = cv.convertScaleAbs(np.abs(lap))
        _, edges = cv.threshold(edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # --- threshold (binary mask) ---
    thr = np.zeros_like(prep)
    if cfg["thr_mode"] != 0:
        ttype = cv.THRESH_BINARY_INV if cfg["thr_inv"] else cv.THRESH_BINARY
        if cfg["thr_mode"] == 1:
            _, thr = cv.threshold(prep, cfg["thr_val"], 255, ttype)
        elif cfg["thr_mode"] == 2:
            _, thr = cv.threshold(prep, 0, 255, ttype + cv.THRESH_OTSU)
        elif cfg["thr_mode"] in (3, 4):
            method = cv.ADAPTIVE_THRESH_MEAN_C if cfg["thr_mode"] == 3 else cv.ADAPTIVE_THRESH_GAUSSIAN_C
            thr = cv.adaptiveThreshold(prep, 255, method, ttype, cfg["adp_block"], cfg["adp_C"])

    # --- HSV mask ---
    if cfg["hsv_enable"]:
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], dtype=np.uint8)
        upper = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]], dtype=np.uint8)
        hsv_mask = cv.inRange(hsv, lower, upper)
    else:
        hsv_mask = np.full(bgr.shape[:2], 255, np.uint8)

    # --- pick a "binary source" for fusion ---
    binary_default = thr if cfg["thr_mode"] != 0 else edges
    binary_default = (binary_default > 0).astype(np.uint8) * 255

    # --- fuse HSV + binary ---
    if cfg["fuse_mode"] == 0:  # OR
        fused = cv.bitwise_or(hsv_mask, binary_default)
    elif cfg["fuse_mode"] == 1:  # AND
        fused = cv.bitwise_and(hsv_mask, binary_default)
    elif cfg["fuse_mode"] == 2:  # HSV only
        fused = hsv_mask.copy()
    else:  # Binary only
        fused = binary_default.copy()

    # --- remove dots by area ---
    fused = connected_components_filter(fused, cfg["cc_min_area"])

    # --- fill holes ---
    if cfg["fill_holes"]:
        fused = fill_holes(fused)

    # --- morphology ---
    morph_out = fused.copy()
    if cfg["morph_enable"]:
        src_map = {0: edges, 1: thr, 2: hsv_mask, 3: fused}
        src = src_map.get(cfg["morph_src"], fused)
        src = (src > 0).astype(np.uint8) * 255

        shape = {0: cv.MORPH_RECT, 1: cv.MORPH_ELLIPSE, 2: cv.MORPH_CROSS}.get(cfg["morph_shape"], cv.MORPH_ELLIPSE)
        k = cfg["morph_k"]
        kernel = cv.getStructuringElement(shape, (k, k))
        it = cfg["morph_it"]

        op = cfg["morph_op"]
        if op == 0:
            morph_out = cv.erode(src, kernel, iterations=it)
        elif op == 1:
            morph_out = cv.dilate(src, kernel, iterations=it)
        elif op == 2:
            morph_out = cv.morphologyEx(src, cv.MORPH_OPEN, kernel, iterations=it)
        elif op == 3:
            morph_out = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel, iterations=it)
        elif op == 4:
            morph_out = cv.morphologyEx(src, cv.MORPH_GRADIENT, kernel, iterations=it)
        elif op == 5:
            morph_out = cv.morphologyEx(src, cv.MORPH_TOPHAT, kernel, iterations=it)
        else:
            morph_out = cv.morphologyEx(src, cv.MORPH_BLACKHAT, kernel, iterations=it)

    # --- final result ---
    result = cv.bitwise_and(bgr, bgr, mask=morph_out)

    return {
        "view": bgr,
        "gray": gray0,
        "prep": prep,
        "edges": edges,
        "thr": thr,
        "hsv_mask": hsv_mask,
        "fused": fused,
        "morph": morph_out,
        "result": result,
        "mean": mean_val,
        "inverted": inverted,
    }


# ----------------------------- Tk app -----------------------------
class TkPlayground(tk.Tk):
    STAGES = [
        ("View", "view"),
        ("Gray", "gray"),
        ("Preprocessed", "prep"),
        ("Edges", "edges"),
        ("Threshold", "thr"),
        ("HSV Mask", "hsv_mask"),
        ("Fused Mask", "fused"),
        ("Morph", "morph"),
        ("Result", "result"),
    ]

    def __init__(self, root: Path):
        super().__init__()
        self.title("OpenCV Playground (Tkinter)")
        self.root_path = root
        self.image_dir = root / "data" / "images"
        self.repo = ImageRepository(self.image_dir)
        self.paths = self.repo.list_images()
        if not self.paths:
            raise SystemExit(f"No images found under: {self.image_dir}")

        self.defaults = Defaults()
        self.idx = 0
        self.img0 = self._load_current()

        # layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.ctrl = ttk.Frame(self, padding=10)
        self.ctrl.grid(row=0, column=0, sticky="nsw")

        self.view = ttk.Frame(self, padding=10)
        self.view.grid(row=0, column=1, sticky="nsew")
        self.view.columnconfigure((0, 1, 2), weight=1)
        self.view.rowconfigure((0, 1, 2), weight=1)

        # status
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.ctrl, textvariable=self.status_var, wraplength=320).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # variables
        self.vars: Dict[str, tk.Variable] = {}
        self._build_controls()

        # image widgets
        self.max_tile_w = 420
        self.max_tile_h = 260
        self._tk_imgs: Dict[str, ImageTk.PhotoImage] = {}
        self._img_labels: Dict[str, ttk.Label] = {}
        self._build_tiles()

        # keybinds
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("p", lambda _e: self.print_status())
        self.bind("r", lambda _e: self.reset_controls())
        self.bind("<Right>", lambda _e: self.next_image())
        self.bind("<Left>", lambda _e: self.prev_image())

        # update loop
        self._updating = True
        self.after(0, self.update_loop)

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
            f"Keys: Esc quit | ←/→ prev/next | r reset | p print"
        )

    def _add_int(self, parent, key: str, label: str, lo: int, hi: int, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        v = tk.IntVar(value=getattr(self.defaults, key))
        self.vars[key] = v
        s = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", command=lambda _x: None)
        s.set(v.get())
        s.grid(row=row + 1, column=0, sticky="ew", pady=(2, 8))
        # keep scale <-> var synced
        def on_var(*_):
            s.set(v.get())
        def on_scale(val):
            v.set(int(float(val)))
        v.trace_add("write", on_var)
        s.configure(command=on_scale)
        ttk.Label(parent, textvariable=v).grid(row=row + 2, column=0, sticky="w", pady=(0, 6))

    def _add_bool(self, parent, key: str, text: str, row: int) -> None:
        v = tk.IntVar(value=getattr(self.defaults, key))
        self.vars[key] = v
        ttk.Checkbutton(parent, text=text, variable=v).grid(row=row, column=0, sticky="w", pady=(0, 6))

    def _add_combo(self, parent, key: str, label: str, values: List[str], row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        v = tk.StringVar(value=values[getattr(self.defaults, key)])
        self.vars[key] = v
        cb = ttk.Combobox(parent, values=values, textvariable=v, state="readonly")
        cb.grid(row=row + 1, column=0, sticky="ew", pady=(2, 8))

    def _build_controls(self) -> None:
        self.ctrl.columnconfigure(0, weight=1)
        self._set_status()

        nb = ttk.Notebook(self.ctrl)
        nb.grid(row=1, column=0, sticky="nsew")

        tab_view = ttk.Frame(nb, padding=8)
        tab_masks = ttk.Frame(nb, padding=8)
        tab_filters = ttk.Frame(nb, padding=8)
        tab_morph = ttk.Frame(nb, padding=8)
        for t in (tab_view, tab_masks, tab_filters, tab_morph):
            t.columnconfigure(0, weight=1)

        nb.add(tab_view, text="View")
        nb.add(tab_filters, text="Filters")
        nb.add(tab_masks, text="Masks")
        nb.add(tab_morph, text="Morph")

        r = 0
        self._add_int(tab_view, "resize_pct", "resize_% (10..200)", 10, 200, r); r += 3
        self._add_int(tab_view, "rotate_deg", "rotate_deg (0..360)", 0, 360, r); r += 3
        self._add_int(tab_view, "brightness", "brightness (-100..100)", -100, 100, r); r += 3
        self._add_int(tab_view, "contrast_x100", "contrast_x100 (0..300)", 0, 300, r); r += 3
        self._add_int(tab_view, "gamma_x100", "gamma_x100 (10..300)", 10, 300, r); r += 3
        self._add_bool(tab_view, "auto_invert", "auto_invert", r); r += 1
        self._add_int(tab_view, "invert_thresh", "invert_thresh (0..255)", 0, 255, r); r += 3

        # Filters
        r = 0
        self._add_bool(tab_filters, "quant_enable", "quant_enable", r); r += 1
        self._add_combo(tab_filters, "quant_mode", "quant_mode", ["BGR", "HSV"], r); r += 2
        self._add_int(tab_filters, "quant_levels", "quant_levels (2..64)", 2, 64, r); r += 3

        self._add_combo(tab_filters, "denoise_mode", "denoise", ["off", "fastNlMeans"], r); r += 2
        self._add_int(tab_filters, "nlm_h", "nlm_h (1..50)", 1, 50, r); r += 3
        self._add_int(tab_filters, "nlm_template", "nlm_template (odd 3..21)", 3, 21, r); r += 3
        self._add_int(tab_filters, "nlm_search", "nlm_search (odd 7..51)", 7, 51, r); r += 3

        self._add_combo(tab_filters, "blur_mode", "blur_mode", ["off", "Gaussian", "Median", "Bilateral"], r); r += 2
        self._add_int(tab_filters, "gauss_k", "gauss_k (odd 1..31)", 1, 31, r); r += 3
        self._add_int(tab_filters, "gauss_sigma_x10", "gauss_sigma_x10 (0..100)", 0, 100, r); r += 3
        self._add_int(tab_filters, "median_k", "median_k (odd 1..31)", 1, 31, r); r += 3
        self._add_int(tab_filters, "bil_d", "bil_d (1..25)", 1, 25, r); r += 3
        self._add_int(tab_filters, "bil_sc", "bil_sigmaColor (1..200)", 1, 200, r); r += 3
        self._add_int(tab_filters, "bil_ss", "bil_sigmaSpace (1..200)", 1, 200, r); r += 3

        self._add_combo(tab_filters, "edge_mode", "edge_mode", ["off", "Canny", "SobelMag", "Laplacian"], r); r += 2
        self._add_int(tab_filters, "canny_low", "canny_low (0..500)", 0, 500, r); r += 3
        self._add_int(tab_filters, "canny_high", "canny_high (0..500)", 0, 500, r); r += 3
        self._add_int(tab_filters, "sobel_ksize", "sobel_ksize (1/3/5/7)", 1, 7, r); r += 3
        self._add_int(tab_filters, "lap_ksize", "lap_ksize (1/3/5/7)", 1, 7, r); r += 3

        # Masks
        r = 0
        self._add_combo(tab_masks, "thr_mode", "thr_mode", ["off", "fixed", "Otsu", "AdaptiveMean", "AdaptiveGauss"], r); r += 2
        self._add_int(tab_masks, "thr_val", "thr_val (0..255)", 0, 255, r); r += 3
        self._add_bool(tab_masks, "thr_inv", "thr_inv", r); r += 1
        self._add_int(tab_masks, "adp_block", "adp_block (odd 3..101)", 3, 101, r); r += 3
        self._add_int(tab_masks, "adp_C", "adp_C (-20..20)", -20, 20, r); r += 3

        self._add_bool(tab_masks, "hsv_enable", "hsv_enable", r); r += 1
        self._add_int(tab_masks, "h_min", "H_min (0..255)", 0, 255, r); r += 3
        self._add_int(tab_masks, "h_max", "H_max (0..255)", 0, 255, r); r += 3
        self._add_int(tab_masks, "s_min", "S_min (0..255)", 0, 255, r); r += 3
        self._add_int(tab_masks, "s_max", "S_max (0..255)", 0, 255, r); r += 3
        self._add_int(tab_masks, "v_min", "V_min (0..255)", 0, 255, r); r += 3
        self._add_int(tab_masks, "v_max", "V_max (0..255)", 0, 255, r); r += 3

        self._add_combo(tab_masks, "fuse_mode", "fuse_mode", ["OR", "AND", "HSV only", "Binary only"], r); r += 2
        self._add_int(tab_masks, "cc_min_area", "cc_min_area (0..5000)", 0, 5000, r); r += 3
        self._add_bool(tab_masks, "fill_holes_enable", "fill_holes", r); r += 1

        # Morph
        r = 0
        self._add_bool(tab_morph, "morph_enable", "morph_enable", r); r += 1
        self._add_combo(tab_morph, "morph_src", "morph_src", ["Edges", "Threshold", "HSV mask", "Fused"], r); r += 2
        self._add_combo(tab_morph, "morph_op", "morph_op", ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"], r); r += 2
        self._add_combo(tab_morph, "morph_shape", "morph_shape", ["rect", "ellipse", "cross"], r); r += 2
        self._add_int(tab_morph, "morph_k", "morph_k (odd 1..51)", 1, 51, r); r += 3
        self._add_int(tab_morph, "morph_it", "morph_it (1..10)", 1, 10, r); r += 3

        # buttons
        btns = ttk.Frame(self.ctrl, padding=(0, 10, 0, 0))
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(btns, text="Prev (←)", command=self.prev_image).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Next (→)", command=self.next_image).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Reset (r)", command=self.reset_controls).grid(row=0, column=2, sticky="ew")

        ttk.Button(self.ctrl, text="Quit (Esc)", command=self.destroy).grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _build_tiles(self) -> None:
        # 3x3 grid of stages
        for i, (title, key) in enumerate(self.STAGES):
            r, c = divmod(i, 3)
            cell = ttk.Frame(self.view, padding=6)
            cell.grid(row=r, column=c, sticky="nsew")
            cell.columnconfigure(0, weight=1)
            cell.rowconfigure(1, weight=1)

            ttk.Label(cell, text=title).grid(row=0, column=0, sticky="w")
            lbl = ttk.Label(cell)
            lbl.grid(row=1, column=0, sticky="nsew")
            self._img_labels[key] = lbl

    def read_cfg(self) -> dict:
        # map UI vars -> processing cfg (same keys as your original read_cfg output)
        d: Dict[str, object] = {}

        d["resize_pct"] = clamp(int(self.vars["resize_pct"].get()), 10, 200)
        d["rotate_deg"] = clamp(int(self.vars["rotate_deg"].get()), 0, 360)
        d["brightness"] = clamp(int(self.vars["brightness"].get()), -100, 100)
        d["contrast"] = clamp(int(self.vars["contrast_x100"].get()), 0, 300) / 100.0
        d["gamma"] = clamp(int(self.vars["gamma_x100"].get()), 10, 300) / 100.0

        d["auto_invert"] = int(self.vars["auto_invert"].get()) == 1
        d["invert_thresh"] = clamp(int(self.vars["invert_thresh"].get()), 0, 255)

        d["quant_enable"] = int(self.vars["quant_enable"].get()) == 1
        d["quant_mode"] = 0 if self.vars["quant_mode"].get() == "BGR" else 1
        d["quant_levels"] = clamp(int(self.vars["quant_levels"].get()), 2, 64)

        d["denoise_mode"] = 0 if self.vars["denoise_mode"].get() == "off" else 1
        d["nlm_h"] = clamp(int(self.vars["nlm_h"].get()), 1, 50)
        d["nlm_template"] = clamp(int(self.vars["nlm_template"].get()) | 1, 3, 21)
        d["nlm_search"] = clamp(int(self.vars["nlm_search"].get()) | 1, 7, 51)

        blur_map = {"off": 0, "Gaussian": 1, "Median": 2, "Bilateral": 3}
        d["blur_mode"] = blur_map[self.vars["blur_mode"].get()]
        d["gauss_k"] = clamp(int(self.vars["gauss_k"].get()) | 1, 1, 31)
        d["gauss_sigma"] = clamp(int(self.vars["gauss_sigma_x10"].get()), 0, 100) / 10.0
        d["median_k"] = clamp(int(self.vars["median_k"].get()) | 1, 1, 31)
        d["bil_d"] = clamp(int(self.vars["bil_d"].get()), 1, 25)
        d["bil_sc"] = clamp(int(self.vars["bil_sc"].get()), 1, 200)
        d["bil_ss"] = clamp(int(self.vars["bil_ss"].get()), 1, 200)

        edge_map = {"off": 0, "Canny": 1, "SobelMag": 2, "Laplacian": 3}
        d["edge_mode"] = edge_map[self.vars["edge_mode"].get()]
        d["canny_low"] = clamp(int(self.vars["canny_low"].get()), 0, 500)
        d["canny_high"] = clamp(int(self.vars["canny_high"].get()), 0, 500)
        d["sobel_ksize"] = clamp(int(self.vars["sobel_ksize"].get()), 1, 7)
        d["lap_ksize"] = clamp(int(self.vars["lap_ksize"].get()), 1, 7)

        thr_map = {"off": 0, "fixed": 1, "Otsu": 2, "AdaptiveMean": 3, "AdaptiveGauss": 4}
        d["thr_mode"] = thr_map[self.vars["thr_mode"].get()]
        d["thr_val"] = clamp(int(self.vars["thr_val"].get()), 0, 255)
        d["thr_inv"] = int(self.vars["thr_inv"].get()) == 1
        d["adp_block"] = clamp(int(self.vars["adp_block"].get()) | 1, 3, 101)
        d["adp_C"] = clamp(int(self.vars["adp_C"].get()), -20, 20)

        d["hsv_enable"] = int(self.vars["hsv_enable"].get()) == 1
        h_min = clamp(int(self.vars["h_min"].get()), 0, 255)
        h_max = clamp(int(self.vars["h_max"].get()), 0, 255)
        s_min = clamp(int(self.vars["s_min"].get()), 0, 255)
        s_max = clamp(int(self.vars["s_max"].get()), 0, 255)
        v_min = clamp(int(self.vars["v_min"].get()), 0, 255)
        v_max = clamp(int(self.vars["v_max"].get()), 0, 255)
        d["h_min"], d["h_max"] = min(h_min, h_max), max(h_min, h_max)
        d["s_min"], d["s_max"] = min(s_min, s_max), max(s_min, s_max)
        d["v_min"], d["v_max"] = min(v_min, v_max), max(v_min, v_max)

        fuse_map = {"OR": 0, "AND": 1, "HSV only": 2, "Binary only": 3}
        d["fuse_mode"] = fuse_map[self.vars["fuse_mode"].get()]
        d["cc_min_area"] = clamp(int(self.vars["cc_min_area"].get()), 0, 5000)
        d["fill_holes"] = int(self.vars["fill_holes_enable"].get()) == 1

        d["morph_enable"] = int(self.vars["morph_enable"].get()) == 1
        src_map = {"Edges": 0, "Threshold": 1, "HSV mask": 2, "Fused": 3}
        d["morph_src"] = src_map[self.vars["morph_src"].get()]
        op_map = {"erode": 0, "dilate": 1, "open": 2, "close": 3, "gradient": 4, "tophat": 5, "blackhat": 6}
        d["morph_op"] = op_map[self.vars["morph_op"].get()]
        shape_map = {"rect": 0, "ellipse": 1, "cross": 2}
        d["morph_shape"] = shape_map[self.vars["morph_shape"].get()]
        d["morph_k"] = clamp(int(self.vars["morph_k"].get()) | 1, 1, 51)
        d["morph_it"] = clamp(int(self.vars["morph_it"].get()), 1, 10)

        return d

    def reset_controls(self) -> None:
        d = Defaults()
        for k, var in self.vars.items():
            if isinstance(var, tk.IntVar):
                var.set(int(getattr(d, k)))
            elif isinstance(var, tk.StringVar):
                # combos are handled by defaults index mapping; just set to current selected list value
                if k == "quant_mode":
                    var.set(["BGR", "HSV"][d.quant_mode])
                elif k == "denoise_mode":
                    var.set(["off", "fastNlMeans"][d.denoise_mode])
                elif k == "blur_mode":
                    var.set(["off", "Gaussian", "Median", "Bilateral"][d.blur_mode])
                elif k == "edge_mode":
                    var.set(["off", "Canny", "SobelMag", "Laplacian"][d.edge_mode])
                elif k == "thr_mode":
                    var.set(["off", "fixed", "Otsu", "AdaptiveMean", "AdaptiveGauss"][d.thr_mode])
                elif k == "fuse_mode":
                    var.set(["OR", "AND", "HSV only", "Binary only"][d.fuse_mode])
                elif k == "morph_src":
                    var.set(["Edges", "Threshold", "HSV mask", "Fused"][d.morph_src])
                elif k == "morph_op":
                    var.set(["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"][d.morph_op])
                elif k == "morph_shape":
                    var.set(["rect", "ellipse", "cross"][d.morph_shape])

    def print_status(self) -> None:
        cfg = self.read_cfg()
        p = self.paths[self.idx]
        print("\n" + "=" * 60)
        print(f"Image: {p.name}")
        for k in sorted(cfg.keys()):
            print(f"{k}: {cfg[k]}")
        print("=" * 60 + "\n")

    def next_image(self) -> None:
        self.idx = (self.idx + 1) % len(self.paths)
        self.img0 = self._load_current()
        self._set_status()

    def prev_image(self) -> None:
        self.idx = (self.idx - 1) % len(self.paths)
        self.img0 = self._load_current()
        self._set_status()

    def update_loop(self) -> None:
        if not self._updating:
            return

        cfg = self.read_cfg()

        # normalize odd kernel sizes here (matches your original logic)
        cfg["nlm_template"] = clamp(cfg["nlm_template"] | 1, 3, 21)
        cfg["nlm_search"] = clamp(cfg["nlm_search"] | 1, 7, 51)
        cfg["gauss_k"] = clamp(cfg["gauss_k"] | 1, 1, 31)
        cfg["median_k"] = clamp(cfg["median_k"] | 1, 1, 31)
        cfg["adp_block"] = clamp(cfg["adp_block"] | 1, 3, 101)
        cfg["morph_k"] = clamp(cfg["morph_k"] | 1, 1, 51)

        out = process(self.img0, cfg)

        # render tiles
        for _title, key in self.STAGES:
            tk_img = to_tk_image(out[key], self.max_tile_w, self.max_tile_h)
            self._tk_imgs[key] = tk_img  # keep ref
            self._img_labels[key].configure(image=tk_img)

        # run again
        self.after(30, self.update_loop)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    app = TkPlayground(root)
    app.mainloop()


if __name__ == "__main__":
    main()