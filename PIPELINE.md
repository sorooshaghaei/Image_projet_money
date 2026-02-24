# PIPELINE.md
**Image Processing and Analysis Pipeline – TER M1 (VMI)**

This document describes the current implemented pipeline used by the CLI (`main.py`), not a planned draft.

---

## 1. Global Pipeline Overview

Runtime flow for one image:

1. Load image and letterbox-resize to `640x480`
2. Preprocess (grayscale + configurable blur)
3. Detect circles with Hough + automatic `minRadius` sweep
4. Analyze coin color/material (one-color, bi-metal, uncertain)
5. Estimate denomination/value from per-coin stats
6. Aggregate metrics and optionally visualize/export debug artifacts

Orchestrator entrypoint: `src/pipeline/orchestrator.py`.

---

## 2. Stage A – Input Normalization

File: `src/pipeline/orchestrator.py`

- Reads image as BGR
- Resizes with letterbox to fixed canvas (`target_width`, `target_height`)

Why:

- Keeps detector thresholds stable across mixed input resolutions
- Makes evaluation/debug comparisons consistent

---

## 3. Stage B – Preprocessing

File: `src/pipeline/preprocessing.py`

Current effective preprocessing:

1. Convert BGR to grayscale
2. Apply blur (`gauss` or `median`)

Notes:

- CLAHE and histogram normalization fields exist in config for compatibility/debug payload, but are currently disabled by default.
- Blur stage name is propagated to viewer (`Gaussian Blur` / `Median Blur`).

---

## 4. Stage C – Circle Detection

File: `src/pipeline/detectors/circle_detection/coin_detector.py`

### 4.1 Adaptive `param1` estimation

`param1` (Canny high threshold inside Hough) is inferred from Scharr gradient magnitude percentiles (`auto_hough_param1_from_gradient`).

### 4.2 Automatic `minRadius` selection

The detector sweeps `minRadius` over a configured range and scores each candidate using geometric penalties:

- concentric duplicates
- nested circles
- severe intrusions

Then it selects a robust candidate via plateau-like voting (or best fallback when no clean candidate exists).

### 4.3 Final detection + overlay

Final Hough run outputs:

- circles (`x, y, r`)
- `circle_count`
- debug overlay image
- sweep diagnostics (`plateau_debug`, `sweep_results`)

---

## 5. Stage D – Color/Material Analysis

File: `src/pipeline/detectors/color_analysis/coin_analyzer.py`

Per detected circle:

1. Split pixels into full / inner / border masks
2. Build a stable material sample ring (`0.45R..0.80R`) and filter V extremes
3. Compute robust HSV center (circular hue + median S/V)
4. Compute material cues and structural cues:
   - HSV delta terms
   - radial k-means agreement
   - radial step score
   - edge roughness score

### Material modes

Configured by `analysis_material_mode`:

- `hsv`: direct heuristic HSV labeling
- `hsv_kmeans`: per-coin HSV k-means (`k in {1,2}` auto-selected)
- `lab_proto` (default): image-level Lab prototypes + scene-level Lab clustering, fused by confidence

Output labels include:

- `one-color-like/bronze`
- `one-color-like/gold`
- `1-euro-like`, `2-euro-like`
- `uncertain`

Rich per-coin diagnostics are saved into `split_stats`.

---

## 6. Stage E – Denomination and Value Estimation

Files:

- `src/pipeline/detectors/valuation/coin_value_estimator.py`
- `src/pipeline/detectors/valuation/value_estimator.py`

Process:

1. Infer family (`bronze`, `gold`, `bimetal`, `unknown`) from analyzed coin row
2. Estimate `px_per_mm` scale (bimetal reference + fallback voting)
3. Cluster radii per family
4. Score denomination probabilities
5. Produce:
   - per-coin best denomination/probability
   - family models and scale debug info
   - denomination counts
   - total value in cents

---

## 7. Stage F – Evaluation and Viewer

CLI file: `src/app/cli.py`  
Viewer file: `src/ui/debug_viewer.py`

Evaluation:

- Coin exact-match accuracy
- Value MAE / MSE
- Per-group summary
- CSV report (`reports/evaluation_rows.csv` by default)

Viewer:

- Browse images and pipeline steps
- Toggle final-only mode
- Export current debug snapshot (`.json` + `.txt`)

---

## 8. Debug Export Format

File: `src/common/debug_export.py`

Export payload contains:

- source/relative path
- current viewer step metadata
- coin/value metrics
- Hough params
- compact per-coin predictions
- full raw debug payload (`split_stats`, family models, scale info, etc.)

Saved under `debug_exports/<group>/` as:

- `<stem>_debug_final.json|txt` (final-only mode)
- `<stem>_debug_sXX.json|txt` (full pipeline mode)

---

## 9. Main Configuration Knobs

File: `src/pipeline/config.py`

Key defaults:

- Hough preset: `test1`
- blur mode: `gauss`
- `analysis_bimetal_mode`: `hybrid`
- `analysis_material_mode`: `lab_proto`
- viewer default: final-only

All values are centralized in `PipelineConfig` and injected through the orchestrator.
