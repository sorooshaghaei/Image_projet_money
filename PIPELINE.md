# Processing Pipeline

This document describes the implemented runtime pipeline used by `main.py`.

## 1. Runtime Entry Flow

Main execution path:

1. `main.py`
2. `src/app/cli.py` (`AppRunner`)
3. `src/pipeline/orchestrator.py` (`Analyzer`)

Per image, the system runs:

1. Input read + letterbox normalization
2. Preprocessing
3. Circle detection
4. Color/material analysis
5. Denomination and total-value estimation
6. Evaluation/reporting
7. Optional interactive viewer and debug export

## 2. Input and Normalization

Files:

- `src/common/image_io.py`
- `src/pipeline/orchestrator.py`

Steps:

- Read image in BGR
- Resize with letterbox to fixed canvas (`640x480` by default)

Goal: keep detector behavior stable across mixed image sizes.

## 3. Preprocessing

File: `src/pipeline/preprocessing.py`

Current active preprocessing:

- BGR -> grayscale
- Blur (`gauss` or `median`)

Notes:

- Blur kernel size is normalized to an odd positive integer.
- No additional CLAHE/histogram normalization stage is used in current runtime.

## 4. Circle Detection (Hough + Radius Sweep)

File: `src/pipeline/detectors/circle_detection/coin_detector.py`

Main logic:

1. Estimate Hough `param1` from Scharr gradient statistics (`auto_hough_param1_from_gradient`)
2. Sweep `minRadius` over a configured interval
3. Score each result with geometric penalties (concentric duplicates, nesting, intrusion)
4. Select `minRadius` using plateau-like voting on stable counts
5. Run final `cv2.HoughCircles` and build debug overlay

Outputs:

- Detected circles `(x, y, r)`
- Circle count
- Effective Hough parameters
- Sweep diagnostics (`plateau_debug`, `sweep_results`)

## 5. Color and Material Analysis

File: `src/pipeline/detectors/color_analysis/coin_analyzer.py`

For each detected circle:

1. Build full/inner/border masks
2. Compute robust HSV/Lab statistics
3. Extract color + structural cues
4. Classify coin appearance:
   - `one-color-like` (bronze/gold tendency)
   - `bi-metal-like` (1e/2e tendency)
   - `uncertain`

Supported material modes (`analysis_material_mode`):

- `hsv`
- `hsv_kmeans`
- `lab_proto` (default)

Detailed per-coin diagnostics are stored in `split_stats`.

## 6. Denomination and Value Estimation

Files:

- `src/pipeline/detectors/valuation/coin_value_estimator.py`
- `src/pipeline/detectors/valuation/value_estimator.py`

Process:

1. Infer family (`bronze`, `gold`, `bimetal`, `unknown`)
2. Estimate scale (`px_per_mm`) using bimetal references plus fallback voting
3. Cluster radii by family
4. Score denomination candidates
5. Produce per-coin predictions, denomination counts, and total cents

Outputs include:

- `predictions` (per coin)
- `counts` by denomination
- `total_cents`
- `scale_info` and `family_models` for debugging

## 7. Evaluation and Reporting

Files:

- `src/evaluation/metrics.py`
- `src/evaluation/reporting.py`
- `src/data/ground_truth.py`

Metrics:

- Coin exact-match accuracy
- Value MAE and MSE
- Per-group summaries

Default CSV report:

- `reports/evaluation_rows.csv`

Rows without matching annotation are marked `SKIP_NO_GT`.

## 8. Viewer and Debug Export

Files:

- `src/ui/debug_viewer.py`
- `src/common/debug_export.py`

Viewer features:

- Navigate image list and pipeline steps
- Toggle final-only/full pipeline view
- Export current state (`.json` + `.txt`) with key `c`/`x`

Debug export payload includes:

- Relative path and viewer metadata
- Coin/value metrics
- Hough parameters
- Per-coin predictions
- Raw debug payload (`split_stats`, `scale_info`, etc.)

## 9. Main Configuration

File: `src/pipeline/config.py`

Main defaults:

- Dataset dir: `data/images`
- Target size: `640x480`
- Hough preset: `test1`
- Blur mode: `gauss`
- Material mode: `lab_proto`
- Viewer default mode: final-only

All runtime parameters are centralized in `PipelineConfig`.
