# Coin Detection Pipeline
Current implementation overview for the euro coin project.

This document describes the **implemented pipeline** (not only a plan), with references to source modules.

## 1. Objective
For each input image:
- detect all visible coins,
- estimate denomination for each coin,
- compute total value in EUR,
- expose debug artifacts for analysis.

## 2. High-Level Flow
```text
Input BGR image
  -> Resize to fixed width
  -> Grayscale normalize (+ optional inversion if dark)
  -> Median blur
  -> Circle ensemble detection
       (strict Hough + loose Hough + contour backup)
  -> Per-coin center/ring color features in LAB/HSV
  -> Color-group scoring and denomination candidates
  -> Global scale fit (px/mm) across all detected coins
  -> Final per-coin denomination + total value
```

Main orchestrator:
- `src/processor.py` (`CoinProcessor`)

## 3. Stage-by-Stage Details
### 3.1 Preprocessing
Module: `src/processor.py`
- Resize image to `DetectionConfig.TARGET_WIDTH`.
- Convert to grayscale and normalize to `[0, 255]`.
- If mean brightness is low, apply inversion.
- Apply median blur (`BLUR_KERNEL_SIZE`) for robust circle gradients.

Why:
- normalizes camera variability,
- improves circle detector stability,
- keeps edges less noisy than raw grayscale.

### 3.2 Circle Detection (Geometry)
Module: `src/processor_circles.py` (`CircleDetector`)

Ensemble strategy:
- strict Hough pass (high precision),
- loose Hough pass (recall recovery),
- contour proposals as backup.

Post-processing:
- merge near-duplicate circles,
- support filtering with:
  - angular edge coverage on ring,
  - inner/outer contrast score,
- nested-overlap suppression to remove redundant inner circles.

Why:
- pure single-pass Hough is unstable across all groups,
- ensemble + support filtering improves robustness.

### 3.3 Color Features and Priors
Module: `src/processor_color.py` (`CoinColorClassifier`)

Per detected circle:
- build masks:
  - full coin,
  - center region,
  - ring annulus,
- compute LAB material scores (bronze/gold/silver),
- compute region HSV/LAB stats (saturation/chroma/hue/yellow/silver cues),
- estimate bimetal confidence and directional evidence (1 EUR vs 2 EUR tendency).

Output:
- color-group scores,
- candidate denomination list used as a soft prior.

Why:
- 1c/2c/5c and 10c/20c/50c can be guided by color family,
- 1 EUR / 2 EUR require center-vs-ring evidence.

### 3.4 Denomination by Global Scale Fit
Module: `src/processor_scale.py` (`ScaleValueClassifier`)

Method:
- generate scale hypotheses from observed diameters and official euro diameters,
- score each hypothesis with per-coin relative error,
- apply soft penalty when denomination falls outside color candidates,
- optionally refine scale via least squares.

Why:
- one global scale enforces physical consistency across all coins in one image,
- avoids independent per-coin assignments that conflict.

## 4. Outputs
Data structure: `src/models.py` (`PipelineResult`)

Key fields:
- `coin_count`
- `coin_labels`
- `coin_color_labels`
- `coin_candidate_denoms`
- `estimated_value_eur`
- `steps` (visual debug pipeline stages)

## 5. Configurable Parameters
Module: `src/config.py` (`DetectionConfig`)

Groups of parameters:
- preprocessing: blur kernel, target width,
- Hough detection: `HOUGH_*`,
- contour fallback: `CONTOUR_*`,
- candidate merge/suppression: `MERGE_*`,
- support thresholds: `CIRCLE_*`, `LOOSE_*`.

## 6. Evaluation Entry Point
Batch app:
- `main.py` -> `src/runner.py` (`PipelineApp`)

Evaluation uses:
- annotation table from `src/dataset.py`,
- image resolution via `src/io_utils.py` (`ImagePathResolver`),
- summary metrics for count and value errors.

## 7. Annotation Contract
Current ground truth source:
- `src/dataset.py` (`DatasetRepository.DATA_ROWS`)

For this project size, no separate external annotation schema is required.
Keep image-level ground truth directly in `DatasetRepository.DATA_ROWS`.

## 8. Known Limitations
- very strong reflections and heavy occlusions still reduce reliability,
- touching coins can still be hard in some scenes,
- cross-device color shifts may affect color priors.

## 9. Next Improvements
- integrate robust specular highlight rejection in color stats,
- add stronger touching-coin splitting fallback,
- expand benchmark reporting by group and lighting condition,
- optionally migrate annotations to CSV only if dataset complexity increases.
