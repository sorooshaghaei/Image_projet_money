# Dataset Scan and OpenCV Method Report

Date: 2026-02-19

## Scope
- Scanned all files in `data/images`: 106 images.
- File formats: 54 `.jpg`, 22 `.jpeg`, 18 `.png`, 12 `.webp`.
- Group distribution: `gp1` 14, `gp2` 15, `gp3` 10, `gp4` 10, `gp5` 25, `gp6` 10, `gp7` 12, `gp8` 10.

Scan outputs are saved in:
- `report/dataset_image_scan.csv`

---

## Dataset Findings From Full Scan

## 1) Image variability
- Width range: 920 to 5712 px (median 2313 px).
- Height range: 864 to 5712 px (median 2343 px).
- Aspect ratio range: 0.504 to 1.777.
- Brightness and contrast vary strongly:
  - gray mean range: 46.14 to 184.96
  - gray std range: 13.49 to 56.48
  - dynamic range (p95-p5) range: 41.80 to 184.81

## 2) Background homogeneity proxy
Using border-zone statistics (`border_cv = border_std / border_mean`):
- dataset mean: 0.2269
- p50: 0.1699
- p90: 0.3842
- max: 0.7805

Groups with less homogeneous backgrounds (higher `border_cv`):
- `gp3` median high and several extreme cases (`3_6.jpg`, `3_7.jpg`, `3_8.jpg`).
- `gp7` generally textured/dark backgrounds.
- part of `gp5` has reflections and wood grain.

More homogeneous examples:
- several `gp4`, `gp8`, and parts of `gp2`.

## 3) Annotation/data consistency check
`DatasetRepository.DATA_ROWS` has 107 rows, but 9 rows do not resolve to a real image path:
- `gp1/exemple1.png`
- `gp5/7.jpeg`
- `grp5/2e01.jpg`
- `grp5/3e19.jpg`
- `grp5/4.17.jpg`
- `grp5/4e22.jpg`
- `grp5/6e19.jpg`
- `grp5/8e88.jpg`
- `grp5/10e05.jpg`

This affects metric reliability, because rows are silently skipped when path resolution fails.

---

## Which OpenCV detection method is easier/better for coins?

For our current dataset, the best practical strategy is the current **ensemble**:
- strict Hough (`cv2.HoughCircles`) for precision,
- loose Hough for recall recovery,
- contour fallback for hard edge cases,
- then support filtering and overlap suppression.

This is exactly how `src/processor_circles.py` is structured.

Method comparison:

1. Hough circles (geometry-first)
- Pros:
  - direct circular prior,
  - good under moderate lighting/background changes,
  - strong baseline for coin counting.
- Cons:
  - sensitive to parameter tuning (`dp`, `minDist`, `param1`, `param2`, radius range),
  - weak when edges are low contrast, heavily blurred, or partially occluded.

2. Contour + circularity filtering
- Pros:
  - can recover misses when Hough is conservative,
  - more interpretable geometric filtering (area, circularity, fill ratio).
- Cons:
  - fragile with broken edges, texture-rich backgrounds, touching coins.

3. Threshold + connected components
- Pros:
  - simple, fast, easy to explain.
- Cons:
  - poor robustness to shadows, reflections, textured/non-homogeneous backgrounds,
  - merges touching coins.

Conclusion:
- No single classic OpenCV method is best for all images.
- For this dataset, **Hough-centered ensemble is the best general OpenCV choice**.

---

## important questions

## 1) Do we have a general best algorithm?
- For our dataset and classical CV, yes: **an ensemble around Hough circles is the best general approach**, not a single pure method.
- There is no universal one-method winner across all acquisition conditions.

## 2) Could we add to count and value accuracy? How?
Yes:
- Add image quality gating before detection (background homogeneity, blur, saturation, glare checks).
- Add adaptive parameter selection per image (instead of one global Hough configuration).
- Improve touching-coin separation (distance transform + watershed on coin mask).
- Calibrate scale using a reference object or camera distance constraints (stabilizes denomination by diameter).
- Add stronger denomination classification (coin patch classifier) and fuse with scale-fit output.
- Keep uncertainty instead of forced labels; use value bounds when low confidence.

## 3) In real life system sends error for non-homogeneous background, why did not we do that?
- Current code has no explicit non-homogeneous background rejection rule.
- `src/processor.py` does preprocessing and detection directly; no quality gate.
- `src/processor_circles.py` filters circle support, but this is not a background-quality validator.
- So the pipeline tries detection on all images instead of failing fast when scene quality is poor.

## 4) Is OpenCV enough for 100% accuracy, or must we rely on ML?
- OpenCV alone is usually not enough for true 100% in uncontrolled real-world capture.
- In controlled capture (fixed camera/light/background), classical CV can be very strong.
- For unconstrained scenes, ML is usually needed for last-mile robustness, especially denomination classification under glare, wear, blur, and color shift.

## 5) Why changing Hough params fixes complex background but ruins simple-background detection?
- It is a precision/recall tradeoff:
  - stricter settings reduce false positives on textured backgrounds,
  - but also remove weak/partial true circle edges on simpler but low-contrast images.
- Example behavior:
  - increasing `param2` or `minDist` helps reject clutter,
  - but increases misses when true coin boundaries are weak.

## 6) Other useful methods besides Hough + HSV? Pros and cons?
Yes:

1. Distance transform + watershed (for touching coins)
- Pros: separates merged blobs better.
- Cons: sensitive to seed quality and thresholding.

2. Ellipse fitting / RANSAC circle fitting on edges
- Pros: can handle mild perspective and partial arcs.
- Cons: computationally heavier, brittle with noisy edges.

3. Template matching (multi-scale)
- Pros: easy for constrained datasets.
- Cons: poor generalization to scale/rotation/lighting/wear variance.

4. Keypoint-based matching (ORB/SIFT-like features)
- Pros: can use coin-face texture details.
- Cons: weak on worn/blurred coins and small patches.


## 7) Why are labeled coverage/recall/precision/F1 high, but MAE low and value metrics terrible?
Main reasons:
- Count metrics and value metrics measure different tasks.
- Current recall/precision are count-level approximations, not localization-matched detection metrics:
  - `matched_count_sum += min(true_count, pred_count)` in `RunStats.update`.
  - This can look high even when detected circles are wrong identities.
- `Labeled Coverage` is only labeled/detected ratio, not label correctness.
- Value depends on correct denomination per coin; small count errors can create large value errors.
- Unlabeled coins (`None`) are excluded from value sum (`estimated_value_eur`), which can under-estimate value.
- Relative value error is normalized by total true value; low-value scenes amplify percentage error.

## 9) How to improve this system for this dataset?
Priority order:

1. Data/annotation cleanup first
- Fix the 9 unresolved annotation rows.
- Ensure all ground-truth rows map to actual files.

2. Add scene-quality gate
- Reject or warn on non-homogeneous background and severe glare before counting.

3. Adaptive detector policy
- Use image-conditioned parameter sets (simple vs textured backgrounds).
- Keep ensemble, but select strictness automatically.

4. Better denomination stage
- Add coin-crop classifier (ML or stronger handcrafted features) and fuse with scale-fit.

5. Better evaluation protocol
- Add per-coin GT (centers/radii + denomination labels), confusion matrix, and proper detection matching.

## 10) Why are accuracy measurements not reliable? How to improve?
Current limitations in measurement design:
- Count precision/recall/F1 are proxy formulas from counts only, not detection matching.
- Label coverage is not label accuracy.
- General accuracy averages detection recall with value accuracy; mixed semantics.
- Missing images are silently skipped, creating selection bias.
- No per-coin denomination ground truth in current table; value can be wrong but hard to diagnose.

How to fix:
- Use detection matching (e.g., center-distance/IoU matching) to compute TP/FP/FN.
- Add per-coin label GT and report denomination confusion matrix.
- Report per-group metrics (`gp1..gp8`) and stratify by scene difficulty.
- Add confidence intervals/bootstrapped uncertainty for metrics.
- Fail evaluation if annotation rows are unresolved, instead of skipping silently.

---

## Code references used for this analysis
- Circle ensemble detection and support filtering:
  - `src/processor_circles.py`
- Preprocessing and label-to-value behavior:
  - `src/processor.py`
- Scale fitting and soft color penalty:
  - `src/processor_scale.py`
- Color group scoring and denomination candidate logic:
  - `src/processor_color.py`
- Metric definitions and aggregation:
  - `src/runner.py`
- Path resolution behavior:
  - `src/io_utils.py`
- Annotation table:
  - `src/dataset.py`
