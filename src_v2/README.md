# src_v2 (second_experiment)

This package runs a scene-aware OpenCV policy:

1. Label background difficulty: `easy`, `medium`, `difficult`.
2. Route method per image:
- clean background -> `contours`
- textured / medium-difficult background -> `hough`
- touching/overlap evidence -> `hough+watershed` (or `watershed` when forced)

## Run

```bash
.venv/bin/python main_v2.py --path data/images --short-root data/images --mode auto
```

Terminal output prints, for each image, the selected method and scene label.

## Save per-image trace CSV

```bash
.venv/bin/python main_v2.py --path data/images --short-root data/images --mode auto --csv report/runtime_v2_policy_trace.csv
```

## Open visualizer

```bash
.venv/bin/python main_v2.py --path data/images/gp5 --short-root data/images --mode auto --max-images 20 --visualize
```

Visualizer controls:
- mode: `auto`, `contours`, `hough`, `watershed`, `hybrid`
- sliders: Hough `param2`, Hough `minDist`, contour circularity, watershed foreground ratio
- keyboard: `left/right` or `a/d` to navigate images
