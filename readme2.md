# src_v2 (second_experiment)

This package runs a scene-aware OpenCV policy:

1. Label background difficulty: `easy`, `medium`, `difficult`.
2. Route method per image:
- clean background -> `contours`
- textured / medium-difficult background -> `hough`
- touching/overlap evidence -> `hough+watershed` (or `watershed` when forced)

## Structure

```text
src_v2/
  runner.py               # app entrypoint + scan/dataset pipeline + metrics summary
  preprocessing.py        # image preprocessing (resize/gray/edges/mask)
  analyzer.py             # hybrid algorithm orchestrator
  detectors.py            # contours/hough/watershed detectors
  policy.py               # background policy + method routing
  visualizer.py           # interactive visual analysis UI
  dataset.py              # dataset rows (ground-truth count/value/group)
  config.py               # runtime and algorithm settings
```

## Run

```bash
.venv/bin/python main_v2.py --scan --path data/images --short-root data/images --mode auto
```

Terminal output prints, for each image, the selected method and scene label.

## Save per-image trace CSV

```bash
.venv/bin/python main_v2.py --scan --path data/images --short-root data/images --mode auto --csv report/runtime_v2_policy_trace.csv
```

## Evaluate Against Dataset (count + value + metrics)

```bash
.venv/bin/python main_v2.py --evaluate-dataset --mode auto --csv report/runtime_v2_dataset_eval.csv
```

This compares each prediction to `src_v2/dataset.py` and prints:
- count match + count error
- predicted vs true value (EUR)
- precision, recall, F1, count MAE
- labeled coverage
- value MAE, value relative error, value accuracy

## Open visualizer

```bash
.venv/bin/python main_v2.py --scan --path data/images/gp5 --short-root data/images --mode auto --max-images 20 --visualize
```

Visualizer controls:
- mode: `auto`, `contours`, `hough`, `watershed`, `hybrid`
- sliders: Hough `param2`, Hough `minDist`, contour circularity, watershed foreground ratio
- keyboard: `left/right` or `a/d` to navigate images
