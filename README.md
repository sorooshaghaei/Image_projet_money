# Image_projet_money

VMI - Analyse d'Image project for euro coin counting and value estimation from images using OpenCV.

## Team

- Maksym DOLHOV
- Mehdi AGHAEI
- Nima DAVARI

## Repository Layout

```text
.
├── data/
│   └── images/                          # Dataset root (default input)
├── presentation/
│   └── overleaf_upload/
│       ├── slides.tex
│       ├── OVERLEAF_README.txt
│       └── assets/generated/*
├── reports/
│   └── evaluation_rows.csv              # Default per-image report
├── src/
│   ├── app/                             # CLI
│   ├── common/                          # IO, formatting, plotting, debug export
│   ├── data/                            # Dataset listing + ground truth repository
│   ├── evaluation/                      # Metrics + CSV reporting
│   ├── pipeline/                        # Preprocessing, detection, valuation
│   └── ui/                              # Interactive debug viewer
├── main.py                              # Entry point
├── PIPELINE.md                          # Detailed processing stages
└── README.md
```

## Requirements

- Python 3.10+
- Packages: `numpy`, `opencv-python`, `matplotlib`, `rich`

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python matplotlib rich
```

## Run

Default run (process + evaluate + viewer):

```bash
python main.py
```

Headless run:

```bash
python main.py --no-view
```

Useful examples:

```bash
# Quick sample
python main.py --limit 20 --no-view

# Evaluate selected groups only
python main.py --eval-groups gp6 gp8 --no-view
python main.py --eval-groups gp6,gp8 --no-view

# Choose Hough preset (test1, test2, test3)
python main.py --preset test2 --no-view

# Save pipeline figures
python main.py --save-dir outputs/pipeline --no-view

# Auto-export debug snapshots (.json + .txt)
python main.py --debug-export-dir debug_exports --no-view

# Custom dataset/report path
python main.py --dataset-dir data/images --evaluation-report reports/evaluation_rows.csv --no-view
```

CLI definition: `src/app/cli.py`.

## Outputs

- Console tables for per-image predictions and dataset metrics
- CSV report: `reports/evaluation_rows.csv` (or `--evaluation-report`)
- Optional pipeline figures: `--save-dir`
- Optional debug exports: `--debug-export-dir`

Ground truth is loaded from `src/data/ground_truth.py`.
Images without annotations are reported as `SKIP_NO_GT`.

## Viewer Controls

- `Right`, `d`, `n`, `Space`: next image
- `Left`, `a`, `p`: previous image
- `Up`, `w`: next step
- `Down`, `s`: previous step
- `f`: toggle full/final-only mode
- `c` or `x`: export current debug payload
- `q` or `Esc`: quit viewer

## Notes

- Runtime defaults are in `src/pipeline/config.py`.
- Current preprocessing path is intentionally compact: grayscale + blur.
- The full technical flow is described in `PIPELINE.md`.

## License

Academic project; see `LICENSE`.
