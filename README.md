# Image_projet_money
Maksym DOLHOV, Mehdi AGHAEI and Nima DAVARI  
## TER M1

This repository contains the TER (Travail d'Étude et de Recherche) project for M1 VMI (Analyse d'Image), focused on euro coin detection and value estimation from images.

---

## Repository Structure

```bash
.
├── data/
│   ├── images/             # Dataset images (grouped folders like gp1..gp8)
│   └── annotations/        # Optional extra annotations
├── debug_exports/          # Optional JSON/TXT debug dumps
├── reports/                # Evaluation CSV outputs
├── src/
│   ├── app/                # CLI entry logic
│   ├── common/             # Plotting, formatting, IO, debug export
│   ├── data/               # Dataset listing + ground truth repository
│   ├── evaluation/         # Metrics + reporting
│   ├── pipeline/           # Preprocessing + detectors + orchestrator
│   └── ui/                 # Interactive debug viewer
├── main.py                 # Root CLI entrypoint
├── PIPELINE.md             # Detailed processing pipeline
└── README.md
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Install project dependencies:

```bash
pip install -r requirements.txt
```

Python `3.10+` is recommended.

---

## Run

### Default run (process all images + evaluation + viewer)

```bash
python main.py
```

### Headless run

```bash
python main.py --no-view
```

### Useful options

```bash
# Quick debug sample
python main.py --limit 20 --no-view

# Evaluate only selected groups
python main.py --eval-groups gp6 gp8 --no-view

# Same as above using comma-separated groups
python main.py --eval-groups gp6,gp8 --no-view

# Choose Hough preset
python main.py --preset test2

# Save rendered pipeline figures
python main.py --save-dir outputs/pipeline --no-view

# Auto-export debug snapshots (json/txt)
python main.py --debug-export-dir debug_exports --no-view
```

CLI options are defined in `src/app/cli.py`.

---

## Evaluation Outputs

- Per-image report CSV: `reports/evaluation_rows.csv` (configurable via `--evaluation-report`)
- Console summary includes:
  - coin-count exact-match accuracy
  - value MAE/MSE
  - group-by-group breakdown

Ground truth is loaded from `src/data/ground_truth.py`.  
Images without ground truth are reported as `SKIP_NO_GT`.

### Current Project Result (snapshot)

```text
      Coin Number Estimation       
╭────────────────────────┬────────╮
│ Accuracy (exact match) │ 76.19% │
│ Correct                │ 80/105 │
╰────────────────────────┴────────╯
                   Value Estimation                   
╭───────────┬────────────────────────────────────────╮
│ Evaluated │                                    103 │
│ MAE       │          1.791 EUR/image (179.1 cents) │
│ MSE       │ 15.3455 EUR^2/image (153454.9 cents^2) │
╰───────────┴────────────────────────────────────────╯
```

### Current Project Result By Group (snapshot)

```text
  GROUP   EVAL   COIN ACC   VAL EVAL   VAL PREC   VAL REC   VAL F1   VAL MAE (EUR)   VAL MSE (EUR^2)
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  gp1       14     78.57%         14     88.70%    82.73%   85.61%            0.83            1.4761
  gp2       15    100.00%         14     95.83%    97.56%   96.69%            0.19            0.0702
  gp3       10     50.00%         10     93.58%    71.38%   80.99%            3.21           29.8663
  gp4       10     90.00%         10     85.20%    74.55%   79.52%            1.20            1.9909
  gp5       24     37.50%         23    100.00%    41.27%   58.43%            4.28           51.3094
  gp6       10    100.00%         10     92.32%    80.97%   86.27%            1.30            3.3800
  gp7       12    100.00%         12     99.34%    95.22%   97.24%            0.21            0.0823
  gp8       10     90.00%         10     62.42%    77.37%   69.09%            1.21            2.5462
```

---

## Viewer Controls

Interactive viewer controls:

- `Right`, `d`, `n`, `Space`: next image
- `Left`, `a`, `p`: previous image
- `Up`, `w`: next pipeline step
- `Down`, `s`: previous pipeline step
- `f`: toggle final-only/full pipeline
- `c` or `x`: export current debug payload (`.json` + `.txt`)
- `q` or `Esc`: quit

---

## Current Runtime Defaults

Defaults are centralized in `src/pipeline/config.py`:

- Target canvas: `640x480` (letterbox)
- Hough preset: `test1`
- Blur mode: `gauss`
- Material mode: `lab_proto`  
  Available modes: `hsv`, `hsv_kmeans`, `lab_proto`

---

## Pipeline Summary

High-level flow:

1. Letterbox resize to stable canvas
2. Preprocessing (grayscale + blur)
3. Circle detection (Hough + automatic `minRadius` sweep)
4. Coin color/material analysis (`one-color`, `bi-metal`, `uncertain`)
5. Denomination estimation and total value computation
6. Evaluation + optional interactive debug viewer

Full technical description: `PIPELINE.md`

---

## Authors

- Maksym DOLHOV
- Mehdi AGHAEI
- Nima DAVARI

---

## License

Project intended for academic use.
