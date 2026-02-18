# Image_projet_money
Euro coin detection and value estimation project (TER M1, Universite Paris Cite).

Authors: Maksym DOLHOV, Mehdi AGHAEI, Nima DAVARI.

## What This Project Does
Given one image with euro coins:
- detect coin circles,
- classify likely material groups (bronze / gold / bimetal),
- assign denominations with a global geometric consistency fit,
- estimate total value in EUR.

Core design:
- geometry-first detection (circle ensemble),
- color as a soft prior (center-ring LAB/HSV features),
- one shared px/mm scale for denomination assignment.

## Current Project Structure
```text
Image_projet_money/
├── main.py
├── README.md
├── PIPELINE.md
├── requirements.txt
├── data/
│   ├── images/                 # local image dataset (not tracked) -> gitignored
│   └── annotations/            # local/optional annotation files (not tracked)
├── report/
│   ├── project_methods_logic.ipynb
│   └── progress_hough_hsv_report.ipynb
└── src/
    ├── config.py               # tunable runtime + detection config
    ├── dataset.py              # current in-code annotation table
    ├── io_utils.py             # image path resolution across groups
    ├── models.py               # PipelineResult / PipelineStep
    ├── coin_metadata.py        # denomination + color mapping constants
    ├── processor.py            # end-to-end orchestration
    ├── processor_circles.py    # circle detection and filtering
    ├── processor_color.py      # center-ring LAB/HSV color logic
    ├── processor_scale.py      # global scale and denomination fitting
    ├── runner.py               # batch evaluation app
    └── visualization.py        # interactive tuning / debug visualization
```

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

`main.py` runs `src.runner.PipelineApp`, which:
- loads annotations from `src/dataset.py`,
- processes dataset images,
- prints count/value metrics,
- can open the interactive tuning browser (depending on `RuntimeConfig`).

## Data and Annotations
Images are expected in:
- `data/images/<group>/<image_name>`

Current ground truth is in:
- `src/dataset.py` (`DatasetRepository.DATA_ROWS`)

For this project, annotations are managed directly in:
- `src/dataset.py` (`DatasetRepository.DATA_ROWS`)

## Reports
Notebooks in `report/` are presentation-ready:
- `project_methods_logic.ipynb`: final architecture and method logic
- `progress_hough_hsv_report.ipynb`: progression from early attempts to final pipeline

## Notes
- Heavy image datasets are intentionally not committed.
- Keep code modular and reproducible.
- Keep `src/dataset.py` synchronized with your local image folders.
