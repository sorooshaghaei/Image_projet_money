# Image_projet_money
Maksym DOLHOV, Mehdi AGHAEI and Nima DAVARI
## TER M1

This repository contains the **TER (Travail d’Étude et de Recherche)** project for **Master 1 – VMI - Analyse d'Image Cours**.  

---

## Repository Structure

```bash
.
├── data/
│   ├── images/        # Image dataset (ignored in git, except .gitkeep)
│   └── annotations/   # Annotations / labels (ignored in git, except .gitkeep)
│
├── src/               # Source code (Python)
│
├── report/            # Slides and notes for the final presentation
│
├── .gitignore
└── README.md
```

---

## Project Goals

* Implement computer vision and machine learning methods related to the chosen TER topic
* Perform data preprocessing, analysis, and experimentation
* Produce clear and reproducible code
* Prepare a final report and presentation for the oral defense

---

## Technologies

* **Language**: Python 3
* **Environment**: macOS / windows
* **Main libraries** (depending on the topic):

  * NumPy
  * OpenCV
  * Matplotlib
  * Jupyter Notebook (for experiments)
  * (this list will get updated)

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet available, install libraries manually as needed.)*

---

## Data Management

* The `data/images/` and `data/annotations/` folders are **ignored by git** to avoid pushing large datasets.
* Only `.gitkeep` files are versioned to preserve the directory structure.
* Datasets must be added locally and never pushed to the remote repository.

---

## Usage

* All source code must be placed in the `src/` directory.
* Scripts should be modular, documented, and reproducible.
* Experiments and temporary files should not be committed unless explicitly required.

---

## Report & Defense

* The `report/` directory contains:

  * Presentation slides
  * Notes for the oral defense
* Temporary files (LaTeX, Office cache files) are ignored by git.

---

## Collaboration Rules

* Commit frequently with clear messages
* Do not commit datasets or generated results unless agreed
* Keep code clean, commented, and structured

---

## Authors

* Maksym DOLHOV, Mehdi AGHAEI and Nima DAVARI
* Université Paris Cité

---

## License

This project is for **academic use only**.

