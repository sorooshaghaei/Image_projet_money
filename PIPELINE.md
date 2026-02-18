# PIPELINE.md  
**Image Processing and Analysis Pipeline – TER M1 (VMI)**

This document describes the planned processing pipeline for the project.  
The objective is to progressively extract a **Region of Interest (ROI)** from input images using classical computer vision techniques, with a focus on **coin detection**.

---

## 1. Global Pipeline Overview

The pipeline follows a classical computer vision workflow:

1. Input image acquisition  
2. Preprocessing  
3. Grayscale conversion  
4. Conditional inversion  
5. Noise reduction  
6. Foreground–background separation  
7. ROI extraction  
8. (Next steps: feature extraction / analysis)

Each step is designed to be modular, reproducible, and adjustable.

---

## 2. Step 1 – Image Acquisition

**Input:**
- Raw RGB images stored in `data/images/`
- Format: PNG / JPG  
- Resolution may vary

**Goal:**
Ensure all images are readable, correctly loaded, and consistent before processing.

---

## 3. Step 2 – Grayscale Conversion

### Description
Conversion of RGB images to grayscale.

### Justification
- Reduces computational complexity  
- Removes color dependency  
- Coin detection relies primarily on shape and intensity  
- Simplifies further processing  

Grayscale images preserve luminance information required for segmentation and object extraction.

---

## 4. Step 3 – Image Inversion (Conditional Step)

### Purpose
Invert pixel intensities depending on the contrast configuration between:
- Coin
- Background

### When inversion SHOULD be applied
- Coin is darker than background  
- Coin boundaries are poorly distinguishable  
- Bright foreground improves separation  

### When inversion SHOULD NOT be applied
- Coin is already brighter than background  
- Contrast is sufficient  
- Inversion increases background noise  

### Decision Rule
Inversion is a **conditional preprocessing step** and should be:
- Evaluated visually  
- Applied only if it improves contrast  
- Used to simplify further segmentation  

### Important Note
After this step, **the primary objective becomes separating coins from the background**.  
All subsequent processing is focused on foreground–background separation, not edge detection.

---

## 5. Step 4 – Noise Reduction (Median Filtering)

### Selected Method: **Median Filter**

### Justification
Median filtering is preferred over Gaussian and bilateral filtering.

### Why NOT Gaussian or Bilateral Filtering
- Gaussian blur smooths edges and weakens object boundaries  
- Bilateral filtering:
  - Is computationally heavier  
  - Preserves unwanted texture  
  - Is sensitive to background patterns  

Textured backgrounds (e.g. wood, surfaces) can strongly affect detection quality.

### Advantages of Median Filtering
- Removes impulse and texture noise  
- Preserves object boundaries  
- Reduces background influence  
- More stable for segmentation  
- Well-suited for coin-like homogeneous regions  

---

## 6. Step 5 – Foreground / Background Separation

### Objective
Isolate coin regions from the background.

### Goals
- Suppress background textures  
- Highlight coin regions  
- Produce a clean binary or semi-binary representation  

### Possible Techniques
- HSV Color Filtering  
- HSV Color Clustering using K-Means
- Hough circle

No edge detection or morphological processing is applied at this stage.

---

## 7. Step 6 – ROI Extraction (Planned)

### Goals
- Detect candidate coin regions  
- Extract bounding boxes or masks  
- Filter regions using:
  - Area  
  - Shape consistency  
  - Optional circularity constraints  

### Possible Approaches
- TODO

---

## 8. Next Steps

- ROI validation  
- Feature extraction  
- Measurement or classification  
- Performance evaluation  
- Visualization and reporting  

---

## 9. Notes

- Each step is implemented as a separate module  
- Parameters are configurable  
- Intermediate results may be saved for debugging  
- The pipeline is designed to be reproducible and extensible  
