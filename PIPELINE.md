# PIPELINE.md  
**Image Processing and Analysis Pipeline – TER M1 (VMI)**

This document describes the planned processing pipeline for the project.  
The objective is to progressively extract a **Region of Interest (ROI)** from input images using classical computer vision techniques, with a focus on **coin detection**.

---

## 1. Global Pipeline Overview

The pipeline follows a classical computer vision workflow:

1. Input image acquisition  
2. Preprocessing  
3. Edge detection  
4. Morphological processing  
5. ROI extraction  
6. (Next steps: feature extraction / classification / analysis)

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
- Coin detection relies primarily on shape and contrast  
- Simplifies edge detection and segmentation  

Grayscale images preserve intensity information required for contour and edge-based methods.

---

## 5. Step 3 – Image Inversion (Conditional Step)

### Purpose
Invert pixel intensities depending on contrast configuration between:
- Coin
- Background

### When inversion SHOULD be applied
- Coin is darker than background
- Edge detection fails to highlight coin boundaries
- Thresholding or edge detection performs better on bright foregrounds

In these cases, inversion improves:
- Edge visibility
- Contrast separation
- Stability of contour extraction

### When inversion SHOULD NOT be applied
- Coin is already brighter than background
- Contrast is sufficient for edge detection
- Inversion would amplify background noise
- Object/background relationship is already optimal

### Decision Rule
Image inversion should be:
- Tested visually
- Applied only if it improves edge clarity
- Treated as a conditional preprocessing step

---

## 4. Step 4 – Noise Reduction (Bilateral Filtering)

### Selected Method: **Bilateral Filter**

### Why bilateral filtering is used instead of Gaussian blur

Unlike Gaussian blur, **bilateral filtering smooths the image while preserving edges**.

This is particularly important for **coin detection**, where:
- Coin boundaries must remain sharp
- Shape integrity is critical
- Edge continuity directly impacts contour detection

### Advantages for this project
- Reduces noise without blurring object borders  
- Preserves circular contours of coins  
- Prevents loss of edge information  
- Improves reliability of later edge detection  

### Why Gaussian blur is not ideal here
- Smooths both noise and edges
- Can weaken coin boundaries
- May cause merging between coin and background
- Reduces contour accuracy

### Conclusion
Bilateral filtering is better suited for:
- Objects with strong, meaningful edges
- Shape-based detection tasks
- Scenarios where boundary preservation is critical

---

## 6. Step 5 – Edge Detection

### Selected Method: Canny Edge Detection

### Justification
- Multi-stage detection (gradient → suppression → hysteresis)
- Produces thin and continuous edges
- Robust to noise when preceded by bilateral filtering
- Well-suited for circular object detection

### Why not Sobel?
- Produces thick edges
- Sensitive to noise
- No hysteresis or edge linking

### Why not simple thresholding?
- Sensitive to lighting
- Fails when foreground and background intensities overlap
- Not reliable for complex textures

---

## 7. Step 6 – Morphological Processing

### Objective
Refine edge maps and prepare for ROI extraction.

### Operations
- **Dilation**: closes gaps in contours  
- **Erosion**: removes isolated noise  
- **Opening**: removes small artifacts  
- **Closing**: fills internal gaps in coin contours  

### Purpose
- Ensure closed coin boundaries  
- Improve contour stability  
- Prepare for accurate ROI extraction  

---

## 8. Step 7 – ROI Extraction (Planned)

### Goals
- Detect coin contours
- Filter candidates using:
  - Area
  - Circularity
  - Shape consistency
- Extract bounding boxes or masks

### Possible Approaches
- Contour detection
- Connected component analysis
- Shape filtering (circularity metrics)

---

## 9. Next Steps

- ROI validation
- Feature extraction
- Classification or measurement
- Performance evaluation
- Final visualization and reporting

---

## 10. Notes

- Each step will be implemented as a separate module  
- Parameters will be configurable  
- Intermediate outputs may be saved for debugging  
- Pipeline is designed to be reproducible and extensible  

---
