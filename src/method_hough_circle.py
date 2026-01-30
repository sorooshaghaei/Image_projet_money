import cv2
import numpy as np
import os
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class DetectionConfig:
    """
    Immutable configuration object for coin detection parameters.
    Keeps 'magic numbers' out of the logic code.
    """
    TARGET_WIDTH: int = 800
    BLUR_KERNEL_SIZE: int = 15
    
    # Hough Circle Parameters
    HOUGH_DP: float = 1.2
    HOUGH_MIN_DIST: int = 70
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 45
    HOUGH_MIN_RADIUS: int = 10
    HOUGH_MAX_RADIUS: int = 150

    # Visualization
    FIG_SIZE: Tuple[int, int] = (16, 6)
    mask_color: Tuple[int, int, int] = (0, 255, 0)

    # Valid file extensions
    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

@dataclass
class PipelineStep:
    name: str
    image: np.ndarray
    cmap: str 

@dataclass
class PipelineResult:
    steps: List[PipelineStep]
    coin_count: int
    is_inverted: bool
    source_filename: str

# ==========================================
# 2. COIN PROCESSOR
# ==========================================
class CoinProcessor:
    """
    Simplified processor: Grayscale, Norm/Invert, Blur, Hough.
    """

    def __init__(self, config: DetectionConfig):
        self._cfg = config

    def execute(self, img: np.ndarray, filename: str = "Unknown") -> Optional[PipelineResult]:
        if img is None or img.size == 0:
            return None

        steps: List[PipelineStep] = []

        # 1. Resize & Prep
        img_resized = self._resize(img)
        display_img = img_resized.copy() 
        # Pass "rgb" as the cmap argument
        steps.append(PipelineStep("1. Original", img_resized, "rgb"))
        
        # 2. Grayscale Conversion
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 3. Normalization (Robust contrast fix)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # 4. Check Brightness & Invert if Dark
        mean_brightness = np.mean(gray)
        if mean_brightness < 110: 
            gray = cv2.bitwise_not(gray)
            steps.append(PipelineStep("2a. Inverted (Low Brightness)", gray, "gray"))
        else:
            steps.append(PipelineStep("2. Grayscale", gray, "gray"))

        # 5. Blur
        blurred = cv2.medianBlur(gray, self._cfg.BLUR_KERNEL_SIZE)
        steps.append(PipelineStep("3. Median Blur", blurred, "gray"))

        # 6. Hough Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=70,       
            param1=50, 
            param2=45,       
            minRadius=10,
            maxRadius=150 
        )

        '''
            dp=1.2, 
            minDist=70,       
            param1=40, 
            param2=50,       
            minRadius=20,
            maxRadius=150 
            
            68,93%
            '''

        # 7. Drawing & Masking
        mask = np.zeros_like(gray)
        coin_count = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            coin_count = circles.shape[1]
            
            for i in circles[0, :]:
                cv2.circle(display_img, (i[0], i[1]), i[2], (0, 255, 0), 3)
                cv2.circle(display_img, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)

        steps.append(PipelineStep("4. Detected Circles", display_img, "rgb"))

        return PipelineResult(
            steps=steps,
            coin_count=coin_count,
            is_inverted=False, 
            source_filename=filename
        )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        if width == 0: return img
        scale = self._cfg.TARGET_WIDTH / width
        return cv2.resize(img, (self._cfg.TARGET_WIDTH, int(height * scale)))

# ==========================================
# 3. UTILITIES & DATA
# ==========================================
def get_image_path(base_dir: str, filename: str, group: str) -> Optional[str]:
    """
    Attempts to find the image file.
    Since duplicates exist (e.g. 1.jpeg in gp5 AND gp2), we must check subfolders.
    """
    # 1. Check strict path: base/group/filename
    path_grouped = os.path.join(base_dir, group, filename)
    if os.path.exists(path_grouped):
        return path_grouped
    
    # 2. Check strict path lowercased/renamed: grp->gp
    if group.startswith("grp"): 
        alt_group = group.replace("grp", "gp")
        path_alt = os.path.join(base_dir, alt_group, filename)
        if os.path.exists(path_alt):
            return path_alt

    # 3. Check flat path (Last resort)
    path_flat = os.path.join(base_dir, filename)
    if os.path.exists(path_flat):
        return path_flat
        
    return None

DATA_ROWS = [
    ["exemple1.png", 4, 7.25, "gp1"],
    ["10.jpg", 9, 3.13, "gp5"],
    ["11.jpg", 12, 6.18, "gp5"],
    ["12.jpg", 16, 8.83, "gp5"],
    ["13.jpg", 19, 12.33, "gp5"],
    ["14.jpg", 28, 15.69, "gp5"],
    ["15.jpg", 35, 17.32, "gp5"],
    ["16.jpg", 48, 18.69, "gp5"],
    ["17.jpg", 48, 18.20, "gp5"],
    ["0.jpeg", 2, 2.20, "gp5"],
    ["1.jpeg", 4, 4.22, "gp5"],
    ["2.jpeg", 3, 3.20, "gp5"],
    ["3.jpeg", 4, 0.80, "gp5"],
    ["4.jpeg", 3, 3.00, "gp5"],
    ["5.jpeg", 2, 1.20, "gp5"],
    ["6.jpeg", 11, 10.26, "gp5"],
    ["7.jpeg", 3, 1.70, "gp5"],
    ["8.jpg", 6, None, "gp5"],
    ["9.jpg", 8, 3.88, "gp5"],

    ["18.png", 7, 4.31, "gp1"],
    ["19.png", 4, 1.60, "gp1"],
    ["20.png", 8, 4.81, "gp1"],
    ["21.png", 6, 3.76, "gp1"],
    ["22.png", 5, 2.25, "gp1"],
    ["23.png", 8, 4.34, "gp1"],
    ["24.png", 3, 2.55, "gp1"],
    ["25.png", 10, 4.40, "gp1"],
    ["26.jpg", 8, 3.51, "gp1"],
    ["27.jpg", 9, 0.88, "gp1"],
    ["28.jpg", 3, 0.21, "gp1"],
    ["29.jpg", 5, 0.36, "gp1"],
    ["30.jpg", 7, 3.72, "gp1"],
    ["31.jpg", 4, 1.70, "gp1"],

    ["3_1.jpg", 8, 5.00, "grp3"],
    ["3_2.jpg", 16, 4.80, "grp3"],
    ["3_3.jpg", 8, 5.00, "grp3"],
    ["3_4.jpg", 10, 4.03, "grp3"],
    ["3_5.jpg", 25, 12.50, "grp3"],
    ["3_6.jpg", 8, 16.00, "grp3"],
    ["3_7.jpg", 8, 16.00, "grp3"],
    ["3_8.jpg", 50, 5.00, "grp3"],
    ["3_9.jpg", 24, 24.00, "grp3"],
    ["3_10.jpg", 35, 3.50, "grp3"],

    ["2e01.jpg", 8, 2.01, "grp5"],
    ["3e19.jpg", 10, 3.19, "grp5"],
    ["4.17.jpg", 12, 4.17, "grp5"],
    ["4e22.jpg", 8, 4.22, "grp5"],
    ["6e19.jpg", 12, 6.19, "grp5"],
    ["8e88.jpg", 20, 8.88, "grp5"],
    ["10e05.jpg", 26, 10.05, "grp5"],

    ["1.jpg", 2, 1.50, "grp4"],
    ["2.jpg", 4, 2.27, "grp4"],
    ["3.jpg", 5, 3.27, "grp4"],
    ["4.jpg", 7, 1.88, "grp4"],
    ["5.jpg", 8, 4.38, "grp4"],
    ["6.jpg", 7, 2.37, "grp4"],
    ["7.jpg", 8, 3.88, "grp4"],
    ["8.jpg", 8, 3.88, "grp4"],
    ["9.jpg", 4, 2.65, "grp4"],
    ["10.jpg", 7, 5.12, "grp4"],

    ["60.jpg", 13, 6.33, "gp6"],
    ["61.jpg", 11, 5.53, "gp6"],
    ["62.jpg", 9, 6.86, "gp6"],
    ["63.jpg", 9, 5.34, "gp6"],
    ["64.jpg", 12, 7.07, "gp6"],
    ["65.jpg", 13, 2.63, "gp6"],
    ["66.jpg", 7, 0.77, "gp6"],
    ["67.jpg", 10, 3.31, "gp6"],
    ["68.jpg", 11, 5.41, "gp6"],
    ["69.jpg", 9, 7.40, "gp6"],

    ["gp7_01.webp", 7, 3.79, "gp7"],
    ["gp7_02.webp", 12, 1.85, "gp7"],
    ["gp7_03.webp", 12, 4.60, "gp7"],
    ["gp7_04.webp", 13, 4.65, "gp7"],
    ["gp7_05.webp", 12, 4.15, "gp7"],
    ["gp7_06.webp", 12, 4.74, "gp7"],
    ["gp7_07.webp", 11, 3.74, "gp7"],
    ["gp7_08.webp", 10, 4.19, "gp7"],
    ["gp7_09.webp", 11, 2.55, "gp7"],
    ["gp7_10.webp", 9, 4.46, "gp7"],
    ["gp7_11.webp", 10, 4.03, "gp7"],
    ["gp7_12.webp", 14, 4.95, "gp7"],

    ["IMG_1136.png", 5, 0.83, "gp8"],
    ["IMG_1137.png", 10, 2.16, "gp8"],
    ["IMG_1138.png", 9, 2.17, "gp8"],
    ["IMG_1139.png", 4, 1.21, "gp8"],
    ["IMG_1140.png", 11, 2.47, "gp8"],
    ["IMG_1141.png", 7, 1.36, "gp8"],
    ["IMG_1142.png", 4, 1.52, "gp8"],
    ["IMG_1143.png", 17, 1.40, "gp8"],
    ["IMG_1144.png", 16, 0.43, "gp8"],
    ["IMG_1145.png", 5, 2.12, "gp8"],

    ["1.jpeg", 8, 3.86, "gp2"],
    ["2.jpeg", 2, 3.00, "gp2"],
    ["3.jpeg", 3, 2.70, "gp2"],
    ["4.jpeg", 8, 3.86, "gp2"],
    ["5.jpeg", 3, 0.24, "gp2"],
    ["6.jpeg", 9, 3.98, "gp2"],
    ["7.jpeg", 9, 3.98, "gp2"],
    ["8.jpeg", 3, 3.50, "gp2"],
    ["9.jpeg", 6, 0.96, "gp2"],
    ["10.jpeg", 6, 0.96, "gp2"],
    ["11.jpeg", 9, 3.37, "gp2"],
    ["12.jpeg", 2, 3.00, "gp2"],
    ["13.jpeg", 9, 3.87, "gp2"],
    ["14.jpeg", 4, 2.45, "gp2"],
    ["15.jpeg", 5, 3.90, "gp2"]
]

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    try:
        # --- CONFIGURATION ---
        IMAGE_DIRECTORY = "/home/max/Desktop/Coding/Github/Image_projet_money/data/images"
        
        config = DetectionConfig()
        
        # Create DataFrame
        df = pd.DataFrame(DATA_ROWS, columns=["image", "pieces", "value_eur", "group"])
        print(f"[INFO] Loaded {len(df)} annotations from Data Table.")

        # Initialize Processor
        processor = CoinProcessor(config)

        # Stats
        correct = 0
        total_processed = 0
        total_abs_error = 0
        
        print("\n" + "="*85)
        print(f"{'FILENAME':<25} | {'GRP':<5} | {'PRED':<6} | {'TRUE':<6} | {'DIFF':<6} | {'STATUS':<10}")
        print("="*85)

        for index, row in df.iterrows():
            filename = row['image']
            true_count = row['pieces']
            group = row['group']
            
            # Resolve Path
            image_path = get_image_path(IMAGE_DIRECTORY, filename, group)
            
            if not image_path:
                # print(f"[SKIP] {filename} not found in {group} or root.")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERR ] Unreadable: {filename}")
                continue

            # Run Pipeline
            result = processor.execute(img, filename)
            
            if result:
                pred = result.coin_count
                diff = pred - true_count
                total_abs_error += abs(diff)
                total_processed += 1
                
                status = "PERFECT" if diff == 0 else "ERROR"
                if diff == 0: correct += 1
                
                print(f"{filename:<25} | {group:<5} | {pred:<6} | {true_count:<6} | {diff:<6} | {status:<10}")

        # Final Summary
        if total_processed > 0:
            acc = (correct / total_processed) * 100
            mae = total_abs_error / total_processed
            print("="*85)
            print(f"Total Images:     {total_processed}")
            print(f"Perfect Matches:  {correct}")
            print(f"Accuracy:         {acc:.2f}%")
            print(f"Mean Abs Error:   {mae:.2f} coins/image")
            print("="*85)
        else:
            print("[WARN] No images processed. Check your IMAGE_DIRECTORY path.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()