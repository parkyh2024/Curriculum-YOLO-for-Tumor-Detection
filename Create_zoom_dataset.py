# =================================================================================================
# Description
# =================================================================================================
# This script generates a new 'zoomed-in' dataset based on an original image dataset.
# Its main purpose is data augmentation to enhance the training performance of object detection
# models (e.g., YOLO). It creates images zoomed in at various magnifications, centered on 
# specific objects (tumors in this case), to train the model to better recognize objects of 
# various sizes and resolutions.

# =================================================================================================
# Process Flow
# =================================================================================================
# 1. User Input:
#    - When the script is run, it prompts the user to enter the 'zoom factor' for the dataset to be created.
#    - The input can be in various formats: a single number (e.g., 5), a range (e.g., 2-5), or a 
#      specific list (e.g., 2,4,6).
#
# 2. Dataset Path Configuration:
#    - It references the paths of the original images and label files (`Original/images`, `Original/labels`).
#    - For each zoom factor, it creates a new folder named `Zoom{factor}_640` to store the results.
#
# 3. Image and Label Processing (Iterated for each zoom factor):
#    a. Load Image and Label:
#       - Iterates through all image files (`*.png`) in the original image folder.
#       - Finds the corresponding label file (`*.txt`) for each image and reads the object information.
#
#    b. Calculate Region of Interest (ROI):
#       - Based on the bounding box information from the label file, it calculates a wider 
#         'Region of Interest (ROI)' expanded by the user-provided `zoom_factor`.
#       - E.g., if `zoom_factor` is 2, the ROI will be twice the width and height of the original bounding box.
#
#    c. Crop Image:
#       - Crops the image from the original based on the calculated ROI.
#
#    d. Resize and Pad Image:
#       - The cropped image is resized to fit the target size (640x640) while maintaining the aspect ratio.
#       - Then, the remaining space is filled with gray (value 114) to make it a 640x640 square (padding).
#
#    e. Save New Image:
#       - The final processed 640x640 image is saved to the new path (`Zoom{factor}_640/images/`).
#
#    f. Recalculate and Save Label:
#       - The bounding box coordinates of the object are recalculated to match the transformed image 
#         (after cropping, resizing, and padding).
#       - These new coordinates are converted to YOLO format and saved as a new label file in 
#         (`Zoom{factor}_640/labels/`).
#
# 4. Completion:
#    - The entire process terminates once the datasets for all specified zoom factors have been created.
# =================================================================================================

import os
import cv2
import re
import numpy as np
import shutil
import argparse
from glob import glob
from tqdm import tqdm

BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "OASBUD_Dataset_with_Mask_and_Labels")
ORIGINAL_IMAGES_DIR = os.path.join(BASE_DATA_PATH, "Original/images")
ORIGINAL_LABELS_DIR = os.path.join(BASE_DATA_PATH, "Original/labels")
TARGET_IMG_SIZE = 640

def create_zoomed_dataset(zoom_factor):
    
    output_dir = os.path.join(BASE_DATA_PATH, f"Zoom{zoom_factor}_640")
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    if os.path.exists(output_dir):
        print(f"\nNote: Directory '{output_dir}' already exists. It will be replaced.")
        shutil.rmtree(output_dir)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"\n===== Starting to create 'Zoom{zoom_factor}_640' dataset =====")
    print(f" -> Output directory: {output_dir}")

    image_paths = sorted(glob(os.path.join(ORIGINAL_IMAGES_DIR, '*.png')))

    for img_path in tqdm(image_paths, desc=f"Processing Zoom{zoom_factor}"):
        basename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(ORIGINAL_LABELS_DIR, filename_no_ext + '.txt')

        if not os.path.exists(label_path):
            continue

        original_image = cv2.imread(img_path)
        H, W, _ = original_image.shape

        with open(label_path, 'r') as f:
            line = f.readline()
            if not line:
                continue
            
            class_id, x_c, y_c, w, h = map(float, line.strip().split())

            tumor_center_x_px = x_c * W
            tumor_center_y_px = y_c * H
            tumor_width_px = w * W
            tumor_height_px = h * H

            roi_width = tumor_width_px * zoom_factor
            roi_height = tumor_height_px * zoom_factor
            
            x1 = int(tumor_center_x_px - (roi_width / 2))
            y1 = int(tumor_center_y_px - (roi_height / 2))
            x2 = int(tumor_center_x_px + (roi_width / 2))
            y2 = int(tumor_center_y_px + (roi_height / 2))
            
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(W, x2)
            y2_clamped = min(H, y2)

            cropped_image = original_image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

            h_crop, w_crop, _ = cropped_image.shape
            
            scale = TARGET_IMG_SIZE / max(h_crop, w_crop)
            new_w, new_h = int(w_crop * scale), int(h_crop * scale)
            resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            padded_image = np.full((TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3), 114, dtype=np.uint8)
            
            pad_x = (TARGET_IMG_SIZE - new_w) // 2
            pad_y = (TARGET_IMG_SIZE - new_h) // 2
            padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
            
            final_image_filename = f"{filename_no_ext}_crop1.png"
            cv2.imwrite(os.path.join(output_images_dir, final_image_filename), padded_image)

            new_center_x_px = tumor_center_x_px - x1_clamped
            new_center_y_px = tumor_center_y_px - y1_clamped

            final_center_x = (new_center_x_px * scale) + pad_x
            final_center_y = (new_center_y_px * scale) + pad_y
            final_w = tumor_width_px * scale
            final_h = tumor_height_px * scale

            final_x_c_norm = final_center_x / TARGET_IMG_SIZE
            final_y_c_norm = final_center_y / TARGET_IMG_SIZE
            final_w_norm = final_w / TARGET_IMG_SIZE
            final_h_norm = final_h / TARGET_IMG_SIZE

            final_x_c_norm = max(0.0, min(1.0, final_x_c_norm))
            final_y_c_norm = max(0.0, min(1.0, final_y_c_norm))
            final_w_norm = max(0.0, min(1.0, final_w_norm))
            final_h_norm = max(0.0, min(1.0, final_h_norm))
            
            new_label_filename = f"{filename_no_ext}_crop1.txt"
            with open(os.path.join(output_labels_dir, new_label_filename), 'w') as out_f:
                out_f.write(f"{int(class_id)} {final_x_c_norm:.6f} {final_y_c_norm:.6f} {final_w_norm:.6f} {final_h_norm:.6f}\n")

    print(f"===== 'Zoom{zoom_factor}_640' dataset creation complete! =====")

def parse_zoom_input(input_str):
    input_str = input_str.strip()
    if '-' in input_str:
        parts = input_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = int(parts[0]), int(parts[1])
            if start > 0 and end >= start:
                return list(range(start, end + 1))
    elif ',' in input_str:
        parts = input_str.split(',')
        if all(p.strip().isdigit() for p in parts):
            return [int(p.strip()) for p in parts if int(p.strip()) > 0]
    elif input_str.isdigit():
        factor = int(input_str)
        if factor > 0:
            return [factor]
    return None

if __name__ == '__main__':
    while True:
        zoom_input = input("Enter the zoom factor(s) to generate (e.g., '5', '2-5', or '2,4,6'): ")
        zoom_factors_to_create = parse_zoom_input(zoom_input)
        if zoom_factors_to_create:
            print(f" -> Creating the following datasets: {zoom_factors_to_create}")
            break
        else:
            print("Error: Invalid format. Please try again.")

    for factor in zoom_factors_to_create:
        create_zoomed_dataset(factor)
        
    print("\n\n" + "="*25 + " All Dataset Creation Complete " + "="*25)