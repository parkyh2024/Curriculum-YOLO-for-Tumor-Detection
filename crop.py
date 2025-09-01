# =================================================================================================
# Description
# =================================================================================================
# This script creates a new dataset by cropping the edges of an original image dataset by a 
# specified number of pixels.
#
# This is a data preprocessing/augmentation technique primarily used to remove unnecessary borders
# from images like ultrasounds (e.g., black areas displaying equipment information), allowing the 
# model to focus more on the core region of the image during training.
#
# In addition to the images, the script also recalculates and saves the bounding box coordinates
# in the corresponding YOLO format label (.txt) files to match the new, cropped image dimensions.

# =================================================================================================
# Process Flow
# =================================================================================================
# 1. User Input:
#    - When the script is executed, it prompts the user to enter an integer value for the number 
#      of pixels to crop from each edge.
#
# 2. Path Setup and Directory Creation:
#    - The 'Original' folder containing the source data is designated as the source.
#    - A destination directory named '{input_pixels}px_crop' is created to store the results.
#      (e.g., a '10px_crop' folder is created for a 10px crop).
#
# 3. Image and Label Processing (File by File):
#    a. Image Crop:
#       - Loads an original image and crops it from all sides (top, bottom, left, right) by the 
#         number of pixels specified by the user.
#       - Saves the cropped image to the 'images' folder in the destination directory.
#
#    b. Label Recalculation:
#       - Reads the original label file (.txt).
#       - Converts the coordinates of each bounding box to absolute pixel values based on the 
#         original image dimensions.
#       - Since the origin (0,0) has shifted due to cropping, it subtracts the crop pixel value
#         from the bounding box's center coordinates.
#       - These new absolute coordinates are then re-normalized (converted to values between 0 and 1)
#         based on the new dimensions of the cropped image.
#       - The width and height of the bounding box are also re-normalized according to the new image size.
#
#    c. Save New Label:
#       - Saves the new label information with the recalculated coordinates to the 'labels' folder 
#         in the destination directory.
#
# 4. Completion:
#    - After processing all files, it prints a completion message and terminates the script.
# =================================================================================================

import os
from PIL import Image

def process_dataset(base_path, crop_pixels=10):
    source_dir = os.path.join(base_path, 'Original')
    dest_dir = os.path.join(base_path, f'{crop_pixels}px_crop')

    source_image_dir = os.path.join(source_dir, 'images')
    source_label_dir = os.path.join(source_dir, 'labels')

    dest_image_dir = os.path.join(dest_dir, 'images')
    dest_label_dir = os.path.join(dest_dir, 'labels')

    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    print(f"Successfully created directory: '{dest_dir}'")

    image_files = [f for f in os.listdir(source_image_dir) if f.endswith('.png')]
    total_files = len(image_files)
    print(f"Processing a total of {total_files} image files.")

    for i, filename in enumerate(image_files):
        source_image_path = os.path.join(source_image_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        source_label_path = os.path.join(source_label_dir, label_filename)

        dest_image_path = os.path.join(dest_image_dir, filename)
        dest_label_path = os.path.join(dest_label_dir, label_filename)

        with Image.open(source_image_path) as img:
            original_w, original_h = img.size
            
            crop_box = (crop_pixels, crop_pixels, original_w - crop_pixels, original_h - crop_pixels)
            cropped_img = img.crop(crop_box)
            cropped_img.save(dest_image_path)
            
        if not os.path.exists(source_label_path):
            continue

        new_label_lines = []
        new_w = original_w - 2 * crop_pixels
        new_h = original_h - 2 * crop_pixels

        with open(source_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = parts[0]
                x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:])

                abs_x_center = x_center_norm * original_w
                abs_y_center = y_center_norm * original_h
                abs_w = w_norm * original_w
                abs_h = h_norm * original_h

                new_abs_x_center = abs_x_center - crop_pixels
                new_abs_y_center = abs_y_center - crop_pixels
                
                new_x_center_norm = new_abs_x_center / new_w
                new_y_center_norm = new_abs_y_center / new_h
                new_w_norm = abs_w / new_w
                new_h_norm = abs_h / new_h

                new_line = f"{class_id} {new_x_center_norm:.6f} {new_y_center_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"
                new_label_lines.append(new_line)
        
        with open(dest_label_path, 'w') as f:
            f.write("\n".join(new_label_lines))
        
        print(f"[{i+1}/{total_files}] Processed: {filename}")

    print("\nAll tasks completed successfully! ✅")

if __name__ == '__main__':
    BASE_PATH = "/990pro/parkyh/Ultsound/2025/OASBUD_Dataset_with_Mask_and_Labels"
    
    while True:
        try:
            crop_input = input("Enter the number of pixels to crop (e.g., 10): ")
            pixels_to_crop = int(crop_input)
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    process_dataset(BASE_PATH, crop_pixels=pixels_to_crop)