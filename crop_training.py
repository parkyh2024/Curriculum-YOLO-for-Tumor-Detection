# =================================================================================================
# Description
# =================================================================================================
# This script automates the training process for a YOLOv5 object detection model.
# Based on a user-specified preprocessed dataset (e.g., 10px_crop) and a seed value,
# it sequentially executes the entire process of data splitting, configuration file generation,
# and model training.
#
# This facilitates repetitive experiments, such as applying different seed values
# to the same dataset or training the model on various datasets under consistent conditions.
# The user can specify multiple datasets at once (e.g., 0, 10, 20-22).

# =================================================================================================
# Process Flow
# =================================================================================================
# 1. User Input:
#    - Prompts the user to enter the 'number of crop pixels' for training. Multiple values
#      can be specified with commas, and ranges with a hyphen.
#      (e.g., '0', '0,10,20', '20-25', '0,10,20-22')
#    - For reproducibility of data splitting and training, it prompts for a 4-digit 'seed'
#      value or generates a random seed if 'n' is entered.
#
# 2. Iterative Training Over Datasets:
#    - The following process is repeated for each of the entered pixel values.
#
# 3. Dataset Splitting and Config File (.txt) Generation:
#    - Retrieves a list of all image files from the specified dataset path.
#    - Uses `sklearn.model_selection.train_test_split` to divide the image list into
#      training (Train), validation (Validation), and testing (Test) sets
#      (65% / 15% / 20% ratio).
#    - Creates `_train.txt`, `_val.txt`, and `_test.txt` files containing the image
#      file paths for each respective set.
#
# 4. Dataset YAML Config File (.yaml) Generation:
#    - Generates the `_dataset.yaml` file required for YOLOv5 training.
#    - This file includes information such as the paths to the train/val/test sets,
#      the number of classes (nc), and class names (names).
#
# 5. YOLOv5 Training Execution:
#    - Executes the `yolov5/train.py` script using `subprocess.run`.
#    - Passes all necessary training parameters as command-line arguments, such as
#      image size, batch size, epochs, the dataset YAML file path, and pre-trained weights.
#
# 6. Completion:
#    - Prints a completion message if all training processes finish successfully,
#      or a list of failures if any errors occurred.
# =================================================================================================

import os
import glob
import yaml
import subprocess
import random
import sys
from sklearn.model_selection import train_test_split

BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "OASBUD_Dataset_with_Mask_and_Labels")
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")
WEIGHTS_PATH = os.path.join(BASE_PROJECT_PATH, "yolov5s.pt")

NUM_WORKERS = 8

def parse_crop_input(input_string):
    pixels_to_train = set()
    try:
        parts = input_string.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end:
                    print(f"Error: Invalid range '{part}'. Start value must be less than or equal to end value.")
                    return None
                pixels_to_train.update(range(start, end + 1))
            else:
                pixels_to_train.add(int(part))
    except ValueError:
        print(f"Error: Invalid format in '{part}'. Please use numbers, commas, and hyphens only.")
        return None
    
    return sorted(list(pixels_to_train))

def run_single_training(dataset_name, image_dir, seed):
    run_name = f"{dataset_name}_seed{seed}"
    
    print(f"\n{'='*25} Starting: '{run_name}' Dataset Training {'='*25}")

    print(f"\n[1/2] Splitting dataset and creating config for '{run_name}'...")
    all_image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    
    if not all_image_paths:
        print(f"!!!!!!!!!! No images found in '{image_dir}'. Skipping this training run. !!!!!!!!!!!")
        return True, f"No images found in {image_dir}"

    train_files, temp_files = train_test_split(all_image_paths, train_size=0.65, random_state=seed)
    val_files, test_files = train_test_split(temp_files, test_size=(0.20 / 0.35), random_state=seed)

    train_txt_path = os.path.join(CONFIG_DIR, f"{run_name}_train.txt")
    val_txt_path = os.path.join(CONFIG_DIR, f"{run_name}_val.txt")
    test_txt_path = os.path.join(CONFIG_DIR, f"{run_name}_test.txt")
    
    with open(train_txt_path, 'w') as f: f.write('\n'.join(train_files))
    with open(val_txt_path, 'w') as f: f.write('\n'.join(val_files))
    with open(test_txt_path, 'w') as f: f.write('\n'.join(test_files))

    yaml_path = os.path.join(CONFIG_DIR, f"{run_name}_dataset.yaml")
    
    yaml_data = {
        'path': BASE_PROJECT_PATH,
        'train': os.path.relpath(train_txt_path, BASE_PROJECT_PATH),
        'val': os.path.relpath(val_txt_path, BASE_PROJECT_PATH),
        'test': os.path.relpath(test_txt_path, BASE_PROJECT_PATH),
        'nc': 2, 'names': ['Benign', 'Malignant']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(" -> Done.")

    print(f"\n[2/2] Starting training for '{run_name}'...")
    train_command = [
        'python3', 'yolov5/train.py',
        '--img', '640', '--batch', '16', '--epochs', '200',
        '--data', yaml_path,
        '--weights', WEIGHTS_PATH,
        '--name', run_name,
        '--workers', str(NUM_WORKERS),
        '--seed', str(seed)
    ]
    train_result = subprocess.run(train_command)
    
    if train_result.returncode != 0:
        print(f"!!!!!!!!!! Error during training for '{run_name}' !!!!!!!!!!!")
        return False, f"Subprocess failed for {run_name}"
        
    print(f" -> Training for '{run_name}' complete.")
    return True, None

def main():
    crop_pixel_list = []
    while not crop_pixel_list:
        user_input = input("Enter crop pixels to train with (e.g., 0, 10, 20-22, 25): ")
        parsed_list = parse_crop_input(user_input)
        if parsed_list is not None and parsed_list:
            crop_pixel_list = parsed_list
        else:
            print("Invalid input. Please try again.")

    seed_value = -1
    while True:
        try:
            seed_input = input("Enter a 4-digit seed (0-9999), or 'n' for a random seed: ")
            
            if seed_input.lower() == 'n':
                seed_value = random.randint(0, 9999)
                print(f" -> A random seed has been generated: {seed_value:04d}")
                break
                
            seed_value = int(seed_input)
            
            if 0 <= seed_value <= 9999:
                break
            else:
                print("Error: Please enter a number between 0 and 9999.")
        except ValueError:
            print("Invalid input. Please enter a number or 'n'.")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    target_datasets = [f"{p}px_crop" for p in crop_pixel_list]
    print(f"\n[Initialize] Target Datasets for training: {target_datasets}")
    print(f"[Initialize] Using seed: {seed_value}")

    failed_trainings = []
    
    for crop_pixels in crop_pixel_list:
        dataset_name = f"{crop_pixels}px_crop"
        image_dir = os.path.join(BASE_DATA_PATH, dataset_name, "images")

        if not os.path.isdir(image_dir):
            print(f"\n!!!!!! ERROR: Dataset directory not found for '{dataset_name}': {image_dir} !!!!!!!")
            print(f"Please make sure you have created the '{dataset_name}' dataset first. Skipping.")
            failed_trainings.append(dataset_name)
            continue
        
        success, reason = run_single_training(dataset_name, image_dir, seed_value)
        if not success:
            failed_trainings.append(f"{dataset_name} ({reason})")

    print("\n\n" + "="*35)
    if not failed_trainings:
        print(f"✅ All training sessions with seed {seed_value} completed successfully!")
    else:
        print(f"❌ Training with seed {seed_value} finished with some failures.")
        print("Failed datasets:")
        for failure in failed_trainings:
            print(f"  - {failure}")
    print("="*35)

if __name__ == "__main__":
    main()