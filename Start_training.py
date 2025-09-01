# =================================================================================================
# Description
# =================================================================================================
# This script is designed to automatically and sequentially run multiple YOLOv5 model
# training experiments.
#
# Its core functionality is to automatically discover all existing datasets based on a
# predefined rule (e.g., 'Zoom2_640', 'Zoom3_640', etc.) and then proceed with training
# for each discovered dataset. The user provides a single seed value at the beginning,
# which is applied consistently across all experiments. This allows for a fair comparison
# of model performance based on different data augmentation methods under uniform conditions.
#
# If any of the experiments fail, the entire process is terminated immediately.

# =================================================================================================
# Process Flow
# =================================================================================================
# 1. User Input:
#    - Prompts the user for a single 'seed' value that will be applied to all experiments.
#      If 'n' is entered, a single random seed is generated for use in all experiments.
#
# 2. Experiment Initialization and Automatic Dataset Discovery:
#    - Adds the 'Original' dataset as the baseline experiment.
#    - Sequentially searches for directories matching the 'Zoom{number}_640' pattern
#      and automatically adds all found Zoom datasets to the experiment list.
#    - Finally, adds the 'Original_Masked' dataset to complete the full list of experiments.
#
# 3. Sequential Training Execution (Iterating through the experiment list):
#    - Iterates through the completed experiment list and executes training for each
#      dataset individually.
#    - The detailed process for each training run (run_single_training) is as follows:
#      a. Data Splitting: Using the common seed value, it splits the current dataset into
#         training/validation/test sets and creates .txt files containing the file paths.
#      b. YAML Config File Generation: Dynamically creates the .yaml configuration file
#         required for YOLOv5 training.
#      c. YOLOv5 Training Execution: Starts the actual model training by running
#         `yolov5/train.py` via `subprocess`.
#
# 4. Termination on Failure and Completion:
#    - If any of the sequential experiments fail (encounter an error), the entire
#      script terminates immediately.
#    - If all experiments are completed successfully, a final completion message is printed.
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

def run_single_training(exp_config, seed):
    dataset_name = exp_config['name']
    run_name = f"{dataset_name}_seed{seed}"
    
    print(f"\n{'='*25} Starting: '{run_name}' Dataset Training {'='*25}")

    print(f"\n[1/2] Splitting dataset and creating config for '{run_name}'...")
    image_dir = DATASET_INFO[dataset_name]["images"]
    all_image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    
    if not all_image_paths:
        print(f"!!!!!!!!!! No images found in {image_dir}. Skipping training. !!!!!!!!!!!")
        return True

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
        return False
        
    print(f" -> Training for '{run_name}' complete.")
    return True

def main():
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

    print(f"\n[Initialize] Using seed: {seed_value} for all experiments.")
    
    print(f"[Initialize] Ensuring config directory exists: {CONFIG_DIR}")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    print("[Initialize] Discovering datasets...")
    global EXPERIMENTS, DATASET_INFO
    EXPERIMENTS = [{'name': 'Original'}]
    DATASET_INFO = {"Original": {"images": os.path.join(BASE_DATA_PATH, "Original/images")}}

    zoom_level = 2
    while True:
        run_name = f"Zoom{zoom_level}_640"
        image_dir = os.path.join(BASE_DATA_PATH, run_name, "images")
        
        if os.path.isdir(image_dir):
            print(f" -> Found dataset: {run_name}")
            EXPERIMENTS.append({'name': run_name})
            DATASET_INFO[run_name] = {"images": image_dir}
            zoom_level += 1
        else:
            print(f" -> Search complete. Did not find '{run_name}'.")
            break
            
    EXPERIMENTS.append({'name': 'Original_Masked'})
    DATASET_INFO["Original_Masked"] = {"images": os.path.join(BASE_DATA_PATH, "Original_Masked/images")}
    print(" -> Final experiment list:", [exp['name'] for exp in EXPERIMENTS])

    for exp_config in EXPERIMENTS:
        success = run_single_training(exp_config, seed_value)
        if not success:
            print(f"\n!!!!!! Experiment '{exp_config['name']}' with seed {seed_value} failed. Aborting process. !!!!!!")
            break
            
    print("\n\n" + "="*30 + f" All Training with seed {seed_value} Complete " + "="*30)

if __name__ == "__main__":
    main()