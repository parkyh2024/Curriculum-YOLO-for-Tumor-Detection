# ==============================================================================
# ## Description
# ==============================================================================
# This script automates a sequential curriculum learning (CL) process for training
# a YOLOv5 model. The core concept is to conduct training in a "relay" fashion,
# progressing through a predefined sequence of datasets or "stages"
# (e.g., from easier, zoomed-in images to more complex, standard-sized images).
#
# For each stage, the script prepares the dataset, initiates YOLOv5 training,
# and then uses the best-performing weights (`best.pt`) from that stage as the
# starting point for the subsequent stage. This automates the entire pipeline,
# requiring only an initial seed value from the user to ensure reproducibility.

# ==============================================================================
# ## Process Flow
# ==============================================================================
# 1.  **Initialization and User Input**:
#     - The script starts by defining global constants for paths, training
#       parameters (image size, epochs, etc.), and the curriculum sequence.
#     - It prompts the user to enter a seed value for reproducibility or to
#       generate a random one.
#
# 2.  **Environment Cleanup**:
#     - Before starting, it scans the configuration directory for any leftover
#       dataset cache files (`*.cache`) from previous runs and deletes them
#       to prevent potential conflicts and ensure a clean state.
#
# 3.  **Sequential Training Loop**:
#     - The script iterates through the `CURRICULUM_STAGES` list, executing one
#       training stage at a time.
#     - **For the first stage**: It uses a pretrained `yolov5s.pt` as the
#       initial weights.
#     - **For all subsequent stages**: It uses the `best.pt` file generated
#       from the immediately preceding stage as the initial weights.
#
# 4.  **Per-Stage Execution (`run_training_stage` function)**:
#     - **Dataset Preparation**: For the current stage, it locates the image
#       directory, splits all images into training (65%), validation (15%), and
#       test (20%) sets using the provided seed. It then writes these file
#       paths into `.txt` files.
#     - **YAML Configuration**: It generates a YOLOv5-compatible `.yaml` dataset
#       configuration file on the fly. This file points to the newly created
#       train/val `.txt` lists.
#     - **Training Command**: It constructs and executes the `yolov5/train.py`
#       command as a separate process using `subprocess.run()`. All necessary
#       parameters like image size, batch size, epochs, weights, and the path
#       to the generated YAML file are passed as command-line arguments.
#     - **Validation**: After the training subprocess completes, it checks the
#       exit code. If the training failed, the entire script aborts.
#     - **Weight Retrieval**: If training was successful, it locates the path to
#       the resulting `best.pt` file within the output directory
#       (`yolov5/runs/train/...`) and returns this path.
#
# 5.  **Finalization**:
#     - After the last stage in the curriculum completes successfully, the loop
#       terminates.
#     - The script takes the final `best.pt` from the last stage and copies it
#       to the main project directory (`BASE_PROJECT_PATH`).
#     - The final model is renamed to include the seed value, for example,
#       `CL_final_seed123.pt`, for easy identification and later use in testing.
#     - A summary of the total execution time is printed.
# ==============================================================================

import os
import glob
import yaml
import subprocess
import random
import time
import shutil
from sklearn.model_selection import train_test_split

# ==============================================================================
# ## IMPORTANT: Directory Configuration
# ==============================================================================
# Please modify the following paths to match your own directory structure.

# TODO: Change this to the absolute path of your main project directory.
# This is the root folder where your models, data, and configs are stored.
BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"

# TODO: Change this to the path where your dataset is located.
# It is currently set relative to BASE_PROJECT_PATH.
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "OASBUD_Dataset_with_Mask_and_Labels")

# TODO: Change this to the path of your initial pretrained weights file.
# This is typically 'yolov5s.pt' or another base model.
INITIAL_WEIGHTS = os.path.join(BASE_PROJECT_PATH, "yolov5s.pt")
# ==============================================================================

# This path is derived from the BASE_PROJECT_PATH.
# Usually, you don't need to change this unless your config files are elsewhere.
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")


IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 200
NUM_WORKERS = 8

CURRICULUM_STAGES = [
    'Zoom2_640',
    'Zoom3_640',
    'Zoom4_640',
    'Resized640'
]

def run_training_stage(stage_name, input_weights_path, seed, epochs):
    """
    Executes a single stage of the curriculum learning process.
    This includes dataset preparation, YAML file creation, and running YOLOv5 training.
    """
    run_name = f"CL_{stage_name}_seed{seed}"
    print(f"\n{'='*30}\n[ Stage Start: {stage_name} ]\n{'='*30}")
    print(f"Run Name: {run_name}")
    print(f"Input Weights: {os.path.basename(input_weights_path)}")

    print(f"\n[1/3] Preparing dataset for '{stage_name}'...")
    image_dir = os.path.join(BASE_DATA_PATH, stage_name, "images")
    all_image_paths = glob.glob(os.path.join(image_dir, '*.png'))

    if not all_image_paths:
        print(f"!!!!!!!!!! No images found in '{image_dir}'. !!!!!!!!!!!")
        return None

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
        'path': os.path.abspath(BASE_DATA_PATH),
        'train': os.path.relpath(train_txt_path, os.path.abspath(BASE_DATA_PATH)),
        'val': os.path.relpath(val_txt_path, os.path.abspath(BASE_DATA_PATH)),
        'nc': 2, 'names': ['Benign', 'Malignant']
    }
    with open(yaml_path, 'w') as f: yaml.dump(yaml_data, f, sort_keys=False)
    print(" -> Dataset preparation complete.")

    print(f"\n[2/3] Starting training for '{run_name}'...")
    stage_start_time = time.time()

    train_command = [
      'python3', 'yolov5/train.py',
      '--img', str(IMG_SIZE),
      '--batch', str(BATCH_SIZE),
      '--epochs', str(epochs),
      '--data', yaml_path,
      '--weights', input_weights_path,
      '--name', run_name,
      '--workers', str(NUM_WORKERS),
      '--seed', str(seed),
      '--noplots',
      '--patience', '0'
    ]
    train_result = subprocess.run(train_command)

    elapsed_time = time.time() - stage_start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)
    print(f" -> Stage training time: {hours}h {minutes}m {seconds}s")

    if train_result.returncode != 0:
        print(f"!!!!!!!!!! Training failed for '{run_name}' !!!!!!!!!!!")
        return None

    print(f"\n[3/3] Finding best weights for the next stage...")
    output_weights_path = os.path.join(BASE_PROJECT_PATH, 'yolov5/runs/train', run_name, 'weights/best.pt')

    if os.path.exists(output_weights_path):
        print(f" -> Success! Found best weights at: {output_weights_path}")
        return output_weights_path
    else:
        print(f"!!!!!!!!!! Could not find 'best.pt' at expected path: {output_weights_path} !!!!!!!!!!!")
        return None

def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)

    seed_input = input(f"Enter a seed value (or 'n' for random): ")
    seed = random.randint(0, 9999) if seed_input.lower() == 'n' else int(seed_input)
    print(f"Using Seed: {seed}")

    print("\n[Cleanup] Deleting old cache files from config directory...")
    cache_files = glob.glob(os.path.join(CONFIG_DIR, '*.cache'))
    if cache_files:
        for f in cache_files:
            os.remove(f)
        print(f" -> Deleted {len(cache_files)} cache file(s).")
    else:
        print(" -> No old cache files to delete.")

    print("\nStarting Curriculum Learning Process...")
    print("Curriculum Order:", " -> ".join(CURRICULUM_STAGES))

    total_start_time = time.time()

    latest_weights_path = INITIAL_WEIGHTS

    for i, stage_name in enumerate(CURRICULUM_STAGES):
        result_weights_path = run_training_stage(
            stage_name=stage_name,
            input_weights_path=latest_weights_path,
            seed=seed,
            epochs=EPOCHS
        )

        if result_weights_path is None:
            print(f"\n!!!!!! Curriculum learning failed at stage '{stage_name}'. Aborting. !!!!!!" )
            return

        latest_weights_path = result_weights_path

    total_elapsed_time = time.time() - total_start_time
    m, s = divmod(int(total_elapsed_time), 60)
    h, m = divmod(m, 60)

    print(f"\n{'='*30}\n## All Curriculum Learning Stages Completed Successfully! ##\n{'='*30}")
    print(f"Final model weights are located at: {latest_weights_path}")
    print(f"Total Execution Time: {h}h {m}m {s}s")

    print(f"\n[Final Step] Copying final weights to project directory...")
    final_filename = f"CL_final_seed{seed}.pt"
    destination_path = os.path.join(BASE_PROJECT_PATH, final_filename)

    try:
        shutil.copy(latest_weights_path, destination_path)
        print(f" -> Success! Final model saved as: {destination_path}")
    except Exception as e:
        print(f" -> Error copying final weights: {e}")

if __name__ == "__main__":
    main()

