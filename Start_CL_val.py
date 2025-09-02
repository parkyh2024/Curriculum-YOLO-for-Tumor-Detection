# ==============================================================================
# ## Description
# ==============================================================================
# This script is designed to run the validation process using a final model
# trained by the 'Start_CL_training.py' script. It leverages YOLOv5's built-in
# 'val.py' script to evaluate the model's performance on a specific test set.
#
# The primary purpose is to automate the validation step. By providing the same
# seed value that was used for training, this script automatically locates the
# final curriculum learning model (`CL_final_seed<seed>.pt`) and the
# corresponding test image list. It then constructs the necessary configuration
# files on the fly and executes the validation command.

# ==============================================================================
# ## Process Flow
# ==============================================================================
# 1.  **User Input**:
#     - The script is executed from the command line and requires a `--seed`
#       argument. This seed must be the same one used to generate the model
#       with the training script.
#
# 2.  **Path and File Configuration**:
#     - It constructs the paths to the final model weights and the test image
#       list based on the provided seed.
#     - It performs a check to ensure that both the model file and the test
#       list file actually exist before proceeding.
#
# 3.  **Dynamic YAML Configuration**:
#     - YOLOv5's `val.py` requires a dataset `.yaml` file to locate the images.
#       This script generates this YAML file dynamically.
#     - The generated YAML file will contain the path to the dataset and,
#       crucially, will point its 'val' field to the test set file list,
#       ensuring `val.py` evaluates on our designated test images.
#
# 4.  **Validation Command Execution**:
#     - The script builds the full command to execute `yolov5/val.py`.
#     - This command includes arguments for the model weights (`--weights`),
#       the path to the dynamically generated YAML file (`--data`), image size
#       (`--imgsz`), and a unique run name (`--name`).
#     - The command is then executed as a subprocess. The output, metrics
#       (like mAP, precision, recall), and any generated plots from `val.py`
#       will be saved to a new directory within `yolov5/runs/val/`.
# ==============================================================================

import os
import yaml
import subprocess
import argparse

# ==============================================================================
# ## IMPORTANT: Directory Configuration
# ==============================================================================
# Please modify the following paths to match your own directory structure.

# TODO: Change this to the absolute path of your main project directory.
# This is the root folder where your models, data, and configs are stored.
BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"

# TODO: Change this to the path where your dataset is located.
# This is needed to correctly configure the dataset YAML file.
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "OASBUD_Dataset_with_Mask_and_Labels")
# ==============================================================================

# This path is derived from the BASE_PROJECT_PATH.
# Usually, you don't need to change this unless your config files are elsewhere.
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")
IMG_SIZE = 640

def run_validation(seed):
    """
    Finds the final model and test set for a given seed, then runs
    YOLOv5's val.py for evaluation.
    """
    print(f"?? Starting validation process for seed: {seed}")
    run_name = f"CL_validation_seed{seed}"

    # 1. Define paths for the model and the test set list
    model_path = os.path.join(BASE_PROJECT_PATH, f"CL_final_seed{seed}.pt")
    test_list_path = os.path.join(CONFIG_DIR, f"CL_Resized640_seed{seed}_test.txt")

    # 2. Verify that the required files exist
    if not os.path.exists(model_path):
        print(f"!! ERROR: Model file not found at '{model_path}'")
        print("!! Please ensure you have successfully run 'Start_CL_training.py' with this seed.")
        return
    if not os.path.exists(test_list_path):
        print(f"!! ERROR: Test image list not found at '{test_list_path}'")
        return

    print(f"? Found model: {os.path.basename(model_path)}")
    print(f"? Found test list: {os.path.basename(test_list_path)}")

    # 3. Create a temporary dataset.yaml file for the val.py script
    print(f"\n[1/2] Generating temporary dataset YAML file...")
    yaml_path = os.path.join(CONFIG_DIR, f"{run_name}_dataset.yaml")
    yaml_data = {
        'path': os.path.abspath(BASE_DATA_PATH),
        # YOLOv5's check_dataset utility requires the 'train' key to be present.
        # For validation-only tasks, we can point both 'train' and 'val'
        # to the same test set list.
        'train': os.path.relpath(test_list_path, os.path.abspath(BASE_DATA_PATH)),
        'val': os.path.relpath(test_list_path, os.path.abspath(BASE_DATA_PATH)),
        'nc': 2,
        'names': ['Benign', 'Malignant']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f" -> YAML file created at: {yaml_path}")

    # 4. Construct and run the validation command
    print(f"\n[2/2] Executing yolov5/val.py...")
    validation_command = [
        'python3', 'yolov5/val.py',
        '--weights', model_path,
        '--data', yaml_path,
        '--imgsz', str(IMG_SIZE),
        # '--task' argument is removed. 'val.py' will default to using the 'val' key
        # from the YAML file, which correctly points to our test list.
        '--name', run_name
    ]

    print("   Command: " + " ".join(validation_command))
    try:
        subprocess.run(validation_command, check=True)
        print("\n? Validation completed successfully.")
        print(f"? Results are saved in 'yolov5/runs/val/{run_name}'")
    except subprocess.CalledProcessError as e:
        print(f"\n!! ERROR: Validation script failed with exit code {e.returncode}.")
    except FileNotFoundError:
        print("\n!! ERROR: 'python3' or 'yolov5/val.py' not found.")
        print("!! Please ensure you are in the correct directory and YOLOv5 is set up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5 validation on a trained curriculum learning model.")
    parser.add_argument('--seed', type=int, required=True,
                        help='The seed value used during training to locate the final model and test list.')
    args = parser.parse_args()

    run_validation(args.seed)

