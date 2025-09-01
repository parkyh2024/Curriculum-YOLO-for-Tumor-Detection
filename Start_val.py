# =================================================================================================
# Description
# =================================================================================================
# This script is an interactive tool for running validation (Testing) on previously
# completed YOLOv5 model training sessions.
#
# It automatically discovers all training results saved in the `yolov5/runs/train` folder
# and displays them as a list to the user. The user can select one or more models from this
# list for validation, and the script will run the `yolov5/val.py` script for each
# selected model to evaluate its performance.
#
# Specifically, the purpose of this script is to measure the generalization performance
# of the final model using the 'test set' that was held out during the model training phase.

# =================================================================================================
# Process Flow
# =================================================================================================
# 1. Discover and List Completed Trainings:
#    - The script scans the `yolov5/runs/train` directory to get a list of all
#      previously completed training session folders.
#    - It prints the discovered list of training sessions with corresponding numbers for the user.
#
# 2. User Selection:
#    - It prompts the user to enter the number(s) of the training session(s) they want
#      to validate.
#    - It can handle various input formats, such as a single number (e.g., 5), a range
#      (e.g., 1-3), or a comma-separated list (e.g., 1,3,5).
#
# 3. Run Validation (Iterates for each selected item):
#    - For each training session selected by the user, it performs the following steps:
#      a. Set Paths: It identifies the path to the best-performing weights file
#         (`weights/best.pt`) within the selected training folder and the path to the
#         corresponding dataset's YAML configuration file generated during the training step.
#      b. Generate Validation Command: It constructs the command to run `yolov5/val.py`.
#         The `--task test` argument is used to specify that the evaluation should be
#         performed on the 'test dataset', which was not used during training.
#      c. Execute Command: It executes the generated command using `subprocess` to start
#         the validation. The validation results are saved under a new name in the
#         `yolov5/runs/val/` folder.
#
# 4. Completion:
#    - Once validation for all selected items is finished, it prints a message
#      indicating that the entire process is complete.
# =================================================================================================

import os
import glob
import argparse
import subprocess

BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"
TRAIN_RUNS_PATH = os.path.join(BASE_PROJECT_PATH, 'yolov5/runs/train')
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")

def parse_selection_input(input_str, max_value):
    indices = set()
    input_str = input_str.strip()
    try:
        if '-' in input_str:
            parts = input_str.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                if 1 <= start <= end <= max_value:
                    for i in range(start, end + 1):
                        indices.add(i - 1)
                    return sorted(list(indices))
        elif ',' in input_str:
            parts = [int(p.strip()) for p in input_str.split(',')]
            if all(1 <= p <= max_value for p in parts):
                for p in parts:
                    indices.add(p - 1)
                return sorted(list(indices))
        elif input_str.isdigit():
            num = int(input_str)
            if 1 <= num <= max_value:
                indices.add(num - 1)
                return sorted(list(indices))
    except (ValueError, IndexError):
        return None
    return None


def run_validation(selected_run, args):
    print(f"\n -> Starting validation for '{selected_run}'")

    weights_path = os.path.join(TRAIN_RUNS_PATH, selected_run, 'weights/best.pt')
    yaml_path = os.path.join(CONFIG_DIR, f"{selected_run}_dataset.yaml")

    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found -> {weights_path}")
        return
    if not os.path.exists(yaml_path):
        print(f"Error: YAML config file not found -> {yaml_path}")
        print("Ensure you have run the training script with the corresponding dataset and seed.")
        return

    val_command = [
        'python3', 'yolov5/val.py',
        '--weights', weights_path,
        '--data', yaml_path,
        '--task', 'test',
        '--imgsz', str(args.img_size),
        '--name', f'{selected_run}_test_results'
    ]
    print(f" -> Executing command: {' '.join(val_command)}")
    subprocess.run(val_command)

    print(f"\n========== Validation for '{selected_run}' is complete. ==========")


def main(args):
    try:
        completed_runs = sorted([d for d in os.listdir(TRAIN_RUNS_PATH) if os.path.isdir(os.path.join(TRAIN_RUNS_PATH, d))])
    except FileNotFoundError:
        print(f"Error: Training runs folder not found -> {TRAIN_RUNS_PATH}")
        return
        
    if not completed_runs:
        print("Error: No completed training sessions found.")
        return

    print("===== Select training session(s) to validate =====")
    for i, run_name in enumerate(completed_runs):
        print(f"  [{i+1}] {run_name}")
    
    while True:
        user_input = input(f"\nEnter number(s) (e.g., '5', '1-3', or '1,3,5'): ")
        selected_indices = parse_selection_input(user_input, len(completed_runs))
        
        if selected_indices is not None:
            break
        else:
            print("Invalid input. Please try again.")
    
    for index in selected_indices:
        run_to_validate = completed_runs[index]
        run_validation(run_to_validate, args)

    print("\n\n" + "="*25 + " All Selected Validations Complete " + "="*25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Interactive Validation Launcher")
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    args = parser.parse_args()
    main(args)