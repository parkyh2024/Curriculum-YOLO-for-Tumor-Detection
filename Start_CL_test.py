# ==============================================================================
# ## Description
# ==============================================================================
# This script is designed to benchmark the inference performance of a trained
# YOLOv5 model. It systematically measures the time required to perform
# inference on each image within a specified test dataset.
#
# The script is specifically tailored to integrate with the outputs of a
# curriculum learning training process. By providing a 'seed' number that was
# used during training, the script can automatically locate the corresponding
# final model weights (.pt file) and the test image list (.txt file) to
# execute the benchmark.

# ==============================================================================
# ## Process Flow
# ==============================================================================
# 1.  **User Input**:
#     - The script is initiated from the command line and requires a '--seed'
#       argument. This integer seed must correspond to a completed training run.
#
# 2.  **Path and File Configuration**:
#     - Based on the input seed, it constructs the full paths to the required
#       files:
#         - The final trained model weights (e.g., 'CL_final_seed<seed>.pt').
#         - The list of test images (e.g., 'CL_Resized640_seed<seed>_test.txt').
#     - It then performs a critical check to ensure both the model file and the
#       image list exist at the constructed paths before proceeding.
#
# 3.  **Model and Device Initialization**:
#     - The script automatically detects and selects a 'cuda' device if a
#       compatible GPU is available, otherwise it defaults to 'cpu'.
#     - It loads the custom-trained YOLOv5 model from the specified .pt file
#       using a local copy of the YOLOv5 repository.
#     - The model is set to evaluation mode (`model.eval()`).
#
# 4.  **Model Warm-up**:
#     - Before starting the actual benchmark, a "warm-up" phase is executed.
#     - This involves running inference on a dummy tensor (a black image)
#       multiple times. This ensures that any one-time initialization costs,
#       such as CUDA context creation, are completed beforehand and do not
#       contaminate the timing results of the actual images.
#
# 5.  **Inference and Timing Loop**:
#     - The script reads the image list file and iterates through each image path.
#     - For each image, it uses `time.perf_counter()` for high-precision timing.
#       It records the timestamp just before and after the model inference call.
#     - If a GPU is used, `torch.cuda.synchronize()` is called to ensure that
#       all GPU operations for the current inference are complete before
#       recording the end time. This is crucial for accurate measurement.
#     - The elapsed time for each image is calculated and stored in milliseconds.
#
# 6.  **Summary Report**:
#     - After processing all images, the script calculates the average inference
#       time across the entire dataset.
#     - It also computes the average Frames Per Second (FPS) based on this
#       average time.
#     - Finally, it prints a formatted summary report to the console, showing
#       the total number of images processed, the average inference time in ms,
#       and the average FPS.
# ==============================================================================

import torch
import time
import os
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# ## IMPORTANT: Directory Configuration
# ==============================================================================
# Please modify the following paths to match your own directory structure.

# TODO: Change this to the absolute path of your main project directory.
# This is the root folder where your models, data, and configs are stored.
BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"

# TODO: Change this path to point to your local clone of the YOLOv5 repository.
# The script uses this path to load the model architecture.
YOLO_REPO_PATH = os.path.join(BASE_PROJECT_PATH, "yolov5")
# ==============================================================================

# This path is derived from the BASE_PROJECT_PATH.
# Usually, you don't need to change this unless your config files are elsewhere.
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")


def measure_inference_time(seed):
    """
    Measures the inference time using the model and test list corresponding
    to the given seed.
    """
    print(f"?? Starting inference benchmark for seed: {seed}")

    # 1. Set file paths
    model_path = os.path.join(BASE_PROJECT_PATH, f"CL_final_seed{seed}.pt")
    image_list_path = os.path.join(CONFIG_DIR, f"CL_Resized640_seed{seed}_test.txt")

    # 2. Check if files exist
    if not os.path.exists(model_path):
        print(f"? Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(image_list_path):
        print(f"? Error: Image list file not found at '{image_list_path}'")
        return

    print(f"? Model found: {os.path.basename(model_path)}")
    print(f"? Image list found: {os.path.basename(image_list_path)}")

    # 3. Set device (prefer GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"?? Using device: {device.upper()}")

    # 4. Load the model
    try:
        model = torch.hub.load(YOLO_REPO_PATH, 'custom', path=model_path, source='local')
        model.to(device)
        model.eval() # Set to evaluation mode
        print("? Model loaded successfully.")
    except Exception as e:
        print(f"? Error loading model: {e}")
        print("     Please ensure the 'YOLO_REPO_PATH' is set correctly and points to a valid YOLOv5 repository.")
        return

    # 5. Warm up the model to exclude initialization overhead from measurements
    print("?? Warming up the model...")
    # Assume image size is 640, as used during training
    dummy_input = torch.zeros(1, 3, 640, 640).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    print("? Warm-up complete.")

    # 6. Read the list of test images
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if not image_paths:
        print("?? No images found in the test list file.")
        return

    total_images = len(image_paths)
    print(f"\n?? Measuring inference time for {total_images} images...")

    # 7. Perform inference and measure time for each image
    inference_times = []
    with torch.no_grad(): # Disable gradient calculation
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"     ?? Warning: Image not found, skipping: {img_path}")
                continue

            # Use perf_counter for precise timing
            start_time = time.perf_counter()

            # Run model inference
            model(img_path)

            # Wait for all GPU operations to complete for accurate timing
            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Convert time to milliseconds (ms)
            duration_ms = (end_time - start_time) * 1000
            inference_times.append(duration_ms)

            print(f"  [{i+1}/{total_images}] Image: {os.path.basename(img_path):<30} | Time: {duration_ms:.2f} ms")

    # 8. Print the summary of results
    if not inference_times:
        print("\n?? No images were processed. Cannot generate summary.")
        return

    avg_time_ms = np.mean(inference_times)
    fps = 1000 / avg_time_ms

    print("\n" + "="*40)
    print("?? Inference Benchmark Summary ??")
    print("="*40)
    print(f"Total Images Processed : {len(inference_times)}")
    print(f"Average Inference Time : {avg_time_ms:.2f} ms")
    print(f"Average FPS (Frames Per Second) : {fps:.2f} FPS")
    print("="*40)


if __name__ == "__main__":
    # Setup to receive the seed value from the command line
    parser = argparse.ArgumentParser(description="Measure YOLOv5 inference time for a list of images.")
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed value used during training to locate the model and test list.')

    args = parser.parse_args()

    measure_inference_time(args.seed)

