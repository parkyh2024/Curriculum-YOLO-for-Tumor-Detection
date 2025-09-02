# Curriculum-YOLO-for-Tumor-Detection

Official code for "Curriculum Learning-Driven YOLO for Accurate Tumor Detection in Ultrasound B-Mode Images Using Hierarchically Zoomed-in Images". This model utilizes curriculum learning and hierarchically zoomed-in images to improve tumor detection accuracy in ultrasound images.

*Figure: The proposed curriculum learning framework, which trains the model progressively from detail-focused zoomed-in images to context-aware full images.*

<br>

## Abstract

Deep learning-based computer-aided diagnosis (CADx) systems show significant promise, but their performance is often hindered by the scarcity of annotated medical imaging data. To address this, we propose an efficient training framework that enhances a YOLO-based object detection model's performance and stability in data-limited settings. The framework incorporates a detail-to-context curriculum learning strategy using hierarchically zoomed-in B-mode images, where learning difficulty is guided by the tumor-to-background area ratio. This approach allows the model to first learn the tumor's morphological features before gradually integrating surrounding contextual information.

<br>

## Key Features

* **Curriculum Learning Strategy:** A novel 'easy-to-hard' training method that begins with tumor-centric `zoom2` images and progressively moves to `zoom3`, `zoom4`, and `full` images. This prevents the model from being confused by complex background information in the initial stages.
* **Enhanced Training Efficiency:** The curriculum learning-driven model achieved 97.2% of its final performance using only 35% of the training data compared to conventional methods.
* **YOLOv5 Integration:** Built upon the YOLOv5 model, which offers a strong balance between accuracy and inference speed, making it suitable for real-time clinical applications.
* **Aspect Ratio-Preserving Preprocessing:** A preprocessing step normalizes all images to a fixed 640×640 resolution while preserving the aspect ratio, which alone improved recall by 2.3% and enhanced model stability.

<br>

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/parkyh2024/Curriculum-YOLO-for-Tumor-Detection.git](https://github.com/parkyh2024/Curriculum-YOLO-for-Tumor-Detection.git)
    cd Curriculum-YOLO-for-Tumor-Detection
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Python 3.8+ is recommended
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

<br>

## Dataset

This study utilizes the **OASBUD (Open Access Series of Breast Ultrasonic Data)**, a publicly available ultrasound image dataset.

* **Download:** The dataset can be downloaded from the official **[OASBUD Dataset](https://zenodo.org/records/545928)**.
* **Setup:** Please download the dataset and structure it as described in the provided data preparation scripts. The scripts assume an `Original` directory containing `images` and `labels` subdirectories.

<br>

## Usage (Workflow)

The entire experimental workflow—from data preparation to training, validation, and benchmarking—is automated with the provided Python scripts.

### Step 1: Data Preparation (Create Zoom-in Datasets)

First, generate the hierarchically zoomed-in datasets required for curriculum learning. This step is a prerequisite for the training process.

* **Create Zoom-in Datasets:**
    Run the `Create_zoom_dataset.py` script. It will prompt you to enter the zoom factors you wish to generate (e.g., '2-4' to create Zoom2, Zoom3, and Zoom4 datasets). This script creates cropped, resized, and padded 640x640 images based on the tumor's bounding box in the original image.
    ```bash
    python Create_zoom_dataset.py
    # Enter the zoom factor(s) to generate (e.g., '2-4' or '2,3,4')
    ```

### Step 2: Automated Curriculum Learning Training

Once the zoom-in datasets are ready, execute the automated curriculum learning pipeline. This script automatically proceeds with training in a predefined sequence (`Zoom2` -> `Zoom3` -> `Zoom4` -> `Resized640`), using the best weights (`best.pt`) from each stage as the initial weights for the next.

* **Run Automated Curriculum Training:**
    Execute the `Start_CL_training.py` script. You will be prompted for a seed value to ensure reproducibility.
    ```bash
    python Start_CL_training.py
    # Enter a seed value (or 'n' for random): 
    ```
    Upon completion, the final model will be saved with a name like `CL_final_seed<your_seed>.pt`.

### Step 3: Model Performance Validation (Accuracy)

After training is complete, evaluate the final model's performance (mAP, Precision, Recall) on the held-out test set.

* **Run Validation:**
    Execute the `Start_CL_val.py` script. You must pass the **same seed value used in Step 2** as the `--seed` argument.
    ```bash
    python Start_CL_val.py --seed YOUR_SEED_VALUE
    # Replace YOUR_SEED_VALUE with the seed used during training (e.g., 123)
    ```
    The script will automatically locate the final model and test image list corresponding to that seed and run YOLOv5's `val.py`.

### Step 4: Benchmark Inference Speed (Optional)

In addition to model accuracy, you can benchmark the inference speed (ms/image and FPS) to assess its performance in a practical clinical setting.

* **Run Benchmark:**
    Execute the `Start_CL_test.py` script. This also requires the **seed value from Step 2**.
    ```bash
    python Start_CL_test.py --seed YOUR_SEED_VALUE
    # Replace YOUR_SEED_VALUE with the seed used during training
    ```
    The script measures the inference time for every image in the specified test set and reports a summary of the average inference time and FPS.

<br>

## Results

Our proposed framework demonstrates a more balanced precision-recall trade-off and enhanced training efficiency. The final model trained for 200 epochs per stage (CL_YOLO_200) showed improved performance over the traditional training approach (Trad_YOLO).

| Model         | Precision | Recall | mAP@0.5 |
| :------------ | :-------: | :----: | :-----: |
| Trad_YOLO     |   0.927   | 0.843  |  0.905  |
| **CL_YOLO_200** | **0.946** | **0.851** | **0.906** |

<br>

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{Park2025Curriculum,
  author  = {Park Yu Hyun, Choi Hongseok, Lee Ki-Baek, Kim Hyungsuk},
  title   = {Curriculum Learning-Driven YOLO for Accurate Tumor Detection in Ultrasound B-Mode Images Using Hierarchically Zoomed-in Images},
  journal = {Applied Sciences},
  year    = {2025},
  volume  = {15},
  pages   = {x}
}
