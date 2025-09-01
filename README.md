# Curriculum-YOLO-for-Tumor-Detection

Official code for "Curriculum Learning-Driven YOLO for Accurate Tumor Detection in Ultrasound B-Mode Images Using Hierarchically Zoomed-in Images". This model utilizes curriculum learning and hierarchically zoomed-in images to improve tumor detection accuracy in ultrasound images.

*Figure: The proposed curriculum learning framework, which trains the model progressively from detail-focused zoomed-in images to context-aware full images.*

<br>

## 📖 Abstract

Deep learning-based computer-aided diagnosis (CADx) systems show significant promise, but their performance is often hindered by the scarcity of annotated medical imaging data. To address this, we propose an efficient training framework that enhances a YOLO-based object detection model's performance and stability in data-limited settings. The framework incorporates a detail-to-context curriculum learning strategy using hierarchically zoomed-in B-mode images, where learning difficulty is guided by the tumor-to-background area ratio. This approach allows the model to first learn the tumor's morphological features before gradually integrating surrounding contextual information.

<br>

## 🚀 Key Features

* **Curriculum Learning Strategy:** A novel 'easy-to-hard' training method that begins with tumor-centric `zoom2` images and progressively moves to `zoom3`, `zoom4`, and `full` images. This prevents the model from being confused by complex background information in the initial stages.
* **Enhanced Training Efficiency:** The curriculum learning-driven model achieved 97.2% of its final performance using only 35% of the training data compared to conventional methods.
* **YOLOv5 Integration:** Built upon the YOLOv5 model, which offers a strong balance between accuracy and inference speed, making it suitable for real-time clinical applications.
* **Aspect Ratio-Preserving Preprocessing:** A preprocessing step normalizes all images to a fixed 640×640 resolution while preserving the aspect ratio, which alone improved recall by 2.3% and enhanced model stability.

<br>

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/parkyh2024/Curriculum-YOLO-for-Tumor-Detection.git](https://github.com/parkyh2024/Curriculum-YOLO-for-Tumor-Detection.git)
    cd Curriculum-YOLO-for-Tumor-Detection
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Python 3.8+ is recommended
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

<br>

## 📊 Dataset

This study utilizes the **OASBUD (Open Access Series of Breast Ultrasonic Data)**, a publicly available ultrasound image dataset.

* **Download:** The dataset can be downloaded from the official **[OASBUD GitHub Repository](https://github.com/oasbud/oasbud)**.
* **Setup:** Please download the dataset and structure it as described in the provided data preparation scripts. The scripts assume an `Original` directory containing `images` and `labels` subdirectories.

<br>

## 📖 Usage (Workflow)

The entire experimental workflow—from data preparation to training and validation—is automated with the provided Python scripts.

### Step 1: Data Preparation & Augmentation

First, generate the hierarchically zoomed-in datasets required for curriculum learning.

* **Create Zoom-in Datasets:**
    Run the `Create_zoom_dataset.py` script. It will prompt you to enter the zoom factors you wish to generate (e.g., '2-4' to create Zoom2, Zoom3, and Zoom4 datasets). This script creates cropped, resized, and padded 640x640 images based on the tumor's bounding box in the original image.
    ```bash
    python Create_zoom_dataset.py
    # Enter the zoom factor(s) to generate (e.g., '2-4' or '2,3,4')
    ```

* **(Optional) Create Cropped Datasets:**
    To remove image borders (e.g., black bars from the ultrasound machine), run `crop.py`.
    ```bash
    python crop.py
    # Enter the number of pixels to crop from each edge
    ```

### Step 2: Model Training

The training process is automated to handle multiple datasets sequentially.

* **Run Automated Training:**
    Execute the `Start_training.py` script. It will automatically discover all prepared datasets (`Original`, `Zoom2_640`, etc.) and train a YOLOv5 model on each one sequentially. You will be prompted for a single seed value to ensure reproducibility across experiments.
    ```bash
    python Start_training.py
    # Enter a 4-digit seed (0-9999), or 'n' for a random seed
    ```
    For the curriculum learning approach, the best weights (`best.pt`) from one stage should be used as the starting weights for the next. This script can be adapted to automate this transfer process.

### Step 3: Model Performance Validation

After training is complete, evaluate the final models on the held-out test set.

* **Run Validation:**
    Execute the `Start_val.py` script. It will scan for all completed training runs and present you with a numbered list.
    ```bash
    python Start_val.py
    ```
    Select the model(s) you wish to evaluate. The script will then run `yolov5/val.py` with the `--task test` flag to measure the model's generalization performance.

<br>

## 📈 Results

Our proposed framework demonstrates a more balanced precision-recall trade-off and enhanced training efficiency. The final model trained for 200 epochs per stage (CL_YOLO_200) showed improved performance over the traditional training approach (Trad_YOLO).

| Model | Precision | Recall | mAP@0.5 |
| :--- | :---: | :---: | :---: |
| Trad_YOLO | 0.927 | 0.843 | 0.905 |
| **CL_YOLO_200**| **0.946** | **0.851** | **0.906** |

<br>

## 📜 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{Park2025Curriculum,
  author  = {Park, Yu Hyun and Choi, Hongseok and Lee, Ki-Baek and Kim, Hyungsuk},
  title   = {Curriculum Learning-Driven YOLO for Accurate Tumor Detection in Ultrasound B-Mode Images Using Hierarchically Zoomed-in Images},
  journal = {Applied Sciences},
  year    = {2025},
  volume  = {15},
  pages   = {x}
}
