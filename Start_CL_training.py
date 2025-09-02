# ==============================================================================
# ## 설명 (Description)
# ==============================================================================
# 이 스크립트는 YOLOv5 모델의 커리큘럼 학습(Curriculum Learning)을
# 자동으로 순차 실행하기 위해 설계되었습니다.
#
# 핵심 기능은 미리 정의된 순서(예: Zoom2 -> Zoom3 -> ...)에 따라 각 데이터셋의
# 학습을 진행하고, 이전 단계에서 학습된 최상의 가중치(best.pt)를 다음 단계의
# 초기 가중치로 사용하는 '이어달리기' 방식의 학습을 구현하는 것입니다.
#
# 이를 통해 사용자는 시드(seed) 값만 입력하면 전체 커리큘럼 학습 과정을
# 자동으로 수행할 수 있습니다.

# ==============================================================================
# ## 핵심 동작 흐름 (Process Flow)
# ==============================================================================
# 1. 사용자 입력 (User Input):
#    - 스크립트 실행 시 재현성을 위한 시드(seed) 값을 입력받습니다.
#
# 2. 환경 준비 (Environment Setup):
#    - 이전 학습으로 인해 생성되었을 수 있는 캐시(*.cache) 파일을 자동으로
#      삭제하여 깨끗한 환경에서 학습을 시작합니다.
#
# 3. 커리큘럼 학습 순차 실행 (Sequential Curriculum Learning Execution):
#    - `CURRICULUM_STAGES` 리스트에 정의된 순서대로 학습을 진행합니다.
#    - (첫 단계) `INITIAL_WEIGHTS`(yolov5s.pt)로 첫 학습을 시작합니다.
#    - (이후 단계) 이전 단계의 결과물인 `best.pt`를 다음 단계의 입력 가중치로
#      사용하여 학습을 이어갑니다.
#    - 한 단계라도 학습에 실패하면 전체 프로세스는 즉시 중단됩니다.
#
# 4. 최종 결과 저장 (Final Result Saving):
#    - 모든 커리큘럼 학습이 성공적으로 완료되면, 최종적으로 생성된 `best.pt` 파일을
#      프로젝트 최상위 경로(`BASE_PROJECT_PATH`)에 `CL_final_seed{seed}.pt`
#      형식의 이름으로 복사하여 저장합니다.
# ==============================================================================

import os
import glob
import yaml
import subprocess
import random
import time
import shutil
from sklearn.model_selection import train_test_split

BASE_PROJECT_PATH = "/990pro/parkyh/Ultsound/2025"
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "OASBUD_Dataset_with_Mask_and_Labels")
CONFIG_DIR = os.path.join(BASE_PROJECT_PATH, "training_configs")
INITIAL_WEIGHTS = os.path.join(BASE_PROJECT_PATH, "yolov5s.pt")

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
    with open(train_txt_path, 'w') as f: f.write('\n'.join(train_files))
    with open(val_txt_path, 'w') as f: f.write('\n'.join(val_files))

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
        '--noplots'
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