# ==============================================================================
# ## ���� (Description)
# ==============================================================================
# �� ��ũ��Ʈ�� YOLOv5 ���� Ŀ��ŧ�� �н�(Curriculum Learning)��
# �ڵ����� ���� �����ϱ� ���� ����Ǿ����ϴ�.
#
# �ٽ� ����� �̸� ���ǵ� ����(��: Zoom2 -> Zoom3 -> ...)�� ���� �� �����ͼ���
# �н��� �����ϰ�, ���� �ܰ迡�� �н��� �ֻ��� ����ġ(best.pt)�� ���� �ܰ���
# �ʱ� ����ġ�� ����ϴ� '�̾�޸���' ����� �н��� �����ϴ� ���Դϴ�.
#
# �̸� ���� ����ڴ� �õ�(seed) ���� �Է��ϸ� ��ü Ŀ��ŧ�� �н� ������
# �ڵ����� ������ �� �ֽ��ϴ�.

# ==============================================================================
# ## �ٽ� ���� �帧 (Process Flow)
# ==============================================================================
# 1. ����� �Է� (User Input):
#    - ��ũ��Ʈ ���� �� �������� ���� �õ�(seed) ���� �Է¹޽��ϴ�.
#
# 2. ȯ�� �غ� (Environment Setup):
#    - ���� �н����� ���� �����Ǿ��� �� �ִ� ĳ��(*.cache) ������ �ڵ�����
#      �����Ͽ� ������ ȯ�濡�� �н��� �����մϴ�.
#
# 3. Ŀ��ŧ�� �н� ���� ���� (Sequential Curriculum Learning Execution):
#    - `CURRICULUM_STAGES` ����Ʈ�� ���ǵ� ������� �н��� �����մϴ�.
#    - (ù �ܰ�) `INITIAL_WEIGHTS`(yolov5s.pt)�� ù �н��� �����մϴ�.
#    - (���� �ܰ�) ���� �ܰ��� ������� `best.pt`�� ���� �ܰ��� �Է� ����ġ��
#      ����Ͽ� �н��� �̾�ϴ�.
#    - �� �ܰ�� �н��� �����ϸ� ��ü ���μ����� ��� �ߴܵ˴ϴ�.
#
# 4. ���� ��� ���� (Final Result Saving):
#    - ��� Ŀ��ŧ�� �н��� ���������� �Ϸ�Ǹ�, ���������� ������ `best.pt` ������
#      ������Ʈ �ֻ��� ���(`BASE_PROJECT_PATH`)�� `CL_final_seed{seed}.pt`
#      ������ �̸����� �����Ͽ� �����մϴ�.
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