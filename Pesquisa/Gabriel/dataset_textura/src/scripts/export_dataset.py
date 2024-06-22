import os
import shutil
import sys
import argparse

sys.path.append('..')
from src.directories import DATA_DIR, DATASET_FINAL_DIR, TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR, DATASET_FINAL_DIR_TRAIN, DATASET_FINAL_DIR_TEST


def export_dataset(src, dest) -> None:
    # Ensure destination directory exists
    if not os.path.exists(dest):
        os.makedirs(dest)

    files = os.listdir(src)
    for file in files:
        src_file_path = os.path.join(src, file)
        dest_file_path = os.path.join(dest, file)

        if os.path.isdir(src_file_path):
            if os.path.exists(dest_file_path):
                shutil.rmtree(dest_file_path)  # Remove the existing directory
            shutil.copytree(src=src_file_path, dst=dest_file_path)
        else:
            shutil.copy2(src=src_file_path, dst=dest_file_path)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Export dataset")
    arg_parser.add_argument('--dataset-type', type=str, required=True, help='Specify dataset type: train or test')
    args = arg_parser.parse_args()

    if args.dataset_type == 'train':
        export_dataset(src=TRAIN_OUTPUT_DIR, dest=DATASET_FINAL_DIR_TRAIN)
    elif args.dataset_type == 'test':
        export_dataset(src=TEST_OUTPUT_DIR, dest=DATASET_FINAL_DIR_TEST)
    else:
        raise ValueError('Invalid dataset type')
