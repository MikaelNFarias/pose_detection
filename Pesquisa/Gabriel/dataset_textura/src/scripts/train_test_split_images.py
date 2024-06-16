import os
import shutil
import random
import argparse
import sys
sys.path.append('..')
from src.directories import *

def train_test_split_images(source_folder, train_folder, test_folder, split_ratio=0.8):
    # Create the train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # List all the files in the source folder
    all_files = os.listdir(source_folder)
    
    # Filter out non-image files (optional, depending on your file types)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Shuffle the list of files to ensure random split
    random.shuffle(image_files)
    
    # Calculate split index
    split_index = int(len(image_files) * split_ratio)
    
    # Split the files
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]
    
    # Move files to respective directories
    for file_name in train_files:
        shutil.move(os.path.join(source_folder, file_name), os.path.join(train_folder, file_name))
        
    for file_name in test_files:
        shutil.move(os.path.join(source_folder, file_name), os.path.join(test_folder, file_name))
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Testing images: {len(test_files)}")

# Define your source folder and the destination folders for train and test sets
source_folder = DOWNLOADS_DIR
train_folder = os.path.join(BACKGROUNDS_DIR,'train')
test_folder = os.path.join(BACKGROUNDS_DIR,'test')
print(source_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split images into train and test sets')
    parser.add_argument('--source_folder', type=str, help='Path to the source folder containing images')
    parser.add_argument('--train_folder', type=str, help='Path to the folder to save training images')
    parser.add_argument('--test_folder', type=str, help='Path to the folder to save testing images')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio to split the images (default: 0.8)')
    args = parser.parse_args()

    train_test_split_images(source_folder=source_folder,
                            train_folder=train_folder,
                            test_folder=test_folder,
                            split_ratio=0.8)
