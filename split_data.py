import os
import shutil
import numpy as np

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits data into train, validation, and test sets without nesting errors.
    
    Args:
    - source_dir: Directory containing class subfolders with data (videos/images)
    - train_dir: Directory to save training data
    - val_dir: Directory to save validation data
    - test_dir: Directory to save test data
    - split_ratio: Tuple specifying the split ratios for train, validation, and test sets
    """
    # Check if split ratios sum to 1
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
    
    # Ensure the destination directories are clean (no nested folders)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create clean directories for train, val, and test
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each class folder
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            np.random.shuffle(files)  # Shuffle files to ensure random distribution
            
            # Determine split sizes
            train_split = int(split_ratio[0] * len(files))
            val_split = int(split_ratio[1] * len(files))
            
            train_files = files[:train_split]
            val_files = files[train_split:train_split + val_split]
            test_files = files[train_split + val_split:]
            
            # Create subdirectories for each class in train, val, and test directories
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Move the files to their respective folders
            for file in train_files:
                shutil.move(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))
            for file in val_files:
                shutil.move(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))
            for file in test_files:
                shutil.move(os.path.join(class_path, file), os.path.join(test_dir, class_name, file))

    print(f"Data split complete. Check {train_dir}, {val_dir}, and {test_dir}.")

# Define source and destination directories
source_folder = '/Users/Dell XPS White/Desktop/dataset'
train_folder = '/Users/Dell XPS White/Desktop/dataset/train'
val_folder = '/Users/Dell XPS White/Desktop/dataset/validate'
test_folder = '/Users/Dell XPS White/Desktop/dataset/test'

# Perform the split with 70% train, 15% validation, and 15% test
split_data(source_folder, train_folder, val_folder, test_folder, split_ratio=(0.7, 0.15, 0.15))
