import os
import shutil
import random

def move_files(directory_path, train_ratio=0.8, test_ratio=0.1, validate_ratio=0.1):
    # Ensure the provided path is a directory
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    # Create subdirectories if they don't exist
    train_dir = os.path.join(directory_path, "../train")
    test_dir = os.path.join(directory_path, "../test")
    validate_dir = os.path.join(directory_path, "../validate")

    for subdir in [train_dir, test_dir, validate_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    # List all files with the "exr" extension
    exr_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".exr")]

    # Group files by sample_id
    grouped_files = {}
    for file_name in exr_files:
        sample_id = file_name.split('_')[0][-4:]
        if sample_id not in grouped_files:
            grouped_files[sample_id] = []
        grouped_files[sample_id].append(file_name)

    # Shuffle the groups randomly
    group_list = list(grouped_files.values())
    random.shuffle(group_list)

    # Calculate the number of groups for each split
    total_groups = len(group_list)
    train_group_count = int(train_ratio * total_groups)
    test_group_count = int(test_ratio * total_groups)
    validate_group_count = total_groups - train_group_count - test_group_count

    # Move groups to the respective subdirectories
    for i, group in enumerate(group_list):
        for file_name in group:
            source_path = os.path.join(directory_path, file_name)

            if i < train_group_count:
                destination_path = os.path.join(train_dir, file_name)
            elif i < train_group_count + test_group_count:
                destination_path = os.path.join(test_dir, file_name)
            else:
                destination_path = os.path.join(validate_dir, file_name)

            shutil.move(source_path, destination_path)

if __name__ == "__main__":
    # Replace 'directory_path' with the path of the directory containing the EXR files
    directory_path = './data/iso-data'
    
    # Specify the desired train, test, and validate ratios
    move_files(directory_path, train_ratio=0.8, test_ratio=0.1, validate_ratio=0.1)