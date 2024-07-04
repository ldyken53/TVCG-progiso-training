import OpenEXR
import os
from PIL import Image
import array
import sys, shutil
import matplotlib.pyplot as plt
import numpy as np
import random

def png_to_openexr(input_png_path, output_exr_path):
    # Open the PNG image using PIL
    png_image = Image.open(input_png_path)
    width, height = png_image.size
    
    # Gamma correction
    gamma = 2.2 
    pixels = np.power(np.array(png_image.getdata()) / 255.0, gamma)

    # Convert the pixel values to a flat float array for each channel
    r_channel = array.array('f', pixels[:, 0])
    g_channel = array.array('f', pixels[:, 1])
    b_channel = array.array('f', pixels[:, 2])

    exr_file = OpenEXR.OutputFile(output_exr_path, OpenEXR.Header(width, height))
    exr_file.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})
    exr_file.close()

def move_files(directory_path, train_ratio=0.8, test_ratio=0.1):
    # Ensure the provided path is a directory
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    # Create subdirectories if they don't exist
    train_dir = os.path.join(directory_path, "../train-vidw")
    test_dir = os.path.join(directory_path, "../test-vidw")
    validate_dir = os.path.join(directory_path, "../validate-vidw")

    for subdir in [train_dir, test_dir, validate_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    # List all files with the "png" extension
    png_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".png")]

    # Group files by sample_id
    grouped_files = {}
    for file_name in png_files:
        # sample_id = file_name.split('_')[0][-11:]
        sample_id = file_name[file_name.find('seq')+3:file_name.find('seq') + 9]
        if sample_id not in grouped_files:
            grouped_files[sample_id] = []
        grouped_files[sample_id].append(file_name)
    print([len(group) for group in grouped_files.values()])

    # Shuffle the groups randomly
    group_list = list(grouped_files.values())
    random.shuffle(group_list)

    # Calculate the number of groups for each split
    total_groups = len(group_list)
    train_group_count = int(train_ratio * total_groups)
    test_group_count = int(test_ratio * total_groups)

    # Move groups to the respective subdirectories
    for i, group in enumerate(group_list):
        if i % 50 == 0:
            print(i)
        # if (len(group) > 2):
        #     new_group = []
        #     for i in range(len(group)):
        #         num_samples = group[i].split('_')[1][3]
        #         if (num_samples == '1' or num_samples == str(len(group))):
        #             new_group.append(group[i])
        #     group = new_group
        for file_name in group:
            new_file_name = file_name[:-4] + ".ldr.exr"
            if i < train_group_count:
                destination_path = os.path.join(train_dir, new_file_name)
            elif i < train_group_count + test_group_count:
                destination_path = os.path.join(test_dir, new_file_name)
            else:
                destination_path = os.path.join(validate_dir, new_file_name)
            png_to_openexr(os.path.join(directory_path, file_name), destination_path)

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path of the directory you want to traverse
    directory_path = './data/video_data_white'
    move_files(directory_path)
    # png_to_openexr(sys.argv[1], "output.exr")