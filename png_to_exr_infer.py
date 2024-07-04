import OpenEXR
import os
from PIL import Image
import array
import sys, shutil
import matplotlib.pyplot as plt
import numpy as np
import random

def png_to_openexr(input_png_path, output_exr_path):
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

def move_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    # Create subdirectories if they don't exist
    proc_dir = os.path.join(directory_path, "proc")
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
    png_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".png")]

    # Group files by sample_id
    grouped_files = {}
    for file_name in png_files[:]:
        # sample_id = file_name.split('_')[0][-11:]
        seq_id = file_name[file_name.find('seq')+3:file_name.find('seq') + 9]
        pass_id = file_name.split('_')[2]
        image_completeness = (int(file_name.split('_')[3]) // 5) * 5
        if seq_id not in grouped_files:
            grouped_files[seq_id] = {}
        if pass_id not in grouped_files[seq_id]:
            grouped_files[seq_id][pass_id] = {}
        if not grouped_files[seq_id][pass_id].get(image_completeness):
            grouped_files[seq_id][pass_id][image_completeness] = file_name
    completeness_groups = [85, 90, 95, 100]
    i = 0
    to_remove = []
    for k, v in grouped_files.items():
        test = True
        for k1, v1 in v.items():
            for group in completeness_groups:
                if not v1.get(group):
                    test = False
                    to_remove.append(k)
                    break
            if not test:
                break
        if test:
            i += 1
    for k in to_remove:
        del grouped_files[k]
    print(len(grouped_files.keys()))
    print(i)
    for group in completeness_groups:
        group_dir = os.path.join(proc_dir, str(group))
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
    i = 0
    for elem in grouped_files.values():
        print(i)
        for elem2 in elem.values():
            for comp, file in elem2.items():
                if comp in completeness_groups:
                    gs = file.split('_')
                    if comp == 100:
                        new_filename = gs[0] + '_' + gs[1] + '_' + gs[2] + '_ref.ldr.exr'
                        file_path = os.path.join(os.path.join(proc_dir, str(comp)), new_filename)
                        png_to_openexr(os.path.join(directory_path, file), file_path)
                        for g in completeness_groups[:-1]:
                            shutil.copyfile(file_path, os.path.join(os.path.join(proc_dir, str(g)), new_filename))
                    else:
                        new_filename = gs[0] + '_' + gs[1] + '_' + gs[2] + '_001spp.ldr.exr'
                        png_to_openexr(os.path.join(directory_path, file), os.path.join(os.path.join(proc_dir, str(comp)), new_filename))
        i += 1

    # Shuffle the groups randomly
    # group_list = list(grouped_files.values())
    # random.shuffle(group_list)

    # Calculate the number of groups for each split
    # total_groups = len(group_list)

    # Move groups to the respective subdirectories
    # for i, group in enumerate(group_list):
    #     if i % 50 == 0:
    #         print(i)
        # if (len(group) > 2):
        #     new_group = []
        #     for i in range(len(group)):
        #         num_samples = group[i].split('_')[1][3]
        #         if (num_samples == '1' or num_samples == str(len(group))):
        #             new_group.append(group[i])
        #     group = new_group
        # for file_name in group:
        #     new_file_name = file_name[:-4] + ".ldr.exr"
        #     destination_path = os.path.join(proc_dir, new_file_name)
        #     png_to_openexr(os.path.join(directory_path, file_name), destination_path)

if __name__ == "__main__":
    # Replace 'directory_path' with the path of the directory you want to traverse
    directory_path = './test-vid'
    move_files(directory_path)
    # png_to_openexr(sys.argv[1], "output.exr")