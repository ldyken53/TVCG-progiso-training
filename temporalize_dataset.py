#!/usr/bin/env python3

import os

from config import *
from util import *
from dataset import *
from image import *

def main():
    # Parse the command line arguments
    cfg = parse_args(
        description='Creates a temporal image from mutliple single frames by stacking them along the y axis.')

    images = [f for f in os.listdir(cfg.input) if f.lower().endswith(".exr")]

    grouped_files = {}
    for file_name in images:
        sample_id = file_name[file_name.find('seq')+3:file_name.find('seq') + 9]
        if sample_id not in grouped_files:
            grouped_files[sample_id] = []
        grouped_files[sample_id].append(file_name)

    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)

    i = 0
    for sample_id, group in grouped_files.items():
        if i % 50 == 0:
            print(f"Iteration {i}")
        input_images = [None for i in range(cfg.num_frames)]
        ref_images = [None for i in range(cfg.num_frames)]
        for image in group:
            infile = os.path.join(cfg.input, image)
            img = load_image(infile, num_channels=3)
            if image.find("ref") != -1:
                ref_images[int(image.split('_')[2]) - 1] = img
            else:
                input_images[int(image.split('_')[2]) - 1] = img
        if np.any(ref_images == None):
            print(f"Skipping {sample_id} because not enough ref images")
        elif np.any(input_images == None):
            print(f"Skipping {sample_id} because not enough input images")
        else:
            outfile_ref = os.path.join(
                cfg.output, f'{group[0][:5]}_seq{sample_id}_ref.ldr.exr')
            outfile_input = os.path.join(
                cfg.output, f'{group[0][:5]}_seq{sample_id}_001spp.ldr.exr')
            input_image = np.concatenate(input_images, axis=0)
            ref_image = np.concatenate(ref_images, axis=0)
            save_image(outfile_input, input_image)
            save_image(outfile_ref, ref_image)

        i += 1


if __name__ == '__main__':
    main()
