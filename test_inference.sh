#!/bin/bash

conda run -n progiso python infer_temporal.py --result noof-ultraminiv12 --device cuda --input_data validation -a --format png --report psnr ssim