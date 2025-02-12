#!/bin/bash

conda run -n progiso python preprocess.py ldr --filter RT --train_data training --valid_data validation
conda run -n progiso python train_temporal.py ldr --filter RT --transfer srgb --save_epochs 5 --train_data training --valid_data validation --device cuda --result new-model --config config/train_temporal_wnet_ultramini.json