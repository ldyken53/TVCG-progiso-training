#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

from doctest import Example
import os
import time
import numpy as np
import torch
import torch.quantization
import torch.onnx
import copy

# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import quant_modules

from config import *
from util import *
from dataset import *
from model.settings import *
from color import *
from result import *
from model.unet import *
from model.wnet import *
from model.temporal_wnet import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Performs inference on a dataset using the specified training result.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Open the result
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')
  print('Result:', cfg.result)

  # Load the result config
  result_cfg = load_config(result_dir)
  # cfg.temp_size = result_cfg.temp_size
  cfg.features = result_cfg.features
  cfg.transfer = result_cfg.transfer
  cfg.model    = result_cfg.model
  # print(result_cfg.model_config)
  if 'model_config' in result_cfg:
    cfg.model_config = result_cfg.model_config
  target_feature = 'hdr' if 'hdr' in cfg.features else 'ldr'
  B, C, W, H = cfg.input_dimensions
  dummy_input = torch.randn(B, C, W, H, device=device)

  # Initialize the model
  model = get_model(cfg)
  # print(model)
  model.to(device)

  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  # Load the trained weights
  load_checkpoint(result_dir, device, cfg.checkpoint, model)
  model.eval()

  if cfg.format == 'onnx':
    dummy_input = torch.randn(B, C, W, H, device=device)
    _, dummy_recurrent = model(dummy_input, None)
    tot = 0
    for feature in dummy_recurrent:
      print(feature.shape)
      print(feature.shape[1] * feature.shape[2] * feature.shape[3])
      tot += feature.shape[1] * feature.shape[2] * feature.shape[3]
    print(tot)

    # enable_onnx_checker needs to be disabled. See notes below.
    torch.onnx.export(model, (dummy_input, dummy_recurrent), os.path.join(result_dir, f"{cfg.result}.onnx"),
                      input_names=["input1", "input2", "input3", "input4", "input5"],
                      output_names=["output1", "output2", "output3", "output4", "output5"],
                      opset_version=11)

  if cfg.format == 'torch':
    traced_model = torch.jit.script(model)
    traced_model.save(os.path.join(result_dir, cfg.result+'_traced.pt'))



if __name__ == '__main__':
  main()
