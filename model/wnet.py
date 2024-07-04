## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet import *
from model.filter_path import *

class WNet(nn.Module):
  def __init__(self, model_config, in_channels):
    super(WNet, self).__init__()
    if 'quantize' in model_config and model_config['quantize']:
      print("Turning on quantization")
    
    self.feature_model = UNet_Features(model_config, in_channels, 3)
    self.filter_model = FilterPath(model_config, num_channels=3, filter_length=3*3)

    # Images must be padded to multiples of the alignment
    self.alignment = 32
    return

  def forward(self, input):
    y, features = self.feature_model(input)
    output = self.filter_model(input[:,0:3,...], features)
    return output
