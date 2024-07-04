## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun

from typing import Optional, List

from model.unet import *
from model.filter_path import *

from model.common import *

import sys

class TemporalWNet(nn.Module):
  def __init__(self, model_config, in_channels):
    super(TemporalWNet, self).__init__()
    if 'quantize' in model_config and model_config['quantize']:
      print("Turning on quantization")
    
    self.feature_model = UNet_Features_Recurrent(model_config, in_channels, 3)
    #self.feature_model = UNet_Features_Recurrent(model_config, in_channels + 3 + 3, 3, predict_temporal_blend_weights=False) # +3 for recurrent input + 3 for additional channels
    #self.feature_model = UNet_Features_Recurrent(model_config, in_channels + 3, 3, predict_temporal_blend_weights=False) # +3 for recurrent input
    #self.feature_model = UNet_Features(model_config, in_channels, 3)

    #self.filter_model = TemporalFilterPath(model_config, num_channels=3, filter_length=3*3, num_stacks=2)
    self.filter_model = FilterPath(model_config, num_channels=3, filter_length=3*3, num_stacks=1)
    #self.filter_model = FilterPath(model_config, num_channels=3, filter_length=3*3, num_stacks=2)

    # Images must be padded to multiples of the alignment
    self.alignment = 32
    return

  def forward(self, input, recurrent_state : Optional[List[torch.Tensor]]):

    # ----
    # Recurrent Features & Temporal Blending
    # ----

    # EPS = 10e-8

    # if recurrent_state is None:
    #   B,C,H,W = input.size()
    #   prev_output = torch.zeros((B,3,H,W), device=input.device, dtype=input.dtype)
    #   #y, features = self.feature_model(torch.cat([input, prev_output], dim=1), None)
    #   y, features = self.feature_model(input, None)

    #   y = y + EPS
    #   for i in range(len(features)):
    #     features[i] = features[i] + EPS

    #   recurrent_state = list()
    #   recurrent_state.append(y)
    #   for feature in features:
    #     recurrent_state.append(feature)

    #   output, _ = self.filter_model(y, torch.zeros_like(y, device=y.device), features)

    # else:
    #   prev_output = recurrent_state[0]
    #   #y, features = self.feature_model(torch.cat([input, prev_output], dim=1), recurrent_state[1:])
    #   y, features = self.feature_model(input, recurrent_state[1:])

    #   y = y + EPS
    #   for i in range(len(features)):
    #     features[i] = features[i] + EPS
      
    #   recurrent_state = list()
    #   recurrent_state.append(y)
    #   for feature in features:
    #     recurrent_state.append(feature)

    #   output, _ = self.filter_model(y, prev_output, features)


    # output = output + EPS
    # recurrent_state[0] = output

    # return output, recurrent_state


    # ----
    # Recurrent Features
    # ----

    EPS = 10e-8

    if recurrent_state is None:
      B,C,H,W = input.size()
      prev_output = torch.zeros((B,3,H,W), device=input.device, dtype=input.dtype)
      #y, features = self.feature_model(torch.cat([input, prev_output], dim=1), None)
      y, features = self.feature_model(input, None)
      #y = y + EPS
      for i in range(len(features)):
        features[i] = features[i] + EPS

      recurrent_state = list()
      recurrent_state.append(y)
      for feature in features:
        recurrent_state.append(feature)
    else:
      prev_output = recurrent_state[0]
      #y, features = self.feature_model(torch.cat([input, prev_output], dim=1), recurrent_state[1:])
      y, features = self.feature_model(input, recurrent_state[1:])
      #y = y + EPS
      for i in range(len(features)):
        features[i] = features[i] + EPS
      
      recurrent_state = list()
      recurrent_state.append(y)
      for feature in features:
        recurrent_state.append(feature)

    output = self.filter_model(y, features)
    output = output + EPS
    recurrent_state[0] = output

    return output, recurrent_state



    # # -----
    # # Recurrent Filters
    # # -----

    # EPS = 10e-8
    # if recurrent_state is None:
    #   B,C,H,W = input.size()
    #   prev_output = torch.zeros((B,3,H,W), device=input.device, dtype=input.dtype)
    #   y, features = self.feature_model(input, None)
    #   #y = y + EPS
    #   for i in range(len(features)):
    #     features[i] = features[i] + EPS

    # else:
    #   prev_output = recurrent_state[0]
    #   y, features = self.feature_model(input, recurrent_state[1:])
    #   #y = y + EPS
    #   for i in range(len(features)):
    #     features[i] = features[i] + EPS

    # # stack features for filter prediction
    # feature_stack = list()
    # if recurrent_state is None:
    #   for i in range(len(features)):
    #     feature_stack.append(torch.cat([features[i], torch.zeros_like(features[i])], dim=1))
    # else:
    #   for i in range(len(features)):
    #     feature_stack.append(torch.cat([features[i], recurrent_state[i+1]], dim=1))
    
    # output, _ = self.filter_model(y, prev_output, feature_stack)


    # recurrent_state = list()
    # recurrent_state.append(output)
    # for feature in features:
    #   recurrent_state.append(feature)
    # return output, recurrent_state
