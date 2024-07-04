## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun

from model.wnet import *
from model.temporal_wnet import *
from model.unet import *
from dataset import *
from color import *


def get_model(cfg):
  type = cfg.model
  num_channels = get_num_channels(cfg.features)
  if type == 'unet':
      return UNet(cfg.model_config, get_num_channels(cfg.features), 3)
  elif type == 'temporal_unet':
      return UNet_Features_Recurrent(cfg.model_config, get_num_channels(cfg.features), 3)
  elif type == 'wnet':
      return WNet(cfg.model_config, get_num_channels(cfg.features))
  elif type == 'temporal_wnet':
      return TemporalWNet(cfg.model_config, get_num_channels(cfg.features))
  else:
      error('invalid model')

