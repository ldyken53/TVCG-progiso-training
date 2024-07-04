## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun

from typing import Optional, List

from model.common import *
from model.quantize import *

class TemporalFilterOp(nn.Module):
  def __init__(self, model_config, size_in):
    super(TemporalFilterOp, self).__init__()

    quantize, num_bits = get_quant_settings(model_config)
    self.flow_predict = QConv2D(size_in//2, 2, kernel_size=1, quantize=quantize, num_bits=num_bits)
    self.blend_predict = QConv2D(size_in//2, 1, kernel_size=1, quantize=quantize, num_bits=num_bits)
    self.act1 = nn.Tanh()
    self.act2 = nn.ReLU()
  
  def forward(self, activation, rgb, prev_rgb):
    _, C, _, _ = activation.size()
    predicted_flow = self.flow_predict(activation[:, :C//2, ...])
    predicted_flow = self.act1(predicted_flow)
    predicted_flow = predicted_flow.permute(0,2,3,1)
    predicted_blend = self.blend_predict(activation[:, C//2:, ...])
    predicted_blend = self.act2(predicted_blend)
    prev_rgb = fun.grid_sample(prev_rgb, predicted_flow)


    return (1.0 - predicted_blend) * rgb + (predicted_blend) * prev_rgb, prev_rgb

class FilterOp(nn.Module):
  def __init__(self, model_config, size_in, num_channels, filter_length, skip=False):
    super(FilterOp, self).__init__()

    if skip:
        size_out = filter_length + 1 
    else:
        size_out = filter_length

    quantize, num_bits = get_quant_settings(model_config)

    self.filter_predict = QConv2D(size_in, size_out, kernel_size=1, quantize=quantize, num_bits=num_bits)
    self.filter_predict.bias.data.fill_(0)
    self.relu = nn.ReLU()
    self.apply_filter = Filter(num_channels, skip=skip)

  def normalize_filter(self, predicted_filter):
    predicted_filter = torch.square(predicted_filter)
    filter_sum = torch.sum(predicted_filter, dim=1, keepdim=True)
    predicted_filter = predicted_filter/filter_sum
    return predicted_filter

  def forward(self, activation, rgb, skipped_rgb:Optional[torch.Tensor]=None):     
    predicted_filter = self.filter_predict(activation)
    predicted_filter = self.normalize_filter(predicted_filter)
    rgb = self.apply_filter(predicted_filter, rgb, skipped_rgb)
    return rgb

class FilterPath(nn.Module):
  def __init__(self, model_config, num_channels, filter_length=9, num_stacks=1, depth=0):
    super(FilterPath, self).__init__()

    enc_config = model_config['encoder_stages']
    enc_size = enc_config[depth][0] * num_stacks

    depth_n = depth + 1

    if depth_n < len(enc_config):
      dec_config = model_config['decoder_stages']
      dec_size = dec_config[depth][0] * num_stacks
   
      self.encoder_filter_stage = FilterOp(model_config, dec_size, num_channels, filter_length)

      self.filter_path = FilterPath(model_config, num_channels, filter_length, num_stacks, depth_n)

      self.decoder_filter_stage = FilterOp(model_config, dec_size, num_channels, filter_length, skip=True)
    else:
      self.encoder_filter_stage = FilterOp(model_config, enc_size, num_channels, filter_length)
      self.filter_path = None
    return

  def forward(self, rgb, activations:List[torch.Tensor]):
    features = activations.pop()
    
    enc_rgb = self.encoder_filter_stage(features, rgb)
    
    if self.filter_path is not None: 
        enc_rgb1 = fun.avg_pool2d(enc_rgb, kernel_size=2)
        enc_rgb1 = self.filter_path(enc_rgb1, activations)
    else:
        return enc_rgb

    dec_rgb = fun.interpolate(enc_rgb1, scale_factor=2.0, mode='bilinear', align_corners=False)
    dec_rgb = self.decoder_filter_stage(features, dec_rgb, enc_rgb)
    return dec_rgb


class TemporalFilterPath(nn.Module):
  def __init__(self, model_config, num_channels, filter_length=9, num_stacks=1, depth=0):
    super(TemporalFilterPath, self).__init__()

    enc_config = model_config['encoder_stages']
    enc_size = enc_config[depth][0] * num_stacks
    quantize, num_bits = get_quant_settings(model_config)

    self.depth_n = depth + 1

    if self.depth_n < len(enc_config):
      dec_config = model_config['decoder_stages']
      dec_size = dec_config[depth][0] * num_stacks
   
      self.encoder_filter_stage = FilterOp(model_config, dec_size//2, num_channels, filter_length)
      self.encoder_temp_filter_stage = TemporalFilterOp(model_config, dec_size//2)

      self.filter_path = TemporalFilterPath(model_config, num_channels, filter_length, num_stacks, self.depth_n)

      self.decoder_filter_stage = FilterOp(model_config, dec_size//2, num_channels, filter_length, skip=True)
      self.decoder_temp_filter_stage = TemporalFilterOp(model_config, dec_size//2)
    else:
      self.encoder_filter_stage = FilterOp(model_config, enc_size//2, num_channels, filter_length)
      self.encoder_temp_filter_stage = TemporalFilterOp(model_config, enc_size//2)
      self.filter_path = None

    # if self.depth_n == 1:
    #   self.temporal_blend_weight_predict = QConv2D(dec_size//2, 1, kernel_size=1, quantize=quantize, num_bits=num_bits)
    #   self.temporal_blend_weight_predict.bias.data.fill_(0)
    #   self.relu = nn.ReLU()

    return

  def forward(self, rgb, prev_rgb, activations:List[torch.Tensor]):
    activation = activations.pop()
    _,C,_,_ = activation.size()

    features = activation[:, :C//2, ...]
    temporal_features = activation[:, C//2:, ...]

   
    # Filter
    enc_rgb = rgb
    enc_rgb, prev_enc_rgb = self.encoder_temp_filter_stage(temporal_features, enc_rgb, prev_rgb)
    enc_rgb = self.encoder_filter_stage(features, enc_rgb)
    
    if self.filter_path is not None: 
        enc_rgb1 = fun.avg_pool2d(enc_rgb, kernel_size=2)
        prev_enc_rgb1 = fun.avg_pool2d(prev_enc_rgb, kernel_size=2)
        enc_rgb1, prev_rgb1 = self.filter_path(enc_rgb1, prev_enc_rgb1, activations)
    else:
        return enc_rgb, prev_enc_rgb

    dec_rgb = fun.interpolate(enc_rgb1, scale_factor=2.0, mode='bilinear', align_corners=False)
    prev_dec_rgb = fun.interpolate(prev_enc_rgb1, scale_factor=2.0, mode='bilinear', align_corners=False)

    dec_rgb = self.decoder_filter_stage(features, dec_rgb, enc_rgb)
    dec_rgb, prev_dec_rgb = self.decoder_temp_filter_stage(temporal_features, dec_rgb, prev_dec_rgb)


    # if self.depth_n == 1:
    #   temporal_blend_weights = self.temporal_blend_weight_predict(temporal_features2)
    #   temporal_blend_weights = self.relu(temporal_blend_weights)

    #   dec_rgb = (1.0 - temporal_blend_weights) * dec_rgb + temporal_blend_weights * prev_rgb

    return dec_rgb, prev_dec_rgb
