## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun

from typing import Optional, List, Tuple

from model.common import *
from model.quantize import *

class EncoderDecoder(nn.Module):
  enoder_decoder: Optional[nn.Module]

  def __init__(self, model_config, size_in, dec_quant, depth=0):
    super(EncoderDecoder, self).__init__()

    enc_config = model_config['encoder_stages']
    dec_config = model_config['decoder_stages']

    enc_size = enc_config[depth][0]
    num_conv = enc_config[depth][1]

    quantize, num_bits = get_quant_settings(model_config)
    enc_quant = TrainedQuantUnsigned(enc_size, quantize, num_bits)

    depth_n = depth + 1

    self.has_encoder_decoder = depth_n != len(enc_config)

    if depth_n == len(enc_config):
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=dec_quant)
      self.encoder_decoder = None
      return
    else:
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=enc_quant)

    enc_size_next = enc_config[depth_n][0]
    dec_size = dec_config[depth][0]
    num_conv = dec_config[depth][1]

    self.encoder_decoder = EncoderDecoder(model_config, enc_size, enc_quant, depth_n)

    dec_size_in = enc_size + enc_size_next
    self.decoder_stage = DecoderStage(model_config,dec_size_in,dec_size,num_conv=num_conv,quantizer=dec_quant)
    return

  def forward(self, x, activation_list:List[torch.Tensor]):
    x = self.encoder_stage(x)

    if self.encoder_decoder is not None:
      y = fun.avg_pool2d(x, kernel_size=2)
      y, _ = self.encoder_decoder(y, activation_list)
      y = fun.interpolate(y, scale_factor=2.0, mode='bilinear',align_corners=False)
      xy = torch.cat([x,y],dim=1)
      y = self.decoder_stage(xy)

      activation_list.append(y)

      return y, activation_list
    else:
      activation_list.append(x)
      return x, activation_list
    
class EncoderDecoderRecurrent(nn.Module):
  enoder_decoder: Optional[nn.Module]

  def __init__(self, model_config, size_in, dec_quant, depth=0, predict_temporal_blend_weights=False):
    super(EncoderDecoderRecurrent, self).__init__()

    enc_config = model_config['encoder_stages']
    dec_config = model_config['decoder_stages']

    enc_size = enc_config[depth][0]
    num_conv = enc_config[depth][1]

    quantize, num_bits = get_quant_settings(model_config)
    enc_quant = TrainedQuantUnsigned(enc_size, quantize, num_bits)

    depth_n = depth + 1

    self.has_encoder_decoder = depth_n != len(enc_config)

    if depth_n == len(enc_config):
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=dec_quant)
      self.encoder_decoder = None
      return
    else:
      self.encoder_stage = EncoderStage(model_config,size_in,enc_size,num_conv=num_conv,quantizer=enc_quant)

    enc_size_next = enc_config[depth_n][0]
    self.dec_size = dec_config[depth][0] if not predict_temporal_blend_weights else dec_config[depth][0] * 2
    num_conv = dec_config[depth][1]

    self.encoder_decoder = EncoderDecoderRecurrent(model_config, enc_size, enc_quant, depth_n)

    # Add the output size of the convolution (dec_size) to accommodate for the recurrent state
    # It is concatenated with the other input data for this layer
    dec_size_in = enc_size + enc_size_next + self.dec_size
    self.decoder_stage = DecoderStage(model_config,dec_size_in,self.dec_size,num_conv=num_conv,quantizer=dec_quant)
    return

  def forward(self, x, activation_list:List[torch.Tensor], recurrent_states:Optional[List[torch.Tensor]]):
    x = self.encoder_stage(x)

    if self.encoder_decoder is not None:
      recurrent_state : Optional[Tensor] = None
      if recurrent_states is not None:
        recurrent_state = recurrent_states.pop()

      y = fun.avg_pool2d(x, kernel_size=2)
      y, _ = self.encoder_decoder(y, activation_list, recurrent_states)
      y = fun.interpolate(y, scale_factor=2.0, mode='bilinear',align_corners=False)

      # Concatenate input, output, and recurrent state
      if recurrent_state is None:
        B,C,H,W = y.size()
        recurrent_state = torch.zeros((B,self.dec_size,H,W), device=y.device, dtype=x.dtype)
      xy = torch.cat([x,y,recurrent_state],dim=1)
      y = self.decoder_stage(xy)

      activation_list.append(y)

      return y, activation_list
    else:
      activation_list.append(x)
      return x, activation_list
   
class UNet(nn.Module):
  activation_list: List[torch.Tensor]


  def __init__(self, model_config, size_in, size_out, apply_final_act=False):
    super(UNet, self).__init__()
    self.apply_final_act = apply_final_act
    out_channels = model_config['decoder_stages'][0][0]

    self.quantize, num_bits_act = get_quant_settings(model_config)
    self.input_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits=8, max_val=1.)
    dec_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits_act)

    # autoencoder
    self.encoder_decoder = EncoderDecoder(model_config, size_in, dec_quant)
    
    # convolution layer to project to RGB
    self.conv = nn.Conv2d(out_channels,size_out,kernel_size=1,bias=True,padding=0)
    self.conv.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.conv.weight)
    self.act = get_act("relu",model_config,self.conv)
      

  def forward(self, x):
    xy, _ = self.encoder_decoder(x, [])
    #xy = torch.cat([x,y],dim=1)
    xy = self.conv(xy)
    if self.apply_final_act:
      xy = self.act(xy)
    return xy

class UNet_Features(nn.Module):
  activation_list: List[torch.Tensor]


  def __init__(self, model_config, size_in, size_out, apply_final_act=False):
    super(UNet_Features, self).__init__()
    self.apply_final_act = apply_final_act
    out_channels = model_config['decoder_stages'][0][0]

    self.quantize, num_bits_act = get_quant_settings(model_config)
    self.input_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits=8, max_val=1.)
    dec_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits_act)

    # autoencoder
    self.encoder_decoder = EncoderDecoder(model_config, size_in, dec_quant)

    # convolution layer to project to RGB
    self.conv = nn.Conv2d(out_channels,size_out,kernel_size=1,bias=True,padding=0)
    self.conv.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.conv.weight)
    self.act = get_act("relu",model_config,self.conv)
      

  def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    x1 = x
    if self.quantize:
      x1 = self.input_quant(x)
    y, activation_list = self.encoder_decoder(x1, [])
    y = self.conv(y)
    y = self.act(y)
    return y, activation_list


class UNet_Features_Recurrent(nn.Module):
  activation_list: List[torch.Tensor]


  def __init__(self, model_config, size_in, size_out, predict_temporal_blend_weights=False):
    super(UNet_Features_Recurrent, self).__init__()
    out_channels = model_config['decoder_stages'][0][0]
    self.quantize, num_bits_act = get_quant_settings(model_config)
    self.input_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits=8, max_val=1.)
    dec_quant = TrainedQuantUnsigned(out_channels, self.quantize, num_bits_act)

    # autoencoder
    self.encoder_decoder = EncoderDecoderRecurrent(model_config, size_in, dec_quant, predict_temporal_blend_weights=predict_temporal_blend_weights)

    # convolution layer to project to RGB
    if predict_temporal_blend_weights:
      self.conv = nn.Conv2d(out_channels*2,size_out,kernel_size=1,bias=True,padding=0)
    else:
      self.conv = nn.Conv2d(out_channels,size_out,kernel_size=1,bias=True,padding=0)
    self.conv.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.conv.weight)
    self.act = get_act("relu",model_config,self.conv)
    
    self.alignment = 32

  def forward(self, x, recurrent_state:Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    x1 = x
    if self.quantize:
      x1 = self.input_quant(x)
    y, activation_list = self.encoder_decoder(x1, [], recurrent_state)
    y = self.conv(y)
    y = self.act(y)
    return y, activation_list
