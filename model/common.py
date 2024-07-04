## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.transforms.functional as visfun
import math

from typing import Optional, Tuple

from model.quantize import *
from image import *

def get_act(act_type, model_config, conv):
  init_type = model_config['init']
  slope = 0.0

  if act_type == "leaky_relu":
    slope = model_config['leaky_slope']
    act = nn.LeakyReLU(negative_slope=slope)
  elif act_type == "prelu":
    slope = model_config['leaky_slope']
    act = nn.PReLU(init=slope)
  elif act_type == "relu":
    act = nn.ReLU()
  elif act_type == "elu":
    act = nn.ELU()
  else:
    act = nn.ReLU()

  init_fn = None
  if init_type == "he_unif":
    init_fn = torch.nn.init.kaiming_uniform_
  elif init_type == "he_norm":
    init_fn = torch.nn.init.kaiming_normal_

  if init_fn:
    if act_type == "leaky_relu" or act_type == "prelu":
      init_fn(conv.weight, a=slope, nonlinearity='leaky_relu')
    else:
      init_fn(conv.weight, nonlinearity='relu')

  return act

def get_quant_settings(model_config):
  if 'quantize' in model_config:
    quantize = model_config['quantize']
  else:
    quantize = False
  if 'num_bits' in model_config:
    num_bits = model_config['num_bits']
  else:
    num_bits = 32
  
  return quantize, num_bits
  
class ConvGroup(nn.Module):
  def __init__(self, model_config, size_in, size_out, kernel_size=3, quantizer=None):
    super(ConvGroup, self).__init__()

    self.use_bn = model_config['use_bn']
    quantize, num_bits = get_quant_settings(model_config)
   
    self.conv = QConv2D(size_in, size_out, kernel_size=kernel_size, bias=not self.use_bn, padding=kernel_size//2, quantize=quantize, num_bits=num_bits)

    #self.bn = None
    self.bn = nn.BatchNorm2d(size_out, affine=True, track_running_stats=True)
    if self.use_bn:
      self.bn = nn.BatchNorm2d(size_out, affine=True, track_running_stats=True)
    else:
      self.conv.bias.data.fill_(0)

    self.act = get_act(model_config['activation'],model_config,self.conv)

    if quantizer:
      self.quantizer = quantizer
    else:
      self.quantizer = TrainedQuantUnsigned(size_out, quantize, num_bits)

  def forward(self, x):
    x = self.conv(x)
    if self.use_bn:
      x = self.bn(x)
    x = self.act(x)
    x = self.quantizer(x)
    return x

class DecoderStage(nn.Module):
  def __init__(self, model_config, size_in, size_out, quantizer=None, num_conv=1):
    super(DecoderStage, self).__init__()
    self.conv_groups = nn.ModuleList()

    if num_conv == 1:
      group = ConvGroup(model_config,size_in,size_out,kernel_size=3, quantizer=quantizer)
    else:
      group = ConvGroup(model_config,size_in,size_out,kernel_size=3, quantizer=None)
    self.conv_groups.append(group)

    for conv_id in range(num_conv-1):
      if conv_id == num_conv-2:
        group = ConvGroup(model_config,size_out,size_out,kernel_size=3,quantizer=quantizer)
      else:
        group = ConvGroup(model_config,size_out,size_out,kernel_size=3,quantizer=None)
      self.conv_groups.append(group)

  def forward(self, x):
    for group in self.conv_groups:
      x = group(x)
    return x

class EncoderStage(nn.Module):
  def __init__(self, model_config, size_in, size_out, num_conv=1, quantizer=None):
    super(EncoderStage, self).__init__()
    self.conv_groups = nn.ModuleList()

    if num_conv == 1:
      group = ConvGroup(model_config,size_in,size_out,kernel_size=3, quantizer=quantizer)
    else:
      group = ConvGroup(model_config,size_in,size_out,kernel_size=3, quantizer=None)
    self.conv_groups.append(group)

    for conv_id in range(num_conv-1):
      if conv_id == num_conv-2:
        group = ConvGroup(model_config, size_out, size_out, kernel_size=3, quantizer=quantizer)
      else:
        group = ConvGroup(model_config, size_out, size_out, kernel_size=3, quantizer=None)
      self.conv_groups.append(group)

  def forward(self, x):
    for group in self.conv_groups:
      x = group(x)
    return x

class Filter(nn.Module):
  def __init__(self, num_channels, num_filters=1,skip=False):
    super(Filter, self).__init__()
    self.num_channels = num_channels
    self.num_filters  = num_filters
    self.skip         = skip

  def forward(self, k, x, skip:Optional[torch.Tensor]=None):
    input_dims = x.size()
    patch_dim = (input_dims[0], self.num_channels, 9) + input_dims[2:4]
    patches_list = []

    for idx in range(self.num_filters):
      x_slice = x[:,idx*self.num_channels : idx*self.num_channels+self.num_channels,...]
      patches = fun.unfold(x_slice,kernel_size=(3,3),padding=1)
      patches = torch.reshape(patches,patch_dim)
      patches_list.append(patches)

    if skip is not None and self.skip:
      skip = skip.unsqueeze(dim=2)
      patches_list.append(skip)

    patches_cat = torch.cat(patches_list,dim=2)
    kernel = k.unsqueeze(dim=1)

    y = patches_cat * kernel
    y = torch.sum(y, dim=2)
    return y

# Non-Local Means Fast Reprojection 
# Reprojects frame I2 to I1
def hierarchical_reproject(I2, I1, depth = 4, kernel_size=5):

  # Downsample the input
  I1_scaled = [I1]
  I2_scaled = [I2]
  for _ in range(depth):
    I1_scaled.append(fun.interpolate(I1_scaled[-1], scale_factor=1/2, mode='bilinear'))
    I2_scaled.append(fun.interpolate(I2_scaled[-1], scale_factor=1/2, mode='bilinear'))

  # Upsample and apply reprojection in each level
  for d in range(depth-1,0,-1):
    motion = motion_kernel(I1_scaled[d], I2_scaled[d], kernel_size)
    motion = motion.permute((0,3,1,2))
    # TODO: If we upsample the motion map, we might introduce unwanted blurriness
    motion = fun.interpolate(motion, scale_factor=2, mode='bilinear', align_corners=False)
    motion = motion.permute((0,2,3,1))
    I2_scaled[d-1] = reproject(I2_scaled[d-1], motion)
  motion = motion_kernel(I1_scaled[0], I2_scaled[0], kernel_size)
  I2_scaled[0] = fun.grid_sample(I2_scaled[0], motion, align_corners=False)
  I2_scaled[0] = blend_kernel(I1, I2_scaled[0], a=0.8)

  # B,_,_,_ = I1.size()
  # for b in range(B):
  #   save_image(f'I1_{b}.png', tensor_to_image(I1[b]))
  #   save_image(f'I2_{b}.png', tensor_to_image(I2[b]))
  #   save_image(f'I_diff_{b}.png', tensor_to_image(torch.abs(I2_scaled[0][b] - I1[0])))
  #   save_image(f'I_diff_orig_{b}.png', tensor_to_image(torch.abs(I2[0] - I1[0])))

  #   for s in range(len(I1_scaled)):
  #     save_image(f'I1_re_{b}_{s}.png', tensor_to_image(I2_scaled[s][b]))

  return I2_scaled[0]

def reproject(I2, I1, kernel_size : int = 20, return_motion : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
  motion = motion_kernel(I1, I2, kernel_size)
  I = fun.grid_sample(I2, motion, align_corners=False)
  #I = blend_kernel(I1, I, a=0.95, k=2.5)

  if return_motion:
    return I, motion
  else:
    return I, None

def motion_kernel(I1, I2, kernel_size : int):
  _, _, H, W = I1.size()
  x, S = distance_kernel(I1, I2, kernel_size, distance_function='L1')
  x = blur_kernel(x)
  x = merge_kernel(x, S, H, W)

  return x

# Calculate the distance between images at various levels of shift
# distance_function can take any value of ['L1', 'L2', 'logL1', 'logL1c']
def distance_kernel(I1, I2, kernel_size : int, distance_function : str = 'L1'):
  # Get device
  device = I1.device

  # Convert input images to grayscale
  I1 = visfun.rgb_to_grayscale(I1[:, 0:3, ...]).repeat(1, int(kernel_size**2), 1, 1)
  I2 = visfun.rgb_to_grayscale(I2[:, 0:3, ...]).repeat(1, int(kernel_size**2), 1, 1)

  # Create a search window grid S
  S = torch.arange(0, kernel_size, device=device).float()
  S_x, S_y = torch.meshgrid(S,S)
  S = torch.stack((S_y, S_x), dim=-1)
  S -= kernel_size // 2
  S = S.reshape(int(kernel_size**2), 2).flip(0)

  S_kernel = fun.one_hot(torch.arange(0, int(kernel_size**2), device=device)).unsqueeze(0).unsqueeze(0).reshape(int(kernel_size**2), 1, kernel_size, kernel_size).float()#.repeat(B,1,1,1)
  I2_shifted = fun.conv2d(input=I2, weight=S_kernel, stride=1, padding=kernel_size//2, groups=I2.size(1))

  if distance_function == 'L2':
    output = (I1 - I2_shifted) ** 2
  elif distance_function == 'L1':
    output = torch.abs(I1 - I2_shifted)
  elif distance_function == 'logL1':
    output = torch.log(2 + torch.abs(I1 - I2_shifted))
  # elif distance_function == 'logL1c':
  #   output = torch.log(2 + torch.abs(I1 - I2_shifted)) * (2 * torch.exp(-0.02*len_Sk**2))
  else:
    output = (I1 - I2_shifted)

  return output, S


def blur_kernel(I):
  blur_filter = get_gaussian_kernel(I.device, kernel_size=5, channels=I.size(1))
  out = fun.conv2d(I, blur_filter, padding=5//2, groups=I.size(1))
  #out = blur_filter(I)
  return out
  #return visfun.gaussian_blur(I, kernel_size=5)

def get_gaussian_kernel(device:torch.device, kernel_size:int=3, sigma:float=2, channels:int=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

    #gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
    #                            kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    #gaussian_filter.weight.data = gaussian_kernel
    #gaussian_filter.weight.requires_grad = False
    
    #return gaussian_filter

def merge_kernel(Is, S, input_height : int, input_width : int):

  # Get device
  device = Is.device

  # Create a map of relative offsets, each in [-1, 1]
  argmin_dist = torch.argmin(Is, dim=1)
  merged = S[argmin_dist]
  merged[..., 0] /= input_height/2
  merged[..., 1] /= input_width/2

  # Create base meshgrid
  grid_x, grid_y = torch.meshgrid(torch.linspace(-1,1,input_height, device=device), torch.linspace(-1,1,input_width, device=device))
  grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).expand(-1, merged.size()[1], -1, -1)

  # Combine the mesh with offsets
  grid = grid - merged

  # Write debug image
  # merged = merged.permute((0,3,1,2))
  # B,_,H,W = merged.size()

  # for b in range(B):
  #   save_image(f'I_motion{b}_{input_width}.exr', tensor_to_image(torch.cat([0.5*(merged[b]+1.0), 0.5*torch.ones(1,H,W).to(merged.device)], dim=0)))

  
  return grid

def blend_kernel(I1, I2_reprojected, a=0.5, k=1.5, n=0.1):

  a = get_error_norm_blend_factor(I1, I2_reprojected, a=a, k=k, n=n)

  # B,_,H,W = I1.size()
  # for b in range(B):
  #   save_image(f'I_error{b}.png', tensor_to_image(a[b, 0:3, ...]))

  return (1.0-a)*I1 + a*I2_reprojected

def get_error_norm_blend_factor(original, reprojected, a=0.5, k=1.5, n=0.1):
  # Normalize images for error distance
  B, C, H, W = original.size()
  I1_norm = original.clone().view(B, C, -1)
  I1_norm -= I1_norm.min(2, keepdim=True)[0]
  I1_norm /= I1_norm.max(2, keepdim=True)[0]
  I1_norm = I1_norm.view(B, C, H, W)
  I2_reprojected_norm = reprojected.clone().view(B, C, -1)
  I2_reprojected_norm -= I2_reprojected_norm.min(2, keepdim=True)[0]
  I2_reprojected_norm /= I2_reprojected_norm.max(2, keepdim=True)[0]
  I2_reprojected_norm = I2_reprojected_norm.view(B, C, H, W)

  # Compute Error
  error = (I1_norm - I2_reprojected_norm) ** 2
  m = k*error - n

  # Compute Blend Factor
  a = torch.clamp(a*(1-m), 0, 1)

  return a

# Exponential Moving Average
def ema(input:torch.Tensor, ema_prev:torch.Tensor, smoothing:float=0.5):
  return (smoothing * input) + ((1 - smoothing) * ema_prev)
