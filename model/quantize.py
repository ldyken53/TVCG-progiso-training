## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch as torch
import torch.nn as nn
import torch.nn.functional as fun


#class QRound(torch.autograd.Function):

class QRound(nn.Module):

  def forward(self, input):
    return torch.round(input)

# class QRound(torch.autograd.Function):

#   @staticmethod
#   def forward(ctx, input):
#     return torch.round(input)

#   @staticmethod
#   def backward(ctx, grad_output):
#     grad_input = grad_output.clone()
#     return grad_input

class QuantUnsigned(nn.Module):
  def __init__(self, num_bits):
    super(QuantUnsigned, self).__init__()
    #self.qround = QRound.apply
    self.qround = QRound()
    levels = 2 ** num_bits
    self.scale = 1 / (levels - 1)
    self.qmax  =  levels - 1

  def forward(self, x, max):
    step = max * self.scale
    xq = self.qround(x / step)
    xq = torch.clamp(xq, min=0, max=self.qmax)
    xq = xq * step 
    return xq

class QuantSymmetric(nn.Module):
  def __init__(self, num_bits):
    super(QuantSymmetric, self).__init__()
    #self.qround = QRound.apply
    self.qround = QRound()
    levels = 2 ** num_bits
    self.scale = 1 / (levels - 1)
    self.qmin  = -levels / 2
    self.qmax  =  levels / 2 - 1

  def forward(self, x, max):
    step = 2 * max * self.scale
    xq = self.qround(x / step)
    xq = torch.clamp(xq, min=self.qmin, max=self.qmax)
    xq = xq * step 
    return xq

class TrainedQuantUnsigned(nn.Module):
  def __init__(self, size_in, quantize=False, num_bits=8, max_val=1.0):
    super(TrainedQuantUnsigned, self).__init__()
    self.max_log = nn.Parameter(torch.log(torch.abs(torch.tensor(max_val, requires_grad=quantize))), requires_grad=quantize) 
    self.quant = QuantUnsigned(num_bits)
    self.quantize = quantize

  def forward(self, x):
    if self.quantize:
      th_max = torch.exp(self.max_log)
      th_max = th_max.view(1,-1,1,1)
      xq = self.quant(x, max=th_max)
      return xq
    else:
      return x

class QConv2D(nn.Conv2d):
  def get_lim_per_channel(self):
    wt_size = self.weight.size()
    weight_flat = self.weight.view(wt_size[0],wt_size[1]*wt_size[2]*wt_size[3])
    (wt_min,_) = torch.min(weight_flat,dim=1)
    (wt_max,_) = torch.max(weight_flat,dim=1)
    wt_abs = torch.max(torch.abs(wt_min),torch.abs(wt_max))
    wt_abs = wt_abs.view(wt_size[0],1,1,1)
    return wt_abs

  def __init__(self, *args, quantize=False, num_bits=8, **kwargs):
    super(QConv2D, self).__init__(*args, **kwargs)
    self.quant = QuantSymmetric(num_bits)
    self.quantize = quantize
    nn.init.xavier_uniform_(self.weight.data)

  def forward(self, input):
    if self.quantize:
      wt_max = self.get_lim_per_channel()
      quant_wts = self.quant(self.weight, max=wt_max)
      out = self._conv_forward(input, quant_wts, self.bias)
    else:
      out = self._conv_forward(input, self.weight, self.bias)
    return out
