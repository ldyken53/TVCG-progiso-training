## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from util import *
from image import *
from ssim import SSIM, MS_SSIM
from vgg19 import *

def get_loss_function(cfg, loss_type='loss'):
  if loss_type in cfg:
    type = vars(cfg)[loss_type]
  else:
    type = cfg
  if type == 'l1':
    return L1Loss()
  elif type == 'l2':
    return L2Loss()
  elif type == 'mape':
    return MAPELoss()
  elif type == 'smape':
    return SMAPELoss()
  elif type == 'ssim':
    return SSIMLoss()
  elif type == 'msssim':
    return MSSSIMLoss()
  elif type == 'vgg':
    return VGGLoss()
  elif type == 'grad':
    return GradientLoss()
  elif type == 'l1_msssim':
    # [Zhao et al., 2018, "Loss Functions for Image Restoration with Neural Networks"]
    return MixLoss(L1Loss(), MSSSIMLoss(), 0.84)
  elif type == 'l1_grad':
    return MixLoss(L1Loss(), GradientLoss(), 0.5)
  elif type == 'l1_vgg':
    return MixLoss(L1Loss(), VGGLoss(), 0.9)
  else:
    error('invalid loss function')

# L1 loss (seems to be faster than the built-in L1Loss)
class L1Loss(nn.Module):
  def forward(self, input, target):
    # if input.shape[1] > 3:
    #   input = input[:,:3,...]

    # if target.shape[1] > 3:
    #   target = target[:,:3,...]

    return torch.abs(input - target).mean()

# L2 (MSE) loss
class L2Loss(nn.Module):
  def forward(self, input, target):
    return ((input - target) ** 2).mean()

# MAPE (relative L1) loss
class MAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean()

# SMAPE (symmetric MAPE) loss
class SMAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(input) + torch.abs(target) + 1e-2)).mean()

# SSIM loss
class SSIMLoss(nn.Module):
  def __init__(self):
    super(SSIMLoss, self).__init__()
    self.ssim = SSIM(data_range=1.)

  def forward(self, input, target):
    with amp.autocast(enabled=False):
      return 1. - self.ssim(input.float(), target.float())

# MS-SSIM loss
class MSSSIMLoss(nn.Module):
  def __init__(self):
    super(MSSSIMLoss, self).__init__()
    self.msssim = MS_SSIM(data_range=1.)

  def forward(self, input, target):
    with amp.autocast(enabled=False):
      return 1. - self.msssim(input.float(), target.float())

class VGGLoss(nn.Module):
  def __init__(self):
    super(VGGLoss, self).__init__()
    self.vgg = VGG19()

  def forward(self, input, target, per_layer=False):
    with amp.autocast(enabled=False):
      return self.vgg(input.float(), target.float(), per_layer=per_layer)

# Gradient loss
class GradientLoss(nn.Module):
  def forward(self, input, target):
    return torch.abs(tensor_gradient(input) - tensor_gradient(target)).mean()

# Mix loss
class MixLoss(nn.Module):
  def __init__(self, loss1, loss2, alpha):
    super(MixLoss, self).__init__()
    self.loss1 = loss1
    self.loss2 = loss2
    self.alpha = alpha

  def forward(self, input, target):
    return (1. - self.alpha) * self.loss1(input, target) + self.alpha * self.loss2(input, target)


def gradient_penalty(discriminator, real_imgs, fake_imgs, gamma=10):
  batch_size = real_imgs.size(0)
  epsilon = torch.rand(batch_size, 1, 1, 1, 1).to(real_imgs.device)
  epsilon = epsilon.expand_as(real_imgs)

  interpolation = epsilon * real_imgs.data + (1 - epsilon) * fake_imgs.data
  interpolation = torch.autograd.Variable(interpolation, requires_grad=True)


  interpolation_logits = discriminator(interpolation)
  grad_outputs = torch.ones(interpolation_logits.size(), device=real_imgs.device)

  gradients = torch.autograd.grad(outputs=interpolation_logits,
                            inputs=interpolation,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True)[0]

  gradients = gradients.contiguous().view(batch_size, -1)
  gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
  return torch.mean(gamma * ((gradients_norm - 1) ** 2))