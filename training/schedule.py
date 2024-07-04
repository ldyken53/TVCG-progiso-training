## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import math as math
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim

def one_cycle(optimizer, end_epoch, lr, max_lr, lr_warmup):
  return optim.lr_scheduler.OneCycleLR(optimizer,
      max_lr=max_lr,
      total_steps=end_epoch,
      pct_start=lr_warmup,
      anneal_strategy='cos',
      div_factor=(25. if lr is None else max_lr / lr),
      final_div_factor=1e4)

def cos_annealing(optimizer, end_epoch, final_lr):
  return optim.lr_scheduler.CosineAnnealingLR(optimizer,
      T_max=end_epoch,
      eta_min=final_lr)

def flat_cos_annealing(optimizer, end_epoch, final_lr, lr):
  constant = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
  cos_annealing = optim.lr_scheduler.CosineAnnealingLR(optimizer,
      T_max=end_epoch,
      eta_min=final_lr)

  return optim.lr_scheduler.SequentialLR(optimizer,
      schedulers=[constant, cos_annealing],
      milestones=[int(end_epoch*0.70)])

# step can be per epoch or a batch. The user controls the update by calling scheduler.step()
def flat():
  def lr_schedule(step):
    return 1.0
  return  lr_schedule

def step(lr, final_lr, steps):
  final_ratio = final_lr / lr
  def lr_schedule(step):
    if step > steps:
      return final_ratio
    else:
      return 1.0
  return  lr_schedule

def linear(lr, final_lr, start_epoch, end_epoch):
  steps = end_epoch - start_epoch
  step = 1
  scale = 1
  def lr_schedule(epoch):
    if epoch >= start_epoch and epoch < end_epoch:
      nonlocal step, scale
      scale = ( 1 - step/steps) + (final_lr/lr * step/steps)
      step = step + 1
      return scale 
    else:
      return scale
  return lr_schedule

def clamped_exp(lr, final_lr, steps):
  final_ratio = final_lr / lr
  falloff = math.exp(math.log(final_ratio) / steps)

  def lr_schedule(step):
    return max(final_ratio, falloff ** step)
  return  lr_schedule

def ramp(steps):
  step_size = 1/steps
  def lr_schedule(step):
    return min(1.0, step * step_size)
  return  lr_schedule

def get_scheduler(cfg, optimizer):
  train_config = cfg.train_config
  lr = float(train_config['lr'])

  if 'schedule' in train_config:
    schedule = train_config['schedule']

    if schedule == 'clamped_exp':
      final_lr    = train_config['final_lr']
      steps       = train_config['steps']
      lr_schedule = clamped_exp(lr, final_lr, steps)

    elif schedule == 'ramp':
      steps       = train_config['steps']
      lr_schedule = ramp(steps)

    elif schedule == 'step':
      final_lr    = train_config['final_lr']
      steps       = train_config['steps']
      lr_schedule = step(lr, final_lr, steps)

    elif schedule == 'linear':
      final_lr    = train_config['final_lr']
      start_epoch = train_config['start_epoch']
      end_epoch   = train_config['end_epoch']
      lr_schedule = linear(lr, final_lr, start_epoch, end_epoch)

    elif schedule == 'one_cycle':
      end_epoch   = train_config['end_epoch']
      lr          = train_config['lr']
      max_lr      = train_config['max_lr']
      lr_warmup   = train_config['lr_warmup']
      return one_cycle(optimizer, end_epoch, lr, max_lr, lr_warmup)

    elif schedule == 'cos_annealing':
      end_epoch   = train_config['end_epoch']
      final_lr    = train_config['final_lr']
      return cos_annealing(optimizer, end_epoch, final_lr)

    elif schedule == 'flat_cos_annealing':
      end_epoch   = train_config['end_epoch']
      final_lr    = train_config['final_lr']
      lr          = train_config['lr']
      return flat_cos_annealing(optimizer, end_epoch, final_lr, lr)

    else:
      lr_schedule = flat()
  else:
    lr_schedule = flat()

  return LambdaLR(optimizer, lr_lambda=[lr_schedule])
