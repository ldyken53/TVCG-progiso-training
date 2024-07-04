## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import torch.optim as optim

from training.ranger import Ranger
from training.radam import RAdam


def get_optimizer(train_config, parameters):
  lr = float(train_config['lr'])

  if 'optimizer' in train_config:
    optimizer = train_config['optimizer']
    weight_decay = train_config['weight_decay'] if 'weight_decay' in train_config else 0

    if optimizer == 'ranger':
      optimizer = Ranger(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'radam':
      optimizer = RAdam(parameters, lr=lr, betas=(.95,0.999), eps=1e-5, weight_decay=weight_decay)
    else:
      optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  else:
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

  return optimizer
