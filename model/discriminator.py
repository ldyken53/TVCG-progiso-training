import torch as torch
import torch.nn as nn
import torch.nn.functional as fun

from typing import Optional, List, Tuple

from model.common import *
from model.quantize import *


class ConvGroup3D(nn.Module):
    def __init__(self, model_config, size_in, size_out, kernel_size=3):
        super(ConvGroup3D, self).__init__()
        self.conv = nn.Conv3d(size_in, size_out, kernel_size=kernel_size, padding=1)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, model_config, size_in, size_out, num_conv=2):
        super(DiscriminatorBlock, self).__init__()
        
        self.conv_groups = nn.ModuleList()

        conv = ConvGroup3D(model_config, size_in, size_out)
        self.conv_groups.append(conv)
        if num_conv > 1:
            for _ in range(num_conv-1):
                self.conv_groups.append(ConvGroup3D(model_config, size_out, size_out))
        

    def forward(self, x):
        x1 = x
        for group in self.conv_groups:
            x1 = group(x1)

        xy = torch.cat([x1,x], dim=1)
        return xy

class Discriminator(nn.Module):

    def __init__(self, model_config, size_in):
        super(Discriminator, self).__init__()

        self.block1 = DiscriminatorBlock(model_config, size_in, 16)
        size_in = size_in//2+8
        self.block2 = DiscriminatorBlock(model_config, size_in, 32)
        size_in = size_in//2+16
        self.block3 = DiscriminatorBlock(model_config, size_in, 64)
        size_in = size_in//2+32
        self.block4 = DiscriminatorBlock(model_config, size_in, 80)
        size_in = size_in//2+40
        self.block5 = DiscriminatorBlock(model_config, size_in, 92)
        size_in = size_in//2 + 46
        self.flatten = nn.Conv3d(size_in, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        y = input.permute(0,2,1,3,4)
        y = self.block1(y)
        y = fun.avg_pool3d(y.permute(0,2,1,3,4), kernel_size=2).permute(0,2,1,3,4)
        y = self.block2(y)
        y = fun.avg_pool3d(y.permute(0,2,1,3,4), kernel_size=2).permute(0,2,1,3,4)
        y = self.block3(y)
        y = fun.avg_pool3d(y.permute(0,2,1,3,4), kernel_size=2).permute(0,2,1,3,4)
        y = self.block4(y)
        y = fun.avg_pool3d(y.permute(0,2,1,3,4), kernel_size=2).permute(0,2,1,3,4)
        y = self.block5(y)
        y = fun.avg_pool3d(y.permute(0,2,1,3,4), kernel_size=2).permute(0,2,1,3,4)
        y = self.flatten(y)

        return y