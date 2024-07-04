import torch
import torch.nn.functional as fun
from torchvision import models

from color import *

class VGG19(torch.nn.Module):
  def __init__(self, input_scaling=False):
    super(VGG19, self).__init__()
    vgg0 = models.vgg19(pretrained=True).features[:3]
    vgg1 = models.vgg19(pretrained=True).features[3:8]
    vgg2 = models.vgg19(pretrained=True).features[8:17]
    vgg3 = models.vgg19(pretrained=True).features[17:26]
    vgg4 = models.vgg19(pretrained=True).features[26:35]

    self.vgg_slices = [vgg0, vgg1, vgg2, vgg3, vgg4]
    # self.weights = [0.7, 0.2, 0.05, 0.03, 0.02]
    self.weights = [0.52879148, 0.13693107, 0.05858061, 0.09629559, 0.17940125]
    self.num_layers = len(self.vgg_slices)

    for vgg_slice in self.vgg_slices:
      vgg_slice.cuda()
      vgg_slice.eval()

      for param in vgg_slice.parameters():
        param.requires_grad_(False)
    
    self.input_scaling = input_scaling
    if input_scaling:
      # todo: update mean and std based on OIDN dataset
      self.shift = [-.030, -0.88, -0.188] 
      self.scale = [0.458, 0.448, 0.450]

  def scale_input(self, input):
    return (input - self.shift) / self.scale

  def normalize_tensor(self, input, reference, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(reference**2, dim=1, keepdim=True))
    input_norm = input / (norm_factor + eps)
    reference_norm = reference / (norm_factor + eps)
    return input_norm, reference_norm

  def forward(self, input, reference, input_scaling=False, per_layer=False):
    if input.shape[1] > 3:
      input = input[:,:3,...]

    if reference.shape[1] > 3:
      reference = reference[:,:3,...]

    if self.input_scaling:
      input = 2 * input - 1
      input = self.scale_input(input)

      reference = 2 * reference - 1
      reference = self.scale_input(reference)

    loss = 0
    if per_layer:
      loss = []
    for i, vgg_slice in enumerate(self.vgg_slices):
      input = vgg_slice(input)
      reference = vgg_slice(reference)
      # input_norm, reference_norm = self.normalize_tensor(input.clone(), reference.clone())
      # loss += torch.abs(input_norm - reference_norm).mean() / self.num_layers
      if per_layer:
        loss.append(torch.abs(input.clone() - reference.clone()).mean().cpu().numpy().item())
      else:
        loss += torch.abs(input.clone() - reference.clone()).mean() * self.weights[i]
    return loss