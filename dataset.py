## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
from collections import defaultdict
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import *
from util import *
from image import *
from color import *
import tza

# Returns a dataset directory path
def get_data_dir(cfg, name):
  return os.path.join(cfg.data_dir, name)

# Returns the ordered list of channel names for the specified features
def get_channels(features):
  if ('hdr' in features) or ('ldr' in features):
    channels = ['r', 'g', 'b']
  if 'alb' in features:
    channels += ['alb.r', 'alb.g', 'alb.b']
  if 'nrm' in features:
    channels += ['nrm.x', 'nrm.y', 'nrm.z']
  return channels

# Returns the number of channels for the specified features
def get_num_channels(features):
  return len(get_channels(features))

# Returns the indices of the specified channels in the dataset
def get_channel_indices(channels, data_channels):
  return [data_channels.index(ch) for ch in channels]

# Shuffles channels according to the specified order
def shuffle_channels(channels, first_channel, order):
  first = channels.index(first_channel)
  new_channels = [channels[first+i] for i in order]
  for i in range(len(new_channels)):
    channels[first+i] = new_channels[i]

# Returns the target features given the input features
def get_target_features(features):
  return list(set(features).intersection({'hdr', 'ldr'}))

# Checks whether the image with specified features exists
def image_exists(name, features):
  return all([os.path.isfile(name + '.' + f + '.exr') for f in features])

# Returns the feature an image represents given its filename
def get_image_feature(filename):
  filename_split = filename.rsplit('.', 2)
  if len(filename_split) < 2:
    return 'srgb' # no extension, assume sRGB
  else:
    ext = filename_split[-1].lower()
    if ext in {'exr', 'pfm', 'hdr'}:
      if len(filename_split) == 3:
        return filename_split[-2]
      else:
        return 'hdr' # assume HDR
    else:
      return 'srgb' # assume sRGB

# Loads target image features in EXR format with given filename prefix
def load_target_image(name, features):
  if 'hdr' in features:
    color_filename = name + '.hdr.exr'
  else:
    color_filename = name + '.ldr.exr'
  color = load_image(color_filename, num_channels=3)
  if 'hdr' in features:
    color = np.maximum(color, 0.)
  else:
    color = np.clip(color, 0., 1.)
  return color

# Loads input image features in EXR format with given filename prefix
def load_input_image(name, features):
  # Color
  color = load_target_image(name, features)
  inputs = [color]

  # Albedo
  if 'alb' in features:
    albedo_filename = name + '.alb.exr'
    albedo = load_image(albedo_filename, num_channels=3)
    albedo = np.clip(albedo, 0., 1.)
    inputs.append(albedo)

  # Normal
  if 'nrm' in features:
    normal_filename = name + '.nrm.exr'
    normal = load_image(normal_filename, num_channels=3)

    # Normalize
    length_sqr = np.add.reduce(np.square(normal), axis=-1, keepdims=True)
    with np.errstate(divide='ignore'):
      rcp_length = np.reciprocal(np.sqrt(length_sqr))
    rcp_length = np.nan_to_num(rcp_length, nan=0., posinf=0., neginf=0.)
    normal *= rcp_length

    # Transform to [0..1] range
    normal = normal * 0.5 + 0.5

    inputs.append(normal)

  return np.concatenate(inputs, axis=2)

# Tries to load metadata for an image with given filename/prefix, returns None if it fails
def load_image_metadata(name):
  dirname, basename = os.path.split(name)
  basename = basename.split('.')[0] # remove all extensions
  while basename:
    metadata_filename = os.path.join(dirname, basename) + '.json'
    if os.path.isfile(metadata_filename):
      return load_json(metadata_filename)
    if '_' in basename:
      basename = basename.rsplit('_', 1)[0]
    else:
      break
  return None

# Saves image metadata to a file with given prefix
def save_image_metadata(name, metadata):
  save_json(name + '.json', metadata)

# Returns groups of image samples (input and target images at different SPPs) as a list of (group, list of input names, target name)
def get_image_sample_groups(dir, features):
  image_filenames = glob(os.path.join(dir, '**', '*.*.exr'), recursive=True)
  target_features = get_target_features(features)

  # Make image groups
  image_groups = defaultdict(set)
  for filename in image_filenames:
    image_name = os.path.relpath(filename, dir)  # remove dir path
    image_name, _, _ = image_name.rsplit('.', 2) # remove extensions
    group = image_name
    if '_' in image_name:
      prefix, suffix = image_name.rsplit('_', 1)
      suffix = suffix.lower()
      if (suffix.isdecimal() or
          (suffix.endswith('spp') and suffix[:-3].isdecimal()) or
          suffix == 'ref' or suffix == 'reference' or
          suffix == 'gt' or suffix == 'target'):
        group = prefix
    image_groups[group].add(image_name)

  # Make sorted image sample (inputs + target) groups
  image_sample_groups = []
  for group in sorted(image_groups):
    # Get the list of inputs and the target
    image_names = sorted(image_groups[group])
    if len(image_names) > 1:
      input_names, target_name = image_names[:-1], image_names[-1]
    else:
      input_names, target_name = image_names, None

    # Check whether all required features exist
    if all([image_exists(os.path.join(dir, name), features) for name in input_names]) and \
       (not target_name or image_exists(os.path.join(dir, target_name), target_features)):
      # Add sample
      image_sample_groups.append((group, input_names, target_name))

  return image_sample_groups

# Transforms a feature image to another feature type
def transform_feature(image, input_feature, output_feature, exposure=1.):
  if input_feature == 'hdr' and output_feature in {'ldr', 'srgb'}:
    image = tonemap(image * exposure)
  if output_feature == 'srgb':
    if input_feature in {'hdr', 'ldr', 'alb'}:
      image = srgb_forward(image)
    elif input_feature == 'nrm':
      # Transform [-1, 1] -> [0, 1]
      image = image * 0.5 + 0.5
  return image

# Returns a data loader and its sampler for the specified dataset
def get_data_loader(rank, cfg, dataset, shuffle=False):
  if cfg.num_devices > 1:
    sampler = DistributedSampler(dataset,
                                 num_replicas=cfg.num_devices,
                                 rank=rank,
                                 shuffle=shuffle)
  else:
    sampler = None

  loader = DataLoader(dataset,
                      batch_size=(cfg.batch_size // cfg.num_devices),
                      sampler=sampler,
                      shuffle=(shuffle if sampler is None else False),
                      num_workers=cfg.loaders,
                      pin_memory=(cfg.device != 'cpu'))

  return loader, sampler

## -----------------------------------------------------------------------------
## Preprocessed dataset
## -----------------------------------------------------------------------------

# Returns a preprocessed dataset directory path
def get_preproc_data_dir(cfg, name):
  data_dir = os.path.join(cfg.preproc_dir, name) + '.'
  if 'hdr' in cfg.features:
    data_dir += 'hdr'
  elif 'ldr' in cfg.features:
    data_dir += 'ldr'
  data_dir += '.' + cfg.transfer
  return data_dir

class PreprocessedDataset(Dataset):
  def __init__(self, cfg, name):
    super(PreprocessedDataset, self).__init__()
    cfg.temp_size = 12

    self.cfg = cfg

    # Check whether the preprocessed images have all required features
    data_dir = get_preproc_data_dir(cfg, name)
    if not os.path.isdir(data_dir):
      self.num_images = 0
      return
    data_cfg = load_config(data_dir)
    if not all(f in data_cfg.features for f in cfg.features):
      error('the preprocessed images have an incompatible set of features')
    if data_cfg.transfer != cfg.transfer:
      error('the preprocessed images have a mismatching transfer function')

    self.tile_size = cfg.tile_size
    self.features = cfg.features
    self.data_channels = get_channels(data_cfg.features)
    self.channels = get_channels(cfg.features)
    self.channel_order = get_channel_indices(self.channels, self.data_channels)

    # Get the image samples
    samples_filename = os.path.join(data_dir, 'samples.json')
    self.samples = load_json(samples_filename)
    self.num_images = len(self.samples)

    if self.num_images == 0:
      return

    # Create the memory mapping based image reader
    tza_filename = os.path.join(data_dir, 'images.tza')
    self.images = tza.Reader(tza_filename)
    
## -----------------------------------------------------------------------------
## Training dataset
## -----------------------------------------------------------------------------

class TrainingDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(TrainingDataset, self).__init__(cfg, name)

    self.max_padding = 32

  def __len__(self):
    return self.num_images

  def __getitem__(self, index):
    # Get the input and target images
    input_name, target_name = self.samples[index]
    input_image,  _ = self.images[input_name]
    target_image, _ = self.images[target_name]

    # Get the size of the image
    height = input_image.shape[0]
    width  = input_image.shape[1]
    if height < self.tile_size or width < self.tile_size:
      error('image is smaller than the tile size')

    # Generate a random crop
    sy = sx = self.tile_size
    if rand() < 0.1:
      # Randomly zero pad later to avoid artifacts for images that require padding
      sy -= randint(self.max_padding)
      sx -= randint(self.max_padding)
    oy = randint(height - sy + 1)
    ox = randint(width  - sx + 1)

    # Randomly permute some channels to improve training quality
    channels = self.channels[:] # copy

    # # Randomly permute the color channels
    color_order = randperm(3)
    shuffle_channels(channels, 'r', color_order)
    if 'alb' in self.features:
      shuffle_channels(channels, 'alb.r', color_order)

    # # Randomly permute the normal channels
    # if 'nrm' in self.features:
    #   normal_order = randperm(3)
    #   shuffle_channels(channels, 'nrm.x', normal_order)

    # Compute the indices of the required input channels
    channel_order = get_channel_indices(channels, self.data_channels)

    # Crop the input and target images
    input_image  = input_image [oy:oy+sy, ox:ox+sx, channel_order]
    target_image = target_image[oy:oy+sy, ox:ox+sx, 0:3]

    # Randomly transform the tiles to improve training quality
    if rand() < 0.5:
      # Flip vertically
      input_image  = np.flip(input_image,  0)
      target_image = np.flip(target_image, 0)

    if rand() < 0.5:
      # Flip horizontally
      input_image  = np.flip(input_image,  1)
      target_image = np.flip(target_image, 1)

    if rand() < 0.5:
      # Transpose
      input_image  = np.swapaxes(input_image,  0, 1)
      target_image = np.swapaxes(target_image, 0, 1)
      sy, sx = sx, sy

    # Zero pad the tiles (always makes a copy)
    pad_size = ((0, self.tile_size - sy), (0, self.tile_size - sx), (0, 0))
    input_image  = np.pad(input_image,  pad_size, mode='constant')

    target_image = np.pad(target_image, pad_size, mode='constant')

    # Randomly zero the color channels if there are auxiliary features
    # This prevents "ghosting" artifacts when the color buffer is entirely black
    # if len(self.channels) > 3 and rand() < 0.01:
    #   input_image[:, :, 0:3] = 0
    #   target_image[:] = 0

    # DEBUG: Save the tile
    # save_image('inputtile_%s.png' % input_name, input_image)
    # save_image('targettile_%s.png' % target_name, target_image)
 
    # Convert the tiles to tensors
    return image_to_tensor(input_image), image_to_tensor(target_image)

## -----------------------------------------------------------------------------
## Validation dataset
## -----------------------------------------------------------------------------

class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationDataset, self).__init__(cfg, name)

    # Split the images into tiles
    self.tiles = []

    for sample_index in range(self.num_images):
      # Get the input image
      input_name,  _ = self.samples[sample_index]
      input_image, _ = self.images[input_name]

      # Get the size of the image
      height = input_image.shape[0]
      width  = input_image.shape[1]
      if height < self.tile_size or width < self.tile_size:
        error('image is smaller than the tile size')

      # Compute the number of tiles
      num_tiles_y = height // self.tile_size
      num_tiles_x = width  // self.tile_size

      # Compute the start offset for centering
      start_y = (height % self.tile_size) // 2
      start_x = (width  % self.tile_size) // 2

      # Add the tiles
      for y in range(num_tiles_y):
        for x in range(num_tiles_x):
          oy = start_y + y * self.tile_size
          ox = start_x + x * self.tile_size
          self.tiles.append((sample_index, oy, ox))

  def __len__(self):
    return len(self.tiles)

  def __getitem__(self, index):
    # Get the tile
    sample_index, oy, ox = self.tiles[index]
    sy = sx = self.tile_size

    # Get the input and target images
    input_name, target_name = self.samples[sample_index]
    input_image,  _ = self.images[input_name]
    target_image, _ = self.images[target_name]

    # Crop the input and target images
    input_image  = input_image [oy:oy+sy, ox:ox+sx, self.channel_order]

    target_image = target_image[oy:oy+sy, ox:ox+sx, :3]

    # Convert the tiles to tensors
    # Copying is required because PyTorch does not support non-writeable tensors
    return image_to_tensor(input_image.copy()), image_to_tensor(target_image.copy())


## -----------------------------------------------------------------------------
## Training temporal dataset
## -----------------------------------------------------------------------------

class TrainingTemporalDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(TrainingTemporalDataset, self).__init__(cfg, name)

    # Set temporal size
    self.temp_size = cfg.temp_size

    # Set maximum padding
    self.max_padding = 32

  def stack_image(self, image):
    return image

  def augment_sequence(self, input, target):
    return input, target

  def __len__(self):
    return self.num_images

  def __getitem__(self, index):
    # Get the input and target images
    input_name, target_name = self.samples[index]
    input_image,  _ = self.images[input_name]
    input_image = self.stack_image(input_image)
    target_image, _ = self.images[target_name]
    target_image = self.stack_image(target_image)

    # Get the size of the image
    total_height = input_image.shape[0]
    height = total_height // self.temp_size
    width  = input_image.shape[1]
    if height < self.tile_size or width < self.tile_size:
      error('image is smaller than the tile size')

    # Generate a random crop
    sy = sx = self.tile_size

    if rand() < 0.1:
      # Randomly zero pad later to avoid artifacts for images that require padding
      sy -= randint(self.max_padding)
      sx -= randint(self.max_padding)
    oy = randint(int(0.2 * height), int(0.8 * height) - sy + 1)
    ox = randint(int(0.2 * width), int(0.8 * width)  - sx + 1)

    # Randomly permute some channels to improve training quality
    channels = self.channels[:] # copy
    # random.shuffle(channels)

    # Compute the indices of the required input channels
    channel_order = get_channel_indices(channels, self.data_channels)

    # Create input and target images and crop them through time
    # temp_input_image = np.empty((sy * self.temp_size, sx, input_image.shape[2]))
    # temp_target_image = np.empty((sy * self.temp_size, sx, 3))
    temp_input_image = np.empty((self.temp_size, sy, sx, len(channels)))
    temp_target_image = np.empty((self.temp_size, sy, sx, 3))

    for t in range(self.temp_size):
      # temp_input_image[t*sy:(t+1)*sy, :, :] = input_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, channel_order]
      # temp_target_image[t*sy:(t+1)*sy, :, :] = target_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, 0:3]
      temp_input_image[t, :, :, :] = input_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, channel_order]
      temp_target_image[t, :, :, :] = target_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, channel_order]

    # Randomly turn a real sequence into a static sequence without movement
    if rand() < 0.3:
      for t in range(1, self.temp_size):
        temp_input_image[t, :, :, :] = temp_input_image[0, :, :, :]
        temp_target_image[t, :, :, :] = temp_target_image[0, :, :, :]
    
    input_image, target_image = self.augment_sequence(temp_input_image, temp_target_image)
    # print(input_image.shape)
    # Randomly convert sequence to grayscale
    # if rand() < 0.3:
    #   #0.299 R + 0.587 G + 0.114 B
    #   x = 0.299 * input_image[..., 0] + 0.587 * input_image[..., 1] + 0.114 * input_image[..., 2]
    #   input_image = np.tile(np.expand_dims(x, axis=3), (1, 1, 1, 3))
    #   y = 0.299 * target_image[..., 0] + 0.587 * target_image[..., 1] + 0.114 * target_image[..., 2]
    #   target_image = np.tile(np.expand_dims(y, axis=3), (1, 1, 1, 3))

    # Randomly add noise to image
    # if rand() < 0.1:
    #   noise = np.random.uniform(0, 0.1, input_image.shape)
    #   input_image += noise[..., 0:1]
    #   #target_image += noise[..., 0:1]

    # Randomly transform the tiles to improve training quality
    if rand() < 0.5:
      # Flip vertically
      input_image  = np.flip(input_image,  1)
      target_image = np.flip(target_image, 1)

    if rand() < 0.5:
      # Flip horizontally
      input_image  = np.flip(input_image,  2)
      target_image = np.flip(target_image, 2)

    if rand() < 0.5:
      # Transpose
      input_image  = np.swapaxes(input_image,  1, 2)
      target_image = np.swapaxes(target_image, 1, 2)
      sy, sx = sx, sy

    # Zero pad the tiles (always makes a copy)
    pad_size = ((0,0), (0, self.tile_size - sy), (0, self.tile_size - sx), (0, 0))
    input_image  = np.pad(input_image,  pad_size, mode='constant')
    target_image = np.pad(target_image, pad_size, mode='constant')

    # DEBUG: Save the tile
    #save_image('tile_%d.png' % index, np.reshape(target_image, (target_image.shape[0]*target_image.shape[1], target_image.shape[2], target_image.shape[3])))
 
    # Convert the tiles to tensors
    return image_sequence_to_tensor(input_image), image_sequence_to_tensor(target_image)


## -----------------------------------------------------------------------------
## Validation temporal dataset
## -----------------------------------------------------------------------------

class ValidationTemporalDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationTemporalDataset, self).__init__(cfg, name)

    # Set temporal size
    self.temp_size = cfg.temp_size

    # Split the images into tiles
    self.tiles = []

    for sample_index in range(self.num_images):
      # Get the input image
      input_name,  _ = self.samples[sample_index]
      input_image, _ = self.images[input_name]
      input_image = self.stack_image(input_image)

      # Get the size of the image
      total_height = input_image.shape[0]
      height = total_height // self.temp_size
      width  = input_image.shape[1]
      if height < self.tile_size or width < self.tile_size:
        error('image is smaller than the tile size')

      # Compute the number of tiles
      num_tiles_y = height // self.tile_size
      num_tiles_x = width  // self.tile_size

      # Compute the start offset for centering
      start_y = (height % self.tile_size) // 2
      start_x = (width  % self.tile_size) // 2

      # Add the tiles
      for y in range(num_tiles_y):
        for x in range(num_tiles_x):
          oy = start_y + y * self.tile_size
          ox = start_x + x * self.tile_size
          self.tiles.append((sample_index, oy, ox))

  def stack_image(self, image):
    return image

  def augment_sequence(self, input, target):
    return input, target

  def __len__(self):
    return len(self.tiles)

  def __getitem__(self, index):
    # Get the tile
    sample_index, oy, ox = self.tiles[index]
    sy = sx = self.tile_size

    # Get the input and target images
    input_name, target_name = self.samples[sample_index]
    input_image,  _ = self.images[input_name]
    input_image = self.stack_image(input_image)
    target_image, _ = self.images[target_name]
    target_image = self.stack_image(target_image)

    # Get the size of the image
    total_height = input_image.shape[0]
    height = total_height // self.temp_size

    channels = self.channels[:]

    # Create input and target images and crop them through time
    temp_input_image = np.empty((self.temp_size, sy, sx, len(channels)))
    temp_target_image = np.empty((self.temp_size, sy, sx, 3))

    for t in range(self.temp_size):
      temp_input_image[t, :, :, :] = input_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, :len(channels)]
      temp_target_image[t, :, :, :] = target_image[(t*height)+oy:(t*height)+oy+sy, ox:ox+sx, 0:3]

    input_image, target_image = self.augment_sequence(temp_input_image, temp_target_image)

    # DEBUG: Save the tile
    #save_image('tile_%d.png' % index, np.reshape(target_image, (target_image.shape[0]*target_image.shape[1], target_image.shape[2], target_image.shape[3])))

    # Convert the tiles to tensors
    # Copying is required because PyTorch does not support non-writeable tensors
    return image_sequence_to_tensor(input_image.copy()), image_sequence_to_tensor(target_image.copy())


## -----------------------------------------------------------------------------
## Mock Training temporal dataset
## -----------------------------------------------------------------------------
class MockTrainingTemporalDataset(TrainingTemporalDataset):

  def __init__(self, cfg, name):
      super(MockTrainingTemporalDataset, self).__init__(cfg, name)

  def stack_image(self, image):
    return np.tile(image, (self.temp_size, 1, 1))

  # def augment_sequence(self, input, target):
  #   for t in range(1,self.temp_size):
  #     inp = input[t-1, ...]
  #     tar = target[t-1, ...]

  #     if rand() > 0.5:
  #       shift = (-5 + int(rand()*10), -5 + int(rand()*10))
  #       inp = np.roll(inp, shift, (0,1))
  #       tar = np.roll(tar, shift, (0,1))

  #     input[t, ...] = inp
  #     target[t, ...] = tar

  #   return input, target




## -----------------------------------------------------------------------------
## Mock Validation temporal dataset
## -----------------------------------------------------------------------------
class MockValidationTemporalDataset(ValidationTemporalDataset):

  def __init__(self, cfg, name):
      super(MockValidationTemporalDataset, self).__init__(cfg, name)

  def stack_image(self, image):
    return np.tile(image, (self.temp_size, 1, 1))

  # def augment_sequence(self, input, target):
  #   for t in range(1,self.temp_size):
  #     inp = input[t-1, ...]
  #     tar = target[t-1, ...]

  #     if rand() > 0.5:
  #       shift = (-5 + int(rand()*10), -5 + int(rand()*10))
  #       inp = np.roll(inp, shift, (0,1))
  #       tar = np.roll(tar, shift, (0,1))

  #     input[t, ...] = inp
  #     target[t, ...] = tar

  #   return input, target
