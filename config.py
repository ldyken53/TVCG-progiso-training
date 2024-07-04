## Copyright 2018-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
from argparse import Namespace
import time
import torch

from util import *


# Returns the config filename in a directory
def get_config_filename(dir):
  return os.path.join(dir, 'config.json')

# Loads the config from a directory
def load_config(dir):
  filename = get_config_filename(dir)
  cfg = load_json(filename)
  return argparse.Namespace(**cfg)

# Saves the config to a directory
def save_config(dir, cfg):
  filename = get_config_filename(dir)
  save_json(filename, vars(cfg))

# Parses the config from the command line arguments
def parse_args(cmd=None, description=None):
  def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

  if cmd is None:
    cmd, _ = os.path.splitext(os.path.basename(sys.argv[0]))

  parser = argparse.ArgumentParser(description=description)
  parser.usage = '\rFoVolNet - Training\n' + parser.format_usage()
  advanced = parser.add_argument_group('optional advanced arguments')


  if cmd in {'preprocess', 'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr', 'vgg19_helper'}:
    parser.add_argument('features', type=str, nargs='*', choices=['hdr', 'ldr', 'albedo', 'alb', 'normal', 'nrm', []], help='set of input features')
    parser.add_argument('--filter', '-f', type=str, choices=['RT', 'RTLightmap'], help='filter to train (sets some default arguments)')
    parser.add_argument('--preproc_dir', '-P', type=str, default='preproc', help='directory of preprocessed datasets')
    parser.add_argument('--train_data', '-t', type=str, default='train', help='name of the training dataset')
    advanced.add_argument('--transfer', '-x', type=str, choices=['srgb', 'pu', 'log'], help='transfer function')

  if cmd in {'preprocess', 'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan',}:
    parser.add_argument('--valid_data', '-v', type=str, default='valid', help='name of the validation dataset')

  if cmd in {'preprocess', 'infer', 'infer_temporal', 'infer_temporal_integrated'}:
    parser.add_argument('--data_dir', '-D', type=str, default='data', help='directory of datasets (e.g. training, validation, test)')

  if cmd in {'preprocess'}:
    parser.add_argument('--clean', action='store_true', help='delete existing preprocessed datasets')

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr', 'infer', 'infer_temporal', 'infer_temporal_integrated', 'export', 'visualize'}:
    parser.add_argument('--results_dir', '-R', type=str, default='results', help='directory of training results')
    parser.add_argument('--result', '-r', type=str, required=(not cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr'}), help='name of the result to save/load')

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'infer', 'infer_temporal', 'infer_temporal_integrated', 'export'}:
    parser.add_argument('--checkpoint', '-c', type=int, default=0, help='result checkpoint to restore')

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'infer', 'infer_temporal', 'infer_temporal_integrated', 'find_lr', 'vgg19_helper'}:
    parser.add_argument('--config', default='configs/train.json', required=False, help="config file")

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan'}:
    parser.add_argument('--epochs', '-e', type=int, default=2100, help='number of training epochs')
    parser.add_argument('--valid_epochs', type=int, default=10, help='perform validation every this many epochs')
    parser.add_argument('--save_epochs', type=int, default=10, help='save checkpoints every this many epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-6, help='minimum learning rate')
    parser.add_argument('--max_lr', '--max_learning_rate', type=float, default=2e-4, help='maximum learning rate')
    parser.add_argument('--lr_cycle_epochs', type=int, default=250, help='number of epochs per learning rate cycle (for CLR)')
    parser.add_argument('--precision', '-p', type=str, choices=['fp32', 'mixed'], help='training precision')

  if cmd in {'find_lr'}:
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-8, help='minimum learning rate')
    parser.add_argument('--max_lr', '--max_learning_rate', type=float, default=0.1, help='maximum learning rate')

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr', 'vgg19_helper'}:
    parser.add_argument('--batch_size', '--bs', type=int, default=8, help='size of the mini-batches')
    parser.add_argument('--loaders', type=int, default=4, help='number of data loader threads per device')
    advanced.add_argument('--model', '-m', type=str, choices=['unet'], default='unet', help='network model')
    advanced.add_argument('--loss', '-l', type=str, choices=['l1', 'mape', 'smape', 'l2', 'ssim', 'msssim', 'l1_msssim', 'l1_grad'], default='l1_msssim', help='loss function')
    advanced.add_argument('--tile_size', '--ts', type=int, default=256, help='size of the cropped image tiles')
    advanced.add_argument('--seed', '-s', type=int, help='seed for random number generation')

  if cmd in {'infer', 'infer_temporal', 'infer_temporal_integrated', 'compare_image'}:
    parser.add_argument('--metric', '-M', type=str, nargs='*', choices=['psnr', 'mse', 'ssim'], default=['psnr', 'ssim'], help='metrics to compute')

  if cmd in {'infer', 'infer_temporal', 'infer_temporal_integrated'}:
    parser.add_argument('--input_data', '-i', type=str, default='test', help='name of the input dataset')
    parser.add_argument('--output_dir', '-O', type=str, default='infer', help='directory of output images')
    parser.add_argument('--format', '-F', type=str, nargs='*', default=['exr'], help='output image formats')
    parser.add_argument('--save_all', '-a', action='store_true', help='save input and target images too')
    parser.add_argument('--report', type=str, nargs='*', choices=['psnr', 'ssim', 'msssim', 'tpsnr'], help='Save a report (.csv) containing performance metrics')

  if cmd in {'export'}:
    parser.add_argument('--format', '-F', type=str, default='torch', help='output model format (\'torch\' or \'onnx\')')
    parser.add_argument('--input_dimensions', type=int, nargs='*', default=[1, 3, 1080, 1920], help='input dimensions to trace the model with')

  if cmd in {'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'infer_temporal', 'infer_temporal_integrated'}: 
    parser.add_argument('--temp_size', type=int, default=16, help='Number of frames to use for temporal training and inference')


  if cmd in {'convert_image', 'convert_dataset', 'temporalize_dataset', 'temporalize_dataset_new'}:
    parser.add_argument('--input', type=str, help='input image')

  if cmd in {'convert_image', 'convert_dataset', 'temporalize_dataset', 'temporalize_dataset_new'}:
    parser.add_argument('--output', type=str, help='output image')

  if cmd in {'convert_image', 'convert_dataset'}:
    parser.add_argument('--exposure', '-E', type=float, default=1., help='linear exposure scale for HDR image')

  if cmd in {'convert_dataset'}:
    parser.add_argument('--format', '-f', type=str, required=True, help='Input format of the images')
    parser.add_argument('--suffix', '-s', type=str, default='ref', help='The suffix to append to the file name')
    parser.add_argument('--cropx', type=int, default=0, help='Amount to crop in the x axis in percent')
    parser.add_argument('--cropy', type=int, default=0, help='Amount to crop in the y axis in percent')

  if cmd in {'preprocess', 'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'infer', 'infer_temporal', 'infer_temporal_integrated', 'export'}:
    parser.add_argument('--device', '-d', type=str,
                        choices=['cpu', 'cuda'], default=get_default_device(),
                        help='type of device(s) to use')
    parser.add_argument('--device_id', '-k', type=int, default=0,
                        help='ID of the first device to use')
    parser.add_argument('--num_devices', '-n', type=int, default=1,
                        help='number of devices to use (with IDs device_id .. device_id+num_devices-1)')
    advanced.add_argument('--deterministic', '--det', action='store_true',
                          default=(cmd in {'preprocess', 'infer', 'export'}),
                          help='makes computations deterministic (slower performance)')

  if cmd in {'temporalize_dataset', 'temporalize_dataset_new'}:
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per sequence')
    parser.add_argument('--overlap', type=int, default=0, help='Number of frames to overlap between sequences')

  # convert arg parse to dict
  cfg = vars(parser.parse_args())

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr', 'vgg19_helper'}:
    # load json config
    with open(cfg['config']) as f:
      config = json.load(f)

    # update cfg dict with parameter from config file 
    cfg.update(config)

  # convert dict to namespace object
  cfg = Namespace(**cfg)

  if cmd in {'preprocess', 'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan',}:

# Replace feature names with IDs
    FEATURE_IDS = {'albedo' : 'alb', 'normal' : 'nrm'}
    cfg.features = [FEATURE_IDS.get(f, f) for f in cfg.features]
    # Remove duplicate features
    cfg.features = list(dict.fromkeys(cfg.features).keys())

    # Check features
    if ('ldr' in cfg.features) == ('hdr' in cfg.features):
      parser.error('either hdr or ldr must be specified as input feature')

    # Set the transfer function if not specified
    if not cfg.transfer:
      if 'hdr' in cfg.features:
        cfg.transfer = 'log' if cfg.filter == 'RTLightmap' else 'pu'
      else:
        cfg.transfer = 'srgb'

  if cmd in {'train', 'train_temporal', 'train_deepfovea', 'train_temporal_opticalflow', 'train_temporal_gan', 'find_lr'}:
      # Set the default training precision
    if cfg.precision is None:
      cfg.precision = 'mixed' if cfg.device == 'cuda' else 'fp32'

    # Check the batch size
    if cfg.batch_size % cfg.num_devices != 0:
      parser.error('batch_size is not divisible by num_devices')

    # Generate a result name if not specified
    if not cfg.result:
      cfg.result = '%x' % int(time.time())

  # Print PyTorch version
  print('PyTorch:', torch.__version__)

  return cfg

# Returns the config filename in a directory
def get_config_filename(dir):
  return os.path.join(dir, 'config.json')

# Loads the config from a directory
def load_config(dir):
  filename = get_config_filename(dir)
  cfg = load_json(filename)
  return argparse.Namespace(**cfg)

# Saves the config to a directory
def save_config(dir, cfg):
  filename = get_config_filename(dir)
  save_json(filename, vars(cfg))

  # Print PyTorch version
  print('PyTorch:', torch.__version__)

  return cfg