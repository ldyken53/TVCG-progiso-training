#!/usr/bin/env python3

## Copyright 2018-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import time
import numpy as np
import torch

from config import *
from util import *
from dataset import *
from model.settings import *
from color import *
from result import *

def main():
  times = []
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  # Parse the command line arguments
  cfg = parse_args(description='Performs inference on a dataset using the specified training result.')

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Open the result
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')
  print('Result:', cfg.result)

  # Load the result config
  result_cfg = load_config(result_dir)
  if cfg.temp_size is None:
    cfg.temp_size = result_cfg.temp_size
  cfg.features = result_cfg.features
  cfg.transfer = result_cfg.transfer
  cfg.model    = result_cfg.model
  if 'model_config' in result_cfg:
    cfg.model_config = result_cfg.model_config
  target_feature = 'hdr' if 'hdr' in cfg.features else 'ldr'

  # Reshape y stack temporal image to proper temporal dimension
  def reshape_image_temporal(image):
    total_height = image.shape[0]
    height = total_height // cfg.temp_size
    out = np.empty((cfg.temp_size, height, image.shape[1], image.shape[2]))

    for t in range(cfg.temp_size):
      out[t, ...] = image[t*height:(t+1)*height, ...]

    # total_height = image.shape[0]
    # height = total_height // cfg.temp_size
    # out = np.empty((cfg.temp_size, 1080, 1920, image.shape[2]))

    # for t in range(cfg.temp_size):
    #   # out[t, ...] = image[t*height:t*height+360, :640, :]
    #   out[t, :720, :1280, :] = image[t*height:(t+1)*height, ...]

    return out

  # Generate the header for the performance report
  def generate_report_header():
    header = ['frame']+[metric for metric in cfg.report]
    return tuple(header)

  # Generate a single row for the performance report
  def generate_report_row(frame, output, target, prev_output = None, prev_target = None):
    row = [frame]
    for metric in cfg.report:
      if metric == 'tpsnr':
        if prev_output != None and prev_target != None:
          doutput = output - prev_output
          dtarget = target - prev_target
          row.append(compare_images(doutput, dtarget, metric='psnr').item())
        else:
          row.append(-1)
      else:
        row.append(compare_images(output, target, metric=metric).item())
    
    return tuple(row)


  # Create shift vectores for mock_temporal
  # Since target and input generation are separate in this code, this is an easier workaround than augmenting both sets of images at the same time
  def generate_mock_shifts():
    shifts = list()
    for _ in range(cfg.temp_size):
      shifts.append((-5 + int(rand()*10), -5 + int(rand()*10)))

    return shifts

  # Simulate temporal changes if the dataset is static
  def mock_temporal(image, shifts):
    for t in range(1,cfg.temp_size):
      img = image[t-1, ...]

      
      shift = shifts[t]
      img = np.roll(img, shift, (0,1))

      image[t, ...] = img

    return image

  # Inference function
  def infer(model, transfer, input, exposure):
    x = input.clone()

    # Apply the transfer function
    color = x[:, :, 0:3, ...]
    if 'hdr' in cfg.features:
      color *= exposure
    color = transfer.forward(color)
    x[:, :, 0:3, ...] = color

    # Create output buffer
    outputs = list()
    features = None

    for t in range(cfg.temp_size):
        inp = x[:, t, ...]

        # Pad the output
        shape = inp.shape
        inp = F.pad(inp, (0, round_up(shape[3], model.alignment) - shape[3],
                        0, round_up(shape[2], model.alignment) - shape[2]))

        # Run the inference
        starter.record()
        output, features = model(inp, features)
        ender.record()
        torch.cuda.synchronize()
        infer_time = starter.elapsed_time(ender)
        print("Time for one inference:" + str(infer_time))
        outputs.append(output)

    for t in range(cfg.temp_size):
      # Unpad the output
      outputs[t] = outputs[t][:, :, :shape[2], :shape[3]]

      # Sanitize the output
      outputs[t] = torch.clamp(outputs[t], min=0.)

      # Apply the inverse transfer function
      outputs[t] = transfer.inverse(outputs[t])
      if 'hdr' in cfg.features:
        outputs[t] /= exposure
      else:
        outputs[t] = torch.clamp(outputs[t], max=1.)

    return torch.stack(outputs, dim=1), infer_time

  # Saves an image in different formats
  def save_images(path, images, images_srgb):
    for t in range(cfg.temp_size):
      image = images[:, t, ...]
      image_srgb = images_srgb[:, t, ...]
      image      = tensor_to_image(image)
      image_srgb = tensor_to_image(image_srgb)
      filename_prefix = path + '.' + target_feature + f'{t:05d}' + '.'
      for format in cfg.format:
        if format in {'exr', 'pfm', 'hdr'}:
          save_image(filename_prefix + format, image)
        else:
          save_image(filename_prefix + format, image_srgb)


  # Initialize the report
  if cfg.report is not None:
    report = []
    report.append(generate_report_header())

  # Initialize the dataset
  data_dir = get_data_dir(cfg, cfg.input_data)
  print(data_dir)
  image_sample_groups = get_image_sample_groups(data_dir, cfg.features)

  # Initialize the model
  model = get_model(cfg)
  # print(model)
  model.to(device)

  # Load the checkpoint
  checkpoint = load_checkpoint(result_dir, device, cfg.checkpoint, model)
  epoch = checkpoint['epoch']

  # Initialize the transfer function
  transfer = get_transfer_function(cfg)

  # Iterate over the images
  print()
  output_dir = os.path.join(cfg.output_dir, cfg.input_data)
  metric_sum = {metric : 0. for metric in cfg.metric}
  print(metric_sum)
  metric_count = 0
  model.eval()
  warmup = True

  with torch.no_grad():
    for group, input_names, target_name in image_sample_groups:
      # Create the output directory if it does not exist
      output_group_dir = os.path.join(output_dir, os.path.dirname(group))
      if not os.path.isdir(output_group_dir):
        os.makedirs(output_group_dir)

      # Load metadata for the images if it exists
      tonemap_exposure = 1.
      metadata = load_image_metadata(os.path.join(data_dir, group))
      if metadata:
        tonemap_exposure = metadata['exposure']
        save_image_metadata(os.path.join(output_dir, group), metadata)

      # Generate shifts to simulate temporal data
      shifts = generate_mock_shifts()

      # Load the target image if it exists
      if target_name:
        target = load_target_image(os.path.join(data_dir, target_name), cfg.features)
        #target = np.tile(target, (cfg.temp_size, 1, 1))
        target = reshape_image_temporal(target)
        #target = mock_temporal(target, shifts)
        target = target.astype(np.float32)
        target = image_sequence_to_tensor(target, batch=True).to(device)
        target_srgb = transform_feature(target, target_feature, 'srgb', tonemap_exposure)

      # Iterate over the input images
      for input_name in input_names:
        print(input_name, '...', end='', flush=True)

        # Load the input image
        input = load_input_image(os.path.join(data_dir, input_name), cfg.features)
        #input = np.tile(input, (cfg.temp_size, 1, 1))
        input = reshape_image_temporal(input)
        #input = mock_temporal(input, shifts)
        input = input.astype(np.float32)

        # Compute the autoexposure value
        exposure = autoexposure(input) if 'hdr' in cfg.features else 1.

        # Infer
        print("TEMP_SIZE", cfg.temp_size)
        input = image_sequence_to_tensor(input, batch=True).to(device)
        output, infer_time = infer(model, transfer, input, exposure)
        if not warmup:
          times.append(infer_time)
        warmup = False

        input = input[:, :, 0:3, ...] # keep only the color
        output = output[:, :, :3,...]
        input_srgb  = transform_feature(input,  target_feature, 'srgb', tonemap_exposure)
        output_srgb = transform_feature(output, target_feature, 'srgb', tonemap_exposure)

        # Compute metrics
        metric_str = ''
        if target_name and cfg.metric:
          for metric in cfg.metric:
            value = compare_image_sequences(output_srgb, target_srgb, metric)
            metric_sum[metric] += value
            if metric_str:
              metric_str += ', '
            metric_str += f'{metric}={value:.4f}'
            # inp_value = compare_image_sequences(input_srgb, target_srgb, metric)
            # metric_str += f', input {metric}={inp_value:.4f}'
          metric_count += 1

        # Save the input and output images
        output_name = input_name + '.' + cfg.result
        if cfg.checkpoint:
          output_name += f'_{epoch}'

        # if not cfg.report:
        if cfg.save_all:
          save_images(os.path.join(output_dir, input_name), input, input_srgb)
        save_images(os.path.join(output_dir, output_name), output, output_srgb)

        # Generate report rows
        if cfg.report is not None:
          for t in range(cfg.temp_size):
            if t > 0:
              rowin = generate_report_row(t, input[:, t, ...], target[:, t, ...], input[:, t-1, ...], target[:, t-1, ...])
              rowout = generate_report_row(t, output[:, t, ...], target[:, t, ...], output[:, t-1, ...], target[:, t-1, ...])
            else:
              rowin = generate_report_row(t, input[:, t, ...], target[:, t, ...])
              rowout = generate_report_row(t, output[:, t, ...], target[:, t, ...])
            report.append(rowin)
            report.append(rowout)
          save_csv(os.path.join(output_dir, output_name+"_report.csv"), report)

        # Print metrics
        if metric_str:
          metric_str = ' ' + metric_str
        print(metric_str)

      # Save the target image if it exists
      if cfg.save_all and target_name:
        save_images(os.path.join(output_dir, target_name), target, target_srgb)

  # Print summary
  if metric_count > 0:
    metric_str = ''
    for metric in cfg.metric:
      value = metric_sum[metric] / metric_count
      if metric_str:
        metric_str += ', '
      metric_str += f'{metric}_avg={value:.4f}'
    print()
    print(f'{cfg.result}: {metric_str} ({metric_count} images)')
    print()
    print(f"Mean inference time: {np.mean(times)}, Median inference time: {np.median(times)}")

if __name__ == '__main__':
  main()