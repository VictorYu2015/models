# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions used by target assigner."""

import tensorflow as tf


def image_shape_to_grids(height, width):
  """Computes xy-grids given the shape of the image.

  Args:
    height: The height of the image.
    width: The width of the image.

  Returns:
    A tuple of two tensors:
      y_grid: A float tensor with shape [height, width] representing the
        y-coordinate of each pixel grid.
      x_grid: A float tensor with shape [height, width] representing the
        x-coordinate of each pixel grid.
  """
  out_height = tf.cast(height, tf.float32)
  out_width = tf.cast(width, tf.float32)
  x_range = tf.range(out_width, dtype=tf.float32)
  y_range = tf.range(out_height, dtype=tf.float32)
  x_grid, y_grid = tf.meshgrid(x_range, y_range, indexing='xy')
  return (y_grid, x_grid)


def coordinates_to_heatmap(y_grid,
                           x_grid,
                           y_coordinates,
                           x_coordinates,
                           sigma,
                           channel_onehot,
                           channel_weights=None):
  """Returns the heatmap targets from a set of point coordinates.

  This function maps a set of point coordinates to the output heatmap image
  applied using a Gaussian kernel. Note that this function be can used by both
  object detection and keypoint estimation tasks. For object detection, the
  "channel" refers to the object class. For keypoint estimation, the "channel"
  refers to the keypoint types.

  Args:
    y_grid: A 2D tensor with shape [height, width] which contains the grid
      y-coordinates given in the (output) image dimensions.
    x_grid: A 2D tensor with shape [height, width] which contains the grid
      x-coordinates given in the (output) image dimensions.
    y_coordinates: A 1D tensor with shape [num_instances] representing the
      y-coordinates of the instances in the output space coordinates.
    x_coordinates: A 1D tensor with shape [num_instances] representing the
      x-coordinates of the instances in the output space coordinates.
    sigma: A 1D tensor with shape [num_instances] representing the standard
      deviation of the Gaussian kernel to be applied to the keypoint.
    channel_onehot: A 2D tensor with shape [num_instances, num_channels]
      representing the one-hot encoded channel labels for each keypoint.
    channel_weights: A 1D tensor with shape [num_instances] corresponding to the
      weight of each instance.

  Returns:
    heatmap: A tensor of size [height, width, num_channels] representing the
      heatmap. Output (height, width) match the dimensions of the input grids.
  """
  num_instances = tf.shape(channel_onehot)[0]
  num_channels = tf.shape(channel_onehot)[1]

  x_grid = tf.expand_dims(x_grid, 2)
  y_grid = tf.expand_dims(y_grid, 2)
  # The raw center coordinates in the output space.
  x_diff = x_grid - tf.math.floor(x_coordinates)
  y_diff = y_grid - tf.math.floor(y_coordinates)
  squared_distance = x_diff**2 + y_diff**2

  gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))

  reshaped_gaussian_map = tf.expand_dims(gaussian_map, axis=-1)
  reshaped_channel_onehot = tf.reshape(channel_onehot,
                                       (1, 1, num_instances, num_channels))
  gaussian_per_box_per_class_map = (
      reshaped_gaussian_map * reshaped_channel_onehot)

  if channel_weights is not None:
    reshaped_weights = tf.reshape(channel_weights, (1, 1, num_instances, 1))
    gaussian_per_box_per_class_map *= reshaped_weights

  # Take maximum along the "instance" dimension so that all per-instance
  # heatmaps of the same class are merged together.
  heatmap = tf.reduce_max(gaussian_per_box_per_class_map, axis=2)

  # Maximum of an empty tensor is -inf, the following is to avoid that.
  heatmap = tf.maximum(heatmap, 0)

  return heatmap


def compute_floor_offsets_with_indices(
    y_source,
    x_source,
    y_target=None,
    x_target=None):
  """Computes offsets from floored source(floored) to target coordinates.

  This function computes the offsets from source coordinates ("floored" as if
  they were put on the grids) to target coordinates. Note that the input
  coordinates should be the "absolute" coordinates in terms of the output image
  dimensions as opposed to the normalized coordinates (i.e. values in [0, 1]).

  Args:
    y_source: A tensor with shape [num_points] representing the absolute
      y-coordinates (in the output image space) of the source points.
    x_source: A tensor with shape [num_points] representing the absolute
      x-coordinates (in the output image space) of the source points.
    y_target: A tensor with shape [num_points] representing the absolute
      y-coordinates (in the output image space) of the target points. If not
      provided, then y_source is used as the targets.
    x_target: A tensor with shape [num_points] representing the absolute
      x-coordinates (in the output image space) of the target points. If not
      provided, then x_source is used as the targets.

  Returns:
    A tuple of two tensors:
      offsets: A tensor with shape [num_points, 2] representing the offsets of
        each input point.
      indices: A tensor with shape [num_points, 2] representing the indices of
        where the offsets should be retrieved in the output image dimension
        space.
  """
  y_source_floored = tf.floor(y_source)
  x_source_floored = tf.floor(x_source)
  if y_target is None:
    y_target = y_source
  if x_target is None:
    x_target = x_source

  y_offset = y_target - y_source_floored
  x_offset = x_target - x_source_floored

  y_source_indices = tf.cast(y_source_floored, tf.int32)
  x_source_indices = tf.cast(x_source_floored, tf.int32)

  indices = tf.stack([y_source_indices, x_source_indices], axis=1)
  offsets = tf.stack([y_offset, x_offset], axis=1)

  return offsets, indices


def get_valid_keypoint_mask_for_class(keypoint_coordinate,
                                      class_id,
                                      class_onehot,
                                      keypoint_indices=None):
  """Mask keypoints by their class ids and indices.

  For a given task, we may want to only consider a subset of instances or
  keypoints. This function is used to provide the mask (in terms of weights) to
  mark those elements which should be considered based on the classes of the
  instances and optionally, their keypoint indices. Note that the NaN values
  in the keypoints will also be masked out.

  Args:
    keypoint_coordinate: A float tensor with shape [num_instances,
      num_keypoints] which contains the values of each keypoint. The value could
      be either y or x coordinates of the keypoints.
    class_id: An integer representing the target class id to be selected.
    class_onehot: A 2D tensor of shape [num_instances, num_classes] repesents
      the onehot (or k-hot) encoding of the class for each instance.
    keypoint_indices: A list of integers representing the keypoint indices used
      to select the values on the keypoint dimension. If provided, the output
      dimension will be [num_instances, len(keypoint_indices)]

  Returns:
    A tuple of tensors:
      mask: A float tensor of shape [num_instances, K], where K is num_keypoints
        or len(keypoint_indices) if provided. The tensor has values either 0 or
        1 indicating whether an element in the input keypoints should be used.
      keypoints_nan_to_zeros: Same as input keypoints with the NaN values
        replaced by zeros and selected columns corresponding to the
        keypoint_indices (if provided). The shape of this tensor will always be
        the same as the output mask.
  """
  (_, num_keypoints) = tf.shape(keypoint_coordinate)
  class_mask = class_onehot[:, class_id]
  reshaped_class_mask = tf.tile(
      tf.expand_dims(class_mask, axis=-1), multiples=[1, num_keypoints])
  not_nan = tf.math.logical_not(tf.math.is_nan(keypoint_coordinate))
  mask = reshaped_class_mask * tf.cast(not_nan, dtype=tf.float32)
  keypoints_nan_to_zeros = tf.where(not_nan, keypoint_coordinate,
                                    tf.zeros_like(keypoint_coordinate))

  if keypoint_indices is not None:
    mask = tf.gather(mask, indices=keypoint_indices, axis=1)
    keypoints_nan_to_zeros = tf.gather(
        keypoints_nan_to_zeros, indices=keypoint_indices, axis=1)
  return mask, keypoints_nan_to_zeros


def normalized_to_absolute(height, width, y_normalized, x_normalized):
  """Converts the normalized coordinates [0~1] to absolute coordinates."""
  return height * y_normalized, width * x_normalized
