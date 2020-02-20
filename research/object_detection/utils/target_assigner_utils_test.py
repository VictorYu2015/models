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
"""Tests for utils.target_assigner_utils."""

import numpy as np
import tensorflow as tf

from object_detection.utils import target_assigner_utils as ta_utils


class TargetUtilTest(tf.test.TestCase):

  def test_image_shape_to_grids(self):
    (y_grid, x_grid) = ta_utils.image_shape_to_grids(height=2, width=3)
    expected_y_grid = np.array([[0, 0, 0], [1, 1, 1]])
    expected_x_grid = np.array([[0, 1, 2], [0, 1, 2]])

    np.testing.assert_array_equal(y_grid.numpy(), expected_y_grid)
    np.testing.assert_array_equal(x_grid.numpy(), expected_x_grid)

  def test_coordinates_to_heatmap(self):
    (y_grid, x_grid) = ta_utils.image_shape_to_grids(height=3, width=5)
    y_coordinates = tf.constant([1.5, 0.5], dtype=tf.float32)
    x_coordinates = tf.constant([2.5, 4.5], dtype=tf.float32)
    sigma = tf.constant([0.1, 0.5], dtype=tf.float32)
    channel_onehot = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    channel_weights = tf.constant([1, 1], dtype=tf.float32)
    heatmap = ta_utils.coordinates_to_heatmap(y_grid, x_grid, y_coordinates,
                                              x_coordinates, sigma,
                                              channel_onehot, channel_weights)
    # Peak at (1, 2) for the first class.
    self.assertAlmostEqual(1.0, heatmap.numpy()[1, 2, 0])
    # Peak at (0, 4) for the second class.
    self.assertAlmostEqual(1.0, heatmap.numpy()[0, 4, 1])

  def test_compute_floor_offsets_with_indices_onlysource(self):
    y_source = tf.constant([1.5, 0.3], dtype=tf.float32)
    x_source = tf.constant([2.5, 4.2], dtype=tf.float32)
    (offsets,
     indices) = ta_utils.compute_floor_offsets_with_indices(y_source, x_source)

    np.testing.assert_array_almost_equal(offsets.numpy(),
                                         np.array([[0.5, 0.5], [0.3, 0.2]]))
    np.testing.assert_array_almost_equal(indices.numpy(),
                                         np.array([[1, 2], [0, 4]]))

  def test_compute_floor_offsets_with_indices_and_targets(self):
    y_source = tf.constant([1.5, 0.3], dtype=tf.float32)
    x_source = tf.constant([2.5, 4.2], dtype=tf.float32)
    y_target = tf.constant([2.1, 0.1], dtype=tf.float32)
    x_target = tf.constant([1.2, 4.5], dtype=tf.float32)
    (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
        y_source, x_source, y_target, x_target)

    np.testing.assert_array_almost_equal(offsets.numpy(),
                                         np.array([[1.1, -0.8], [0.1, 0.5]]))
    np.testing.assert_array_almost_equal(indices.numpy(),
                                         np.array([[1, 2], [0, 4]]))

  def test_get_valid_keypoints_mask(self):
    class_onehot = tf.constant(
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1]], dtype=tf.float32)
    keypoints = tf.constant(
        [[0.1, float('nan'), 0.2, 0.0],
         [0.0, 0.0, 0.1, 0.9],
         [3.2, 4.3, float('nan'), 0.2]],
        dtype=tf.float32)
    mask, keypoints_nan_to_zeros = ta_utils.get_valid_keypoint_mask_for_class(
        keypoint_coordinate=keypoints,
        class_id=2,
        class_onehot=class_onehot,
        keypoint_indices=[1, 2])
    expected_mask = np.array([[0, 1], [0, 0], [1, 0]])
    expected_keypoints = tf.constant([[0.0, 0.2], [0.0, 0.1], [4.3, 0.0]],
                                     dtype=tf.float32)
    np.testing.assert_array_equal(mask.numpy(), expected_mask)
    np.testing.assert_array_equal(keypoints_nan_to_zeros.numpy(),
                                  expected_keypoints)

  def test_normalized_to_absolute(self):
    y_coordinates = tf.constant([0.5, 0.1], dtype=tf.float32)
    x_coordinates = tf.constant([0.2, 0.9], dtype=tf.float32)
    (y_abs, x_abs) = ta_utils.normalized_to_absolute(100, 80, y_coordinates,
                                                     x_coordinates)

    np.testing.assert_array_almost_equal(y_abs.numpy(), np.array([50.0, 10.0]))
    np.testing.assert_array_almost_equal(x_abs.numpy(), np.array([16.0, 72.0]))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
