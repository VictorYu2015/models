# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the CenterNet Meta architecture code."""

from __future__ import division

import functools

import numpy as np
import tensorflow as tf

from object_detection.core import losses
from object_detection.core import preprocessor
from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models import center_net_resnet_feature_extractor


class CenterNetMetaArchPredictionHeadTest(tf.test.TestCase):
  """Test CenterNet meta architecture prediction head."""

  def test_prediction_head(self):
    head = center_net_meta_arch.make_prediction_net(num_out_channels=7)
    output = head(np.zeros((4, 128, 128, 8)))

    self.assertEqual((4, 128, 128, 7), output.shape)


class CenterNetMetaArchHelpersTest(tf.test.TestCase):
  """Test for CenterNet meta architecture related functions."""

  def test_row_col_indices_from_flattened_indices(self):
    """Tests that the computation of row, col, channel indices is correct."""

    r_grid, c_grid, ch_grid = (np.zeros((5, 4, 3), dtype=np.int),
                               np.zeros((5, 4, 3), dtype=np.int),
                               np.zeros((5, 4, 3), dtype=np.int))

    r_grid[..., 0] = r_grid[..., 1] = r_grid[..., 2] = np.array(
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    )

    c_grid[..., 0] = c_grid[..., 1] = c_grid[..., 2] = np.array(
        [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]
    )

    for i in range(3):
      ch_grid[..., i] = i

    indices = np.arange(60)
    ri, ci, chi = center_net_meta_arch.row_col_channel_indices_from_flattened_indices(
        indices, 4, 3)

    np.testing.assert_array_equal(ri, r_grid.flatten())
    np.testing.assert_array_equal(ci, c_grid.flatten())
    np.testing.assert_array_equal(chi, ch_grid.flatten())

  def test_flattened_indices_from_row_col_indices(self):

    r = np.array(
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2]]
    )

    c = np.array(
        [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]
    )

    idx = center_net_meta_arch.flattened_indices_from_row_col_indices(r, c, 4)
    np.testing.assert_array_equal(np.arange(12), idx.flatten())

  def test_get_valid_anchor_weights_in_flattened_image(self):
    """Tests that the anchor weights are valid upon flattening out."""

    valid_weights = np.zeros((2, 5, 5), dtype=np.float)

    valid_weights[0, :3, :4] = 1.0
    valid_weights[1, :2, :2] = 1.0

    true_image_shapes = tf.constant([[3, 4], [2, 2]])
    w = center_net_meta_arch.get_valid_anchor_weights_in_flattened_image(
        true_image_shapes, 5, 5)

    np.testing.assert_allclose(w, valid_weights.reshape(2, -1))
    self.assertEqual((2, 25), w.shape)

  def test_convert_strided_predictions_to_normalized_boxes(self):
    """Tests that boxes have correct coordinates in normalized input space."""

    boxes = np.zeros((2, 3, 4), dtype=np.float32)

    boxes[0] = [[10, 20, 30, 40], [20, 30, 50, 100], [50, 60, 100, 180]]
    boxes[1] = [[-5, -5, 5, 5], [45, 60, 110, 120], [150, 150, 200, 250]]

    true_image_shapes = tf.constant([[100, 90, 3], [150, 150, 3]])

    clipped_boxes = center_net_meta_arch.\
                    convert_strided_predictions_to_normalized_boxes(
                        boxes, 2, true_image_shapes)

    expected_boxes = np.zeros((2, 3, 4), dtype=np.float32)
    expected_boxes[0] = [[0.2, 4./9, 0.6, 8./9], [0.4, 2./3, 1, 1],
                         [1, 1, 1, 1]]
    expected_boxes[1] = [[0., 0, 1./15, 1./15], [3./5, 4./5, 1, 1],
                         [1, 1, 1, 1]]

    np.testing.assert_allclose(expected_boxes, clipped_boxes)

  def test_top_k_feature_map_locations(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 1] = 1.0
    feature_map_np[0, 2, 1, 1] = 0.9  # Get's filtered due to max pool.
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 0] = 0.5
    feature_map_np[0, 2, 2, 1] = -0.3
    feature_map_np[1, 2, 1, 1] = 0.7
    feature_map_np[1, 1, 0, 0] = 0.4
    feature_map_np[1, 1, 2, 0] = 0.1
    feature_map = tf.constant(feature_map_np)
    scores, y_inds, x_inds, channel_inds = (
        center_net_meta_arch.top_k_feature_map_locations(
            feature_map, max_pool_kernel_size=3, k=3))

    np.testing.assert_allclose([1.0, 0.7, 0.5], scores[0])
    np.testing.assert_array_equal([2, 0, 2], y_inds[0])
    np.testing.assert_array_equal([0, 1, 2], x_inds[0])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.1], scores[1])
    np.testing.assert_array_equal([2, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 2], x_inds[1])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[1])

  def test_top_k_feature_map_locations_no_pooling(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 1] = 1.0
    feature_map_np[0, 2, 1, 1] = 0.9
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 0] = 0.5
    feature_map_np[0, 2, 2, 1] = -0.3
    feature_map_np[1, 2, 1, 1] = 0.7
    feature_map_np[1, 1, 0, 0] = 0.4
    feature_map_np[1, 1, 2, 0] = 0.1
    feature_map = tf.constant(feature_map_np)
    scores, y_inds, x_inds, channel_inds = (
        center_net_meta_arch.top_k_feature_map_locations(
            feature_map, max_pool_kernel_size=1, k=3))

    np.testing.assert_allclose([1.0, 0.9, 0.7], scores[0])
    np.testing.assert_array_equal([2, 2, 0], y_inds[0])
    np.testing.assert_array_equal([0, 1, 1], x_inds[0])
    np.testing.assert_array_equal([1, 1, 0], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.1], scores[1])
    np.testing.assert_array_equal([2, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 2], x_inds[1])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[1])

  def test_top_k_feature_map_locations_per_channel(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 0] = 1.0  # Selected.
    feature_map_np[0, 2, 1, 0] = 0.9  # Selected.
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 1] = 0.5  # Selected.
    feature_map_np[0, 0, 0, 1] = 0.3  # Selected.
    feature_map_np[1, 2, 1, 0] = 0.7  # Selected.
    feature_map_np[1, 1, 0, 0] = 0.4  # Selected.
    feature_map_np[1, 1, 2, 0] = 0.3
    feature_map_np[1, 1, 0, 1] = 0.8  # Selected.
    feature_map_np[1, 1, 2, 1] = 0.3  # Selected.
    feature_map = tf.constant(feature_map_np)
    scores, y_inds, x_inds, channel_inds = (
        center_net_meta_arch.top_k_feature_map_locations(
            feature_map, max_pool_kernel_size=1, k=2, per_channel=True))

    np.testing.assert_allclose([1.0, 0.9, 0.5, 0.3], scores[0])
    np.testing.assert_array_equal([2, 2, 2, 0], y_inds[0])
    np.testing.assert_array_equal([0, 1, 2, 0], x_inds[0])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.8, 0.3], scores[1])
    np.testing.assert_array_equal([2, 1, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 0, 2], x_inds[1])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[1])

  def test_top_k_feature_map_locations_per_channel_in_graph_mode(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 0] = 1.0  # Selected.
    feature_map_np[0, 2, 1, 0] = 0.9  # Selected.
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 1] = 0.5  # Selected.
    feature_map_np[0, 0, 0, 1] = 0.3  # Selected.
    feature_map_np[1, 2, 1, 0] = 0.7  # Selected.
    feature_map_np[1, 1, 0, 0] = 0.4  # Selected.
    feature_map_np[1, 1, 2, 0] = 0.3
    feature_map_np[1, 1, 0, 1] = 0.8  # Selected.
    feature_map_np[1, 1, 2, 1] = 0.3  # Selected.
    feature_map = tf.constant(feature_map_np)

    @tf.function
    def _top_k_feature_map_locations_wrapped(feature_map):
      scores, y_inds, x_inds, channel_inds = (
          center_net_meta_arch.top_k_feature_map_locations(
              feature_map, max_pool_kernel_size=1, k=2, per_channel=True))
      return scores, y_inds, x_inds, channel_inds

    scores, y_inds, x_inds, channel_inds = _top_k_feature_map_locations_wrapped(
        feature_map)

    np.testing.assert_allclose([1.0, 0.9, 0.5, 0.3], scores[0])
    np.testing.assert_array_equal([2, 2, 2, 0], y_inds[0])
    np.testing.assert_array_equal([0, 1, 2, 0], x_inds[0])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.8, 0.3], scores[1])
    np.testing.assert_array_equal([2, 1, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 0, 2], x_inds[1])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[1])

  def test_box_prediction(self):

    class_pred = np.zeros((3, 128, 128, 5), dtype=np.float32)
    hw_pred = np.zeros((3, 128, 128, 2), dtype=np.float32)
    offset_pred = np.zeros((3, 128, 128, 2), dtype=np.float32)

    # Sample 1, 2 boxes
    class_pred[0, 10, 20] = [0.3, .7, 0.0, 0.0, 0.0]
    hw_pred[0, 10, 20] = [40, 60]
    offset_pred[0, 10, 20] = [1, 2]

    class_pred[0, 50, 60] = [0.55, 0.0, 0.0, 0.0, 0.45]
    hw_pred[0, 50, 60] = [50, 50]
    offset_pred[0, 50, 60] = [0, 0]

    # Sample 2, 2 boxes (at same location)
    class_pred[1, 100, 100] = [0.0, 0.1, 0.9, 0.0, 0.0]
    hw_pred[1, 100, 100] = [10, 10]
    offset_pred[1, 100, 100] = [1, 3]

    # Sample 3, 3 boxes
    class_pred[2, 60, 90] = [0.0, 0.0, 0.0, 0.2, 0.8]
    hw_pred[2, 60, 90] = [40, 30]
    offset_pred[2, 60, 90] = [0, 0]

    class_pred[2, 65, 95] = [0.0, 0.7, 0.3, 0.0, 0.0]
    hw_pred[2, 65, 95] = [20, 20]
    offset_pred[2, 65, 95] = [1, 2]

    class_pred[2, 75, 85] = [1.0, 0.0, 0.0, 0.0, 0.0]
    hw_pred[2, 75, 85] = [21, 25]
    offset_pred[2, 75, 85] = [5, 2]

    class_pred = tf.constant(class_pred)
    hw_pred = tf.constant(hw_pred)
    offset_pred = tf.constant(offset_pred)

    boxes, classes, scores, num_dets = center_net_meta_arch.\
        prediction_tensors_to_boxes(
            class_pred, hw_pred, offset_pred, num_boxes=2)
    np.testing.assert_array_equal(num_dets, [2, 2, 2])

    np.testing.assert_allclose(
        [[-9, -8, 31, 52], [25, 35, 75, 85]], boxes[0])
    np.testing.assert_allclose(
        [[96, 98, 106, 108], [96, 98, 106, 108]], boxes[1])
    np.testing.assert_allclose(
        [[69.5, 74.5, 90.5, 99.5], [40, 75, 80, 105]], boxes[2])

    np.testing.assert_array_equal(classes[0], [1, 0])
    np.testing.assert_array_equal(classes[1], [2, 1])
    np.testing.assert_array_equal(classes[2], [0, 4])

    np.testing.assert_allclose(scores[0], [.7, .55])
    np.testing.assert_allclose(scores[1][:1], [.9])
    np.testing.assert_allclose(scores[2], [1., .8])

  def test_keypoint_candidate_prediction(self):
    keypoint_heatmap_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    keypoint_heatmap_np[0, 0, 0, 0] = 1.0
    keypoint_heatmap_np[0, 2, 1, 0] = 0.7
    keypoint_heatmap_np[0, 1, 1, 0] = 0.6
    keypoint_heatmap_np[0, 0, 2, 1] = 0.7
    keypoint_heatmap_np[0, 1, 1, 1] = 0.3  # Filtered by low score.
    keypoint_heatmap_np[0, 2, 2, 1] = 0.2
    keypoint_heatmap_np[1, 1, 0, 0] = 0.6
    keypoint_heatmap_np[1, 2, 1, 0] = 0.5
    keypoint_heatmap_np[1, 0, 0, 0] = 0.4
    keypoint_heatmap_np[1, 0, 0, 1] = 1.0
    keypoint_heatmap_np[1, 0, 1, 1] = 0.9
    keypoint_heatmap_np[1, 2, 0, 1] = 0.8

    keypoint_heatmap_offsets_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    keypoint_heatmap_offsets_np[0, 0, 0] = [0.5, 0.25]
    keypoint_heatmap_offsets_np[0, 2, 1] = [-0.25, 0.5]
    keypoint_heatmap_offsets_np[0, 1, 1] = [0.0, 0.0]
    keypoint_heatmap_offsets_np[0, 0, 2] = [1.0, 0.0]
    keypoint_heatmap_offsets_np[0, 2, 2] = [1.0, 1.0]
    keypoint_heatmap_offsets_np[1, 1, 0] = [0.25, 0.5]
    keypoint_heatmap_offsets_np[1, 2, 1] = [0.5, 0.0]
    keypoint_heatmap_offsets_np[1, 0, 0] = [0.0, -0.5]
    keypoint_heatmap_offsets_np[1, 0, 1] = [0.5, -0.5]
    keypoint_heatmap_offsets_np[1, 2, 0] = [-1.0, -0.5]

    keypoint_heatmap = tf.constant(keypoint_heatmap_np, dtype=tf.float32)
    keypoint_heatmap_offsets = tf.constant(keypoint_heatmap_offsets_np,
                                           dtype=tf.float32)

    keypoint_cands, keypoint_scores, num_keypoint_candidates = (
        center_net_meta_arch.prediction_tensors_to_keypoint_candidates(
            keypoint_heatmap, keypoint_heatmap_offsets,
            keypoint_score_threshold=0.5, max_pool_kernel_size=1,
            max_candidates=2))

    expected_keypoint_candidates = [
        [  # Example 0.
            [[0.5, 0.25], [1.75, 1.5]],  # Keypoint 1.
            [[1.0, 2.0], [1.0, 1.0]],  # Keypoint 2.
        ],
        [  # Example 1.
            [[1.25, 0.5], [2.5, 1.0]],  # Keypoint 1.
            [[0.0, -0.5], [0.5, 0.5]],  # Keypoint 2.
        ],
    ]
    expected_keypoint_scores = [
        [  # Example 0.
            [1.0, 0.7],  # Keypoint 1.
            [0.7, 0.3],  # Keypoint 2.
        ],
        [  # Example 1.
            [0.6, 0.5],  # Keypoint 1.
            [1.0, 0.9],  # Keypoint 2.
        ],
    ]
    expected_num_keypoint_candidates = [
        [2, 1],
        [2, 2]
    ]
    np.testing.assert_allclose(expected_keypoint_candidates, keypoint_cands)
    np.testing.assert_allclose(expected_keypoint_scores, keypoint_scores)
    np.testing.assert_array_equal(expected_num_keypoint_candidates,
                                  num_keypoint_candidates)

  def test_regressed_keypoints_at_object_centers(self):
    batch_size = 2
    num_keypoints = 5
    num_instances = 6
    regressed_keypoint_feature_map_np = np.random.randn(
        batch_size, 10, 10, 2 * num_keypoints).astype(np.float32)
    regressed_keypoint_feature_map = tf.constant(
        regressed_keypoint_feature_map_np, dtype=tf.float32)
    y_indices = np.random.choice(10, (batch_size, num_instances))
    x_indices = np.random.choice(10, (batch_size, num_instances))
    offsets = np.stack([y_indices, x_indices], axis=2).astype(np.float32)

    gathered_regressed_keypoints = (
        center_net_meta_arch.regressed_keypoints_at_object_centers(
            regressed_keypoint_feature_map, y_indices, x_indices))

    expected_gathered_keypoints_0 = regressed_keypoint_feature_map_np[
        0, y_indices[0], x_indices[0], :]
    expected_gathered_keypoints_1 = regressed_keypoint_feature_map_np[
        1, y_indices[1], x_indices[1], :]
    expected_gathered_keypoints = np.stack([
        expected_gathered_keypoints_0,
        expected_gathered_keypoints_1], axis=0)
    expected_gathered_keypoints = np.reshape(
        expected_gathered_keypoints,
        [batch_size, num_instances, num_keypoints, 2])
    expected_gathered_keypoints += np.expand_dims(offsets, axis=2)
    expected_gathered_keypoints = np.reshape(
        expected_gathered_keypoints,
        [batch_size, num_instances, -1])
    np.testing.assert_allclose(expected_gathered_keypoints,
                               gathered_regressed_keypoints)


def build_center_net_meta_arch():

  feature_extractor = center_net_resnet_feature_extractor.\
                       CenterNetResnetFeatureExtractor('resnet_v2_101')
  image_resizer_fn = functools.partial(
      preprocessor.resize_to_range,
      min_dimension=128,
      max_dimension=128,
      pad_to_max_dimesnion=True)
  object_detection_params = center_net_meta_arch.ObjectDetectionParams(
      classification_loss=losses.PenaltyReducedLogisticFocalLoss(
          alpha=1.0, beta=1.0),
      classification_loss_weight=1.0,
      localization_loss=losses.L1LocalizationLoss(),
      offset_loss_weight=1.0,
      scale_loss_weight=0.1,
      min_box_overlap_iou=1.0,
      max_box_predictions=5)
  return center_net_meta_arch.CenterNetMetaArch(
      is_training=True,
      add_summaries=False,
      num_classes=10,
      feature_extractor=feature_extractor,
      image_resizer_fn=image_resizer_fn,
      object_detection_params=object_detection_params)


def _logit(p):
  return np.log(p / (1 - p))


class CenterNetMetaArchTest(tf.test.TestCase):
  """Tests for the CenterNet meta architecture."""

  def setUp(self):

    self._boxes = [
        tf.constant([[0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 0.5, 0.5]]),
        tf.constant([[0.5, 0.5, 1.0, 1.0],
                     [0.0, 0.5, 1.0, 1.0]]),
    ]

    self._classes = [
        tf.one_hot([0, 1], depth=10),
        tf.one_hot([2, 3], depth=10),
    ]

    self._weights = [
        tf.constant([1., 0.]),
        tf.constant([1., 1.]),
    ]

    super(CenterNetMetaArchTest, self).setUp()

  def test_compute_class_center_targets(self):
    """Test computation of class center targets."""

    model = build_center_net_meta_arch()
    model.provide_groundtruth(groundtruth_boxes_list=self._boxes,
                              groundtruth_weights_list=self._weights,
                              groundtruth_classes_list=self._classes)
    targets = model._compute_class_center_targets(128, 128)
    targets = targets.numpy()

    self.assertEqual(targets.shape, (2, 32, 32, 10))
    self.assertAlmostEqual(targets[0, 16, 16, 0], 1.0)
    self.assertAlmostEqual(targets[1, 24, 24, 2], 1.0)
    self.assertAlmostEqual(targets[1, 16, 24, 3], 1.0)
    self.assertAlmostEqual(targets.sum(), 3.0)

  def test_compute_height_width_targets(self):
    """Test the computation of height-width targets."""

    model = build_center_net_meta_arch()
    model.provide_groundtruth(groundtruth_boxes_list=self._boxes,
                              groundtruth_weights_list=self._weights,
                              groundtruth_classes_list=self._classes)
    (batch_indices, height_width_targets,
     _, batch_weights) = model._compute_size_and_offset_targets(128, 128)

    self.assertAllEqual(batch_indices, [[0, 16, 16],
                                        [0, 8, 8],
                                        [1, 24, 24],
                                        [1, 16, 24],])

    self.assertAllEqual(height_width_targets, [[32, 32],
                                               [16, 16],
                                               [16, 16],
                                               [32, 16]])
    self.assertAllClose(batch_weights, [1.0, 0.0, 1.0, 1.0])

  def test_prediction_heads(self):
    """Test creation of prediction heads."""

    model = build_center_net_meta_arch()
    model._make_prediction_heads(10, 2,
                                 class_prediction_bias_init=0.0)

    input_tensor = tf.zeros([4, 32, 32, 16], dtype=tf.float32)

    self.assertEqual(model._class_center_heads[0](input_tensor).shape,
                     [4, 32, 32, 10])
    self.assertEqual(model._class_center_heads[1](input_tensor).shape,
                     [4, 32, 32, 10])

    self.assertEqual(model._height_width_heads[0](input_tensor).shape,
                     [4, 32, 32, 2])
    self.assertEqual(model._height_width_heads[1](input_tensor).shape,
                     [4, 32, 32, 2])

    self.assertEqual(model._offset_heads[0](input_tensor).shape,
                     [4, 32, 32, 2])
    self.assertEqual(model._offset_heads[1](input_tensor).shape,
                     [4, 32, 32, 2])

  def test_compute_offset_targets(self):
    """Test the computation of offset targets."""

    boxes = [
        tf.constant([[0.0 + (2.0)/128, 0.0, 1.0, 1.0],
                     [0.0 + (4.0)/128, 0.0, 0.5, 0.5]]),
        tf.constant([[0.5 + (6.0)/128, 0.5, 1.0, 1.0],
                     [0.0 + (8.0)/128, 0.5, 1.0, 1.0]]),
    ]

    model = build_center_net_meta_arch()
    model.provide_groundtruth(groundtruth_boxes_list=boxes,
                              groundtruth_weights_list=self._weights,
                              groundtruth_classes_list=self._classes)
    (_, _, batch_offsets, _) = model._compute_size_and_offset_targets(128, 128)

    self.assertAllEqual(batch_offsets, [[1.0/4, 0],
                                        [2.0/4, 0],
                                        [3.0/4, 0],
                                        [0.0, 0]])

  def test_predict(self):
    """Test the predict function."""

    model = build_center_net_meta_arch()
    prediction_dict = model.predict(tf.zeros([2, 128, 128, 3]), None)

    self.assertEqual(prediction_dict['preprocessed_inputs'].shape,
                     (2, 128, 128, 3))
    self.assertEqual(prediction_dict['class_center_logits'][0].shape,
                     (2, 32, 32, 10))
    self.assertEqual(prediction_dict['height_width'][0].shape,
                     (2, 32, 32, 2))
    self.assertEqual(prediction_dict['offset'][0].shape,
                     (2, 32, 32, 2))

  def test_compute_class_center_loss(self):
    """Test computation of the center heatmap loss."""

    model = build_center_net_meta_arch()
    predictions = [np.ones([2, 32, 32, 1]), np.ones([2, 32, 32, 1])]
    targets = np.zeros([2, 32, 32, 1])
    anchor_weights = np.zeros([2, 1024, 1])
    anchor_weights[0, 0, 0] = 1.0

    targets[0, 0, 0, 0] = 1.0
    predictions[0][0, 0, 0, 0] = _logit(0.5)
    predictions[1][0, 0, 0, 0] = _logit(0.25)

    targets = tf.constant(targets)
    predictions = [tf.constant(prediction) for prediction in predictions]
    anchor_weights = tf.constant(anchor_weights)

    loss = model._compute_class_center_loss(
        predictions, targets, anchor_weights, num_boxes=1.0)
    self.assertAlmostEqual(loss.numpy(),
                           -(0.5 * np.log(0.5) + 0.75 * np.log(0.25)) / 2)

  def test_compute_height_width_loss(self):
    """Test computation of the height-width loss."""

    model = build_center_net_meta_arch()
    predictions = [np.ones([1, 32, 32, 2]), np.ones([1, 32, 32, 2])]
    predictions[0][0, 10, 20] = 21, 32
    predictions[1][0, 10, 20] = 23, 34
    predictions = [tf.constant(prediction) for prediction in predictions]

    batch_indices = tf.constant([
        [0, 10, 20],
    ])
    batch_weights = tf.constant(
        [1.,]
    )
    height_width_targets = tf.constant([
        [20., 30.],
    ])

    loss = model._compute_height_width_loss(batch_indices, batch_weights,
                                            predictions, height_width_targets,
                                            num_boxes=2.0)
    self.assertAlmostEqual(loss.numpy(), 0.1 * (10.0) / 4.0)

  def test_compute_offset_loss(self):
    """Test computation of the offset loss."""

    model = build_center_net_meta_arch()
    predictions = [np.ones([1, 32, 32, 2]), np.ones([1, 32, 32, 2])]
    predictions[0][0, 10, 20] = 0.101, 0.202
    predictions[1][0, 10, 20] = 0.103, 0.204
    predictions = [tf.constant(prediction) for prediction in predictions]

    batch_indices = tf.constant([
        [0, 10, 20],
    ])
    batch_weights = tf.constant(
        [1.,]
    )
    offset_targets = tf.constant([
        [0.1, 0.2],
    ])
    loss = model._compute_height_width_loss(batch_indices, batch_weights,
                                            predictions, offset_targets,
                                            num_boxes=2.0)
    self.assertAlmostEqual(loss.numpy(), 0.1 * (0.01) / 4.0)

  def test_loss(self):
    """Test the loss function."""

    model = build_center_net_meta_arch()

    boxes = [
        tf.constant([[0.0, 0.0, 1.0, 1.0]]),
    ]
    classes = [
        tf.one_hot([1], depth=1),
    ]
    weights = [
        tf.constant([1.]),
    ]

    # Test loss value with a [1, 1] image
    class_center = np.zeros((1, 1, 1, 1), dtype=np.float32)
    class_center[0, 0, 0, 0] = _logit(0.5)
    class_center = tf.constant(class_center)

    height_width = np.zeros((1, 1, 1, 2), dtype=np.float32)
    height_width[0, 0, 0] = 2, 3
    height_width = tf.constant(height_width)

    offset = np.zeros((1, 1, 1, 2), dtype=np.float32)
    offset[0, 0, 0] = 0.6, 0.8
    offset = tf.constant(offset)

    prediction_dict = {
        'class_center_logits': [class_center],
        'height_width': [height_width],
        'offset': [offset],
        'preprocessed_inputs': tf.zeros((1, 4, 4, 3))
    }
    model.provide_groundtruth(groundtruth_boxes_list=boxes,
                              groundtruth_weights_list=weights,
                              groundtruth_classes_list=classes)

    loss_dict = model.loss(prediction_dict, tf.constant([[4, 4, 3]]))
    self.assertAlmostEqual(loss_dict['class_center_loss'].numpy(),
                           -0.5 * np.log(0.5))
    self.assertAlmostEqual(loss_dict['height_width_loss'].numpy(), 0.3)
    self.assertAlmostEqual(loss_dict['offset_loss'].numpy(), 0.4)

  def test_postprocess(self):
    """Test the postprocess function."""

    model = build_center_net_meta_arch()

    class_center = np.zeros((1, 32, 32, 10), dtype=np.float32)
    height_width = np.zeros((1, 32, 32, 2), dtype=np.float32)
    offset = np.zeros((1, 32, 32, 2), dtype=np.float32)

    class_probs = np.zeros(10)
    class_probs[1] = _logit(0.75)
    class_center[0, 16, 16] = class_probs
    height_width[0, 16, 16] = [5, 10]
    offset[0, 16, 16] = [.25, .5]

    class_center = tf.constant(class_center)
    height_width = tf.constant(height_width)
    offset = tf.constant(offset)

    prediction_dict = {
        'class_center_logits': [class_center],
        'height_width': [height_width],
        'offset': [offset]
    }

    detections = model.postprocess(prediction_dict,
                                   tf.constant([[128, 128, 3]]))
    self.assertAllClose(detections['detection_boxes'][0, 0],
                        np.array([55, 46, 75, 86]) / 128.0)
    self.assertAllClose(detections['detection_scores'][0],
                        [.75, .5, .5, .5, .5])
    self.assertEqual(detections['detection_classes'][0, 0], 1)
    self.assertEqual(detections['num_detections'], [5])

  def test_restore_map_resnet(self):
    """Test restore map for a resnet backbone."""

    model = build_center_net_meta_arch()
    restore_map = model.restore_map('classification')
    self.assertIsInstance(restore_map['feature_extractor'], tf.keras.Model)


class DummyFeatureExtractor(center_net_meta_arch.CenterNetFeatureExtractor):

  def __init__(self, channel_means, channel_stds, bgr_ordering):
    super(DummyFeatureExtractor, self).__init__(
        channel_means=channel_means, channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)

  def predict(self):
    pass

  def loss(self):
    pass

  def postprocess(self):
    pass

  def restore_map(self):
    pass


class CenterNetFeatureExtractorTest(tf.test.TestCase):
  """Test the base feature extractor class."""

  def test_preprocess(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(1.0, 2.0, 3.0),
        channel_stds=(10., 20., 30.), bgr_ordering=False)

    img = np.zeros((2, 32, 32, 3))
    img[:, :, :] = 11, 22, 33
    output = feature_extractor.preprocess(img)

    self.assertAlmostEqual(output.numpy().sum(), 2 * 32 * 32 * 3)

  def test_bgr_ordering(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(0.0, 0.0, 0.0),
        channel_stds=(1., 1., 1.), bgr_ordering=True)

    img = np.zeros((2, 32, 32, 3), dtype=np.float32)
    img[:, :, :] = 1, 2, 3
    output = feature_extractor.preprocess(img)
    output = output.numpy()

    self.assertAllClose(output[..., 2], 1 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 1], 2 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 0], 3 * np.ones((2, 32, 32)))

  def test_default_ordering(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(0.0, 0.0, 0.0),
        channel_stds=(1., 1., 1.), bgr_ordering=False)

    img = np.zeros((2, 32, 32, 3), dtype=np.float32)
    img[:, :, :] = 1, 2, 3
    output = feature_extractor.preprocess(img)
    output = output.numpy()

    self.assertAllClose(output[..., 0], 1 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 1], 2 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 2], 3 * np.ones((2, 32, 32)))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
