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

"""Tests for target assigners designed to be used in v2 mode."""

import numpy as np
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import target_assigner
from object_detection.utils import np_box_ops


def _array_argmax(array):
  return np.unravel_index(np.argmax(array), array.shape)


class CenterNetTargetAssignerTest(tf.test.TestCase):

  def setUp(self):

    super(CenterNetTargetAssignerTest, self).setUp()

    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]

  def test_center_location(self):
    """Test that the centers are at the correct location."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_lower_left]))
    ]

    classes = [
        tf.one_hot([0, 1], depth=4),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    targets = assigner.assign_center_targets(80, 80, box_batch, classes)

    self.assertEqual((10, 10), _array_argmax(targets[0, :, :, 0]))
    self.assertAlmostEqual(1.0, targets[0, 10, 10, 0].numpy())

    self.assertEqual((15, 5), _array_argmax(targets[0, :, :, 1]))
    self.assertAlmostEqual(1.0, targets[0, 15, 5, 1].numpy())

  def test_center_batch_shape(self):
    """Test that the shape of the target for a batch is correct."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_lower_left])),
        box_list.BoxList(tf.constant([self._box_center])),
        box_list.BoxList(tf.constant([self._box_center_small])),
    ]
    classes = [
        tf.one_hot([0, 1], depth=4),
        tf.one_hot([2], depth=4),
        tf.one_hot([3], depth=4),
    ]
    assigner = target_assigner.CenterNetTargetAssigner(4)
    targets = assigner.assign_center_targets(80, 80, box_batch, classes)
    self.assertEqual((3, 20, 20, 4), targets.shape)

  def test_center_overlap_maximum(self):
    """Test that when boxes overlap we, are computing the maximum."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_center_offset,
                                      self._box_center,
                                      self._box_center_offset]))
    ]
    classes = [
        tf.one_hot([0, 0, 1, 2], depth=4),
    ]
    assigner = target_assigner.CenterNetTargetAssigner(4)
    targets = assigner.assign_center_targets(80, 80, box_batch, classes)

    class0_targets = targets[0, :, :, 0]
    class1_targets = targets[0, :, :, 1]
    class2_targets = targets[0, :, :, 2]

    np.testing.assert_allclose(class0_targets,
                               np.maximum(class1_targets, class2_targets))

  def test_size_blur(self):
    """Test that the heatmap of a larger box is more blurred."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center,
                                      self._box_center_small]))
    ]

    classes = [
        tf.one_hot([0, 1], depth=4),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    targets = assigner.assign_center_targets(80, 80, box_batch, classes)
    targets = targets.numpy()

    self.assertGreater(np.count_nonzero(targets[:, :, :, 0]),
                       np.count_nonzero(targets[:, :, :, 1]))

  def test_weights(self):
    """Test that the weights correctly ignore ground truth."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_lower_left])),
        box_list.BoxList(tf.constant([self._box_center])),
        box_list.BoxList(tf.constant([self._box_center_small])),
    ]
    classes = [
        tf.one_hot([0, 1], depth=4),
        tf.one_hot([2], depth=4),
        tf.one_hot([3], depth=4),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    targets = assigner.assign_center_targets(80, 80, box_batch,
                                             classes).numpy()

    self.assertAlmostEqual(1.0, targets[0, :, :, 0].max())
    self.assertAlmostEqual(1.0, targets[0, :, :, 1].max())
    self.assertAlmostEqual(1.0, targets[1, :, :, 2].max())
    self.assertAlmostEqual(1.0, targets[2, :, :, 3].max())

    self.assertAlmostEqual(0.0, targets[0, :, :, [2, 3]].max())
    self.assertAlmostEqual(0.0, targets[1, :, :, [0, 1, 3]].max())
    self.assertAlmostEqual(0.0, targets[2, :, :, :3].max())

    weights = [
        tf.constant([0., 1.]),
        tf.constant([1.]),
        tf.constant([1.]),
    ]

    targets = assigner.assign_center_targets(80, 80, box_batch,
                                             classes, weights).numpy()

    self.assertAlmostEqual(1.0, targets[0, :, :, 1].max())
    self.assertAlmostEqual(1.0, targets[1, :, :, 2].max())
    self.assertAlmostEqual(1.0, targets[2, :, :, 3].max())

    self.assertAlmostEqual(0.0, targets[0, :, :, [0, 2, 3]].max())
    self.assertAlmostEqual(0.0, targets[1, :, :, [0, 1, 3]].max())
    self.assertAlmostEqual(0.0, targets[2, :, :, :3].max())

  def test_overlap(self):
    """Test the effect of the overlap parameter."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center]))
    ]

    classes = [
        tf.one_hot([0], depth=2),
    ]

    assigner_low_overlap = target_assigner.CenterNetTargetAssigner(
        4, min_overlap=0.1)

    targets_low_overlap = assigner_low_overlap.assign_center_targets(
        80, 80, box_batch, classes).numpy()
    self.assertLess(1, np.count_nonzero(targets_low_overlap))

    assigner_medium_overlap = target_assigner.CenterNetTargetAssigner(
        4, min_overlap=0.6)

    targets_medium_overlap = assigner_medium_overlap.assign_center_targets(
        80, 80, box_batch, classes).numpy()
    self.assertLess(1, np.count_nonzero(targets_medium_overlap))

    assigner_high_overlap = target_assigner.CenterNetTargetAssigner(
        4, min_overlap=0.99)

    targets_high_overlap = assigner_high_overlap.assign_center_targets(
        80, 80, box_batch, classes).numpy()

    self.assertTrue(np.all(targets_low_overlap >= targets_medium_overlap))
    self.assertTrue(np.all(targets_medium_overlap >= targets_high_overlap))

  def test_max_distance_for_overlap(self):
    """Test that the distance ensures the IoU with random boxes."""

    # TODO(vighneshb) remove this after the `_smallest_positive_root`
    # function if fixed.
    self.skipTest(('Skipping test because we are using an incorrect version of'
                   'the `max_distance_for_overlap` function to reproduce'
                   ' results.'))

    rng = np.random.RandomState(0)
    n_samples = 100

    width = rng.uniform(1, 100, size=n_samples)
    height = rng.uniform(1, 100, size=n_samples)
    min_iou = rng.uniform(0.1, 1.0, size=n_samples)

    max_dist = target_assigner.max_distance_for_overlap(
        height, width, min_iou)

    xmin1 = np.zeros(n_samples)
    ymin1 = np.zeros(n_samples)
    xmax1 = np.zeros(n_samples) + width
    ymax1 = np.zeros(n_samples) + height

    xmin2 = max_dist*np.cos(rng.uniform(0, 2*np.pi))
    ymin2 = max_dist*np.sin(rng.uniform(0, 2*np.pi))
    xmax2 = width + max_dist*np.cos(rng.uniform(0, 2*np.pi))
    ymax2 = height + max_dist*np.sin(rng.uniform(0, 2*np.pi))

    boxes1 = np.vstack([ymin1, xmin1, ymax1, xmax1]).T
    boxes2 = np.vstack([ymin2, xmin2, ymax2, xmax2]).T

    iou = np.diag(np_box_ops.iou(boxes1, boxes2))

    self.assertTrue(np.all(iou >= min_iou))

  def test_max_distance_for_overlap_centernet(self):
    """Test the version of the function used in the CenterNet paper."""

    distance = target_assigner.max_distance_for_overlap(10, 5, 0.5)
    self.assertAlmostEqual(2.807764064, distance.numpy())

  def test_assign_size_and_offset_targets(self):
    """Test the assign_size_and_offset_targets function."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_lower_left])),
        box_list.BoxList(tf.constant([self._box_center_offset])),
        box_list.BoxList(tf.constant([self._box_center_small,
                                      self._box_odd_coordinates])),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    indices, hw, yx_offset, weights = assigner.assign_size_and_offset_targets(
        80, 80, box_batch)

    self.assertEqual(indices.shape, (5, 3))
    self.assertEqual(hw.shape, (5, 2))
    self.assertEqual(yx_offset.shape, (5, 2))
    self.assertEqual(weights.shape, (5,))

    np.testing.assert_array_equal(
        indices, [[0, 10, 10], [0, 15, 5], [1, 11, 10],
                  [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_equal(
        hw, [[20, 20], [10, 10], [18, 19], [10, 10], [8, 15]])
    np.testing.assert_array_equal(
        yx_offset, [[0, 0], [0, 0], [0, 0.5], [0, 0], [0.25, 0.75]])
    np.testing.assert_array_equal(weights, 1)

  def test_assign_size_and_offset_targets_weights(self):
    """Test the assign_size_and_offset_targets function with box weights."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center,
                                      self._box_lower_left])),
        box_list.BoxList(tf.constant([self._box_lower_left,
                                      self._box_center_small])),
        box_list.BoxList(tf.constant([self._box_center_small,
                                      self._box_odd_coordinates])),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    weights_batch = [tf.constant([0.0, 1.0]), tf.constant([1.0, 1.0]),
                     tf.constant([0.0, 0.0])]
    indices, hw, yx_offset, weights = assigner.assign_size_and_offset_targets(
        80, 80, box_batch, weights_batch)

    self.assertEqual(indices.shape, (6, 3))
    self.assertEqual(hw.shape, (6, 2))
    self.assertEqual(yx_offset.shape, (6, 2))
    self.assertEqual(weights.shape, (6,))

    np.testing.assert_array_equal(
        indices, [[0, 10, 10], [0, 15, 5], [1, 15, 5],
                  [1, 10, 10], [2, 10, 10], [2, 7, 11]])
    np.testing.assert_array_equal(
        hw, [[20, 20], [10, 10], [10, 10], [10, 10], [10, 10], [8, 15]])
    np.testing.assert_array_equal(
        yx_offset, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.25, 0.75]])
    np.testing.assert_array_equal(weights, [0, 1, 1, 1, 0, 0])

  def test_get_batch_predictions_from_indices(self):
    """Test the get_batch_predictions_from_indices function.

    This test verifies that the indices returned by
    assign_size_and_offset_targets function work as expected with a predicted
    tensor.

    """

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center, self._box_lower_left])),
        box_list.BoxList(tf.constant([self._box_center_small,
                                      self._box_odd_coordinates])),
    ]

    pred_array = np.ones((2, 40, 20, 2), dtype=np.int32) * -1000
    pred_array[0, 20, 10] = [1, 2]
    pred_array[0, 30, 5] = [3, 4]
    pred_array[1, 20, 10] = [5, 6]
    pred_array[1, 14, 11] = [7, 8]

    pred_tensor = tf.constant(pred_array)

    assigner = target_assigner.CenterNetTargetAssigner(4)
    indices, _, _, _ = assigner.assign_size_and_offset_targets(
        160, 80, box_batch)

    preds = target_assigner.get_batch_predictions_from_indices(
        pred_tensor, indices)
    np.testing.assert_array_equal(preds, [[1, 2], [3, 4], [5, 6], [7, 8]])

  def test_empty_box_list(self):
    """Test that an empty box list gives an all 0 heatmap."""
    box_batch = [
        box_list.BoxList(tf.zeros((0, 4), dtype=tf.float32)),
    ]

    classes = [
        tf.zeros((0, 5), dtype=tf.float32),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(
        4, min_overlap=0.1)

    targets = assigner.assign_center_targets(
        80, 80, box_batch, classes).numpy()

    np.testing.assert_allclose(targets, 0.)

  def test_tf_function_compatibility(self):
    """Test that the target assigner works as a tf.function."""

    box_batch = [
        box_list.BoxList(tf.constant([self._box_center,
                                      self._box_center_small]))
    ]

    classes = [
        tf.one_hot([0, 1], depth=4),
    ]

    assigner = target_assigner.CenterNetTargetAssigner(4)
    assign_size_and_offset_targets = tf.function(
        assigner.assign_size_and_offset_targets)
    indices, hw, yx_offset, weights = assign_size_and_offset_targets(
        80, 80, box_batch)

    assign_center_targets = tf.function(
        assigner.assign_center_targets)
    heatmap = assign_center_targets(80, 80, box_batch, classes)

    self.assertEqual(indices.shape, (2, 3))
    self.assertEqual(hw.shape, (2, 2))
    self.assertEqual(yx_offset.shape, (2, 2))
    self.assertEqual(weights.shape, (2,))
    self.assertEqual(heatmap.shape, (1, 20, 20, 4))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
