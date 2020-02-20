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

"""Test for losses designed to be used in v2 mode."""

from __future__ import division
import numpy as np
import tensorflow as tf

from object_detection.core import losses

LOG_2 = np.log(2)
LOG_3 = np.log(3)


class PenaltyReducedLogisticFocalLossTest(tf.test.TestCase):
  """Testing loss function from Equation (1) in [1].

  [1]: https://arxiv.org/abs/1904.07850
  """

  def setUp(self):

    super(PenaltyReducedLogisticFocalLossTest, self).setUp()

    self._prediction = tf.constant([
        # First batch
        [[1 / 2, 1 / 4, 3 / 4],
         [3 / 4, 1 / 3, 1 / 3]],
        # Second Batch
        [[0.0, 1.0, 1 / 2],
         [3 / 4, 2 / 3, 1 / 3]]])
    self._prediction = tf.log(self._prediction/(1 - self._prediction))

    self._target = tf.constant([
        # First batch
        [[1.0, 0.91, 1.0],
         [0.36, 0.84, 1.0]],
        # Second Batch
        [[0.01, 1.0, 0.75],
         [0.96, 1.0, 1.0]]])

  def test_returns_correct_loss(self):

    weights = tf.constant([
        [[1.0], [1.0]],
        [[1.0], [1.0]],
    ])

    loss = losses.PenaltyReducedLogisticFocalLoss(alpha=2.0, beta=0.5)

    computed_value = loss._compute_loss(self._prediction, self._target,
                                        weights)
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.3 * 0.0625 * (2 * LOG_2 - LOG_3),
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.8 * 9 / 16 * 2 * LOG_2,
          0.4 * 1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.2 * 9 / 16 * 2 * LOG_2,
          1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]]])

    np.testing.assert_allclose(computed_value.numpy(), expected_value,
                               rtol=1e-3, atol=1e-3)

  def test_returns_correct_loss_weighted(self):

    weights = tf.constant([
        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
    ])

    loss = losses.PenaltyReducedLogisticFocalLoss(alpha=2.0, beta=0.5)

    computed_value = loss._compute_loss(self._prediction, self._target,
                                        weights)
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.0,
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.0,
          0.0,
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.0,
          0.0,
          0.0]]])

    np.testing.assert_allclose(computed_value.numpy(), expected_value,
                               rtol=1e-3, atol=1e-3)


class L1LocalizationLossTest(tf.test.TestCase):

  def test_returns_correct_loss(self):

    loss = losses.L1LocalizationLoss()

    pred = [[0.1, 0.2], [0.7, 0.5]]
    target = [[0.9, 1.0], [0.1, 0.4]]

    weights = [[1.0, 0.0], [1.0, 1.0]]

    np.testing.assert_allclose(loss._compute_loss(pred, target, weights),
                               [[0.8, 0.0], [0.6, 0.1]], rtol=1e-6)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
