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
"""Tests for object_detection.core.preprocessor in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from object_detection.core import preprocessor


class PreprocessorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scale_1', 1.0),
      ('scale_1.5', 1.5),
      ('scale_0.5', 0.5)
  )
  def test_square_crop_by_scale(self, scale):

    image = np.random.randn(256, 256, 1)

    masks = tf.constant(image[:, :, 0].reshape(1, 256, 256))
    keypoints = tf.constant([[[0.25, 0.25], [0.75, 0.75]]])

    boxes = tf.constant([[0.25, .25, .75, .75]])
    labels = tf.constant([[1]])
    label_weights = tf.constant([[1.]])

    (image, boxes, _, _,
     masks, keypoints) = preprocessor.random_square_crop_by_scale(
         image, boxes, labels, label_weights, masks=masks, keypoints=keypoints,
         max_border=256, scale_min=scale, scale_max=scale)

    ymin, xmin, ymax, xmax = boxes[0].numpy()
    self.assertAlmostEqual(ymax - ymin, 0.5 / scale)
    self.assertAlmostEqual(xmax - xmin, 0.5 / scale)

    k1 = keypoints[0, 0].numpy()
    k2 = keypoints[0, 1].numpy()
    self.assertAlmostEqual(k2[0] - k1[0], 0.5 / scale)
    self.assertAlmostEqual(k2[1] - k1[1], 0.5 / scale)

    size = tf.reduce_max(image.shape).numpy()
    self.assertAlmostEqual(scale * 256.0, size)

    self.assertAllClose(image.numpy()[:, :, 0],
                        masks.numpy()[0, :, :])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
