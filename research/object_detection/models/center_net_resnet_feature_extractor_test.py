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
"""Testing ResNet v2 models for the CenterNet meta architecture."""

import numpy as np
import tensorflow as tf

from object_detection.models import center_net_resnet_feature_extractor


class CenterNetResnetFeatureExtractorTest(tf.test.TestCase):

  def test_output_size(self):
    """Verify that shape of features returned by the backbone is correct."""

    model = center_net_resnet_feature_extractor.\
                CenterNetResnetFeatureExtractor('resnet_v2_101')
    img = np.zeros((8, 224, 224, 3), dtype=np.float32)
    processed_img = model.preprocess(img)

    self.assertEqual(model(processed_img)[0].shape, (8, 56, 56, 64))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
