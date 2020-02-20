# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for object detection model library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib_v2
from object_detection.protos import train_pb2
from object_detection.utils import config_util


# Model for test. Current options are:
# 'ssd_mobilenet_v2_pets_keras'
MODEL_NAME_FOR_TEST = 'ssd_mobilenet_v2_pets_keras'


def _get_data_path():
  """Returns an absolute path to TFRecord file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data',
                      'pets_examples.record')


def get_pipeline_config_path(model_name):
  """Returns path to the local pipeline config file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'samples',
                      'configs', model_name + '.config')


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'data',
                      'pet_label_map.pbtxt')


def _get_config_kwarg_overrides():
  """Returns overrides to the configs that insert the correct local paths."""
  data_path = _get_data_path()
  label_map_path = _get_labelmap_path()
  return {
      'train_input_path': data_path,
      'eval_input_path': data_path,
      'label_map_path': label_map_path
  }


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  filename = get_pipeline_config_path(model_name)
  configs = config_util.get_configs_from_pipeline_file(filename)
  configs = config_util.merge_external_params_with_configs(
      configs, kwargs_dict=_get_config_kwarg_overrides())
  return configs


class ModelLibTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint:disable=g-missing-super-call
    tf.keras.backend.clear_session()

  def test_train_loop_then_eval_loop(self):
    """Tests that Estimator and input function are constructed correctly."""
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    config_kwarg_overrides = _get_config_kwarg_overrides()
    model_dir = tf.test.get_temp_dir()

    train_steps = 2
    model_lib_v2.train_loop(
        hparams,
        pipeline_config_path,
        model_dir=model_dir,
        train_steps=train_steps,
        checkpoint_every_n=1,
        **config_kwarg_overrides)

    model_lib_v2.eval_continuously(
        hparams,
        pipeline_config_path,
        model_dir=model_dir,
        checkpoint_dir=model_dir,
        train_steps=train_steps,
        wait_interval=10,
        **config_kwarg_overrides)


class SimpleModel(tf.keras.Model):
  """A model with a single weight vector."""

  def __init__(self):
    super(SimpleModel, self).__init__(self)
    self.weight = tf.keras.backend.variable(np.ones(10), name='weight')

  def restore_map(self, *args, **kwargs):
    return {'model': self}


class IncompatibleModel(SimpleModel):

  def restore_map(self, *args, **kwargs):
    return {'weight': self.weight}


class CheckpointV2Test(tf.test.TestCase):

  def setUp(self):
    super(CheckpointV2Test, self).setUp()

    self._model = SimpleModel()
    tf.keras.backend.set_value(self._model.weight, np.ones(10) * 42)
    ckpt = tf.train.Checkpoint(model=self._model)

    self._test_dir = tf.test.get_temp_dir()
    self._ckpt_path = ckpt.save(os.path.join(self._test_dir, 'ckpt'))
    tf.keras.backend.set_value(self._model.weight, np.ones(10))

  def test_restore_v2(self):
    """Test that restoring a v2 style checkpoint works."""

    model_lib_v2.load_fine_tune_checkpoint(
        self._model, self._ckpt_path, checkpoint_type='',
        checkpoint_version=train_pb2.CheckpointVersion.V2,
        load_all_detection_checkpoint_vars=True, input_dataset=None,
        unpad_groundtruth_tensors=True)
    np.testing.assert_allclose(self._model.weight.numpy(), 42)

  def test_restore_map_incompatible_error(self):
    """Test that restoring an incompatible restore map causes an error."""

    model = IncompatibleModel()
    with self.assertRaisesRegex(TypeError,
                                r'.*received a \(str -> ResourceVariable\).*'):
      model_lib_v2.load_fine_tune_checkpoint(
          model, self._ckpt_path, checkpoint_type='',
          checkpoint_version=train_pb2.CheckpointVersion.V2,
          load_all_detection_checkpoint_vars=True, input_dataset=None,
          unpad_groundtruth_tensors=True)


