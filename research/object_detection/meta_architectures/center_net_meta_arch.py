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
"""The CenterNet meta architecture as described in the "Objects as Points" paper [1].

[1]: https://arxiv.org/abs/1904.07850
"""

import abc
import collections
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops

from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import shape_utils


# Number of channels needed to predict size and offsets.
NUM_OFFSET_CHANNELS = 2
NUM_SIZE_CHANNELS = 2

# Error range for detecting peaks.
PEAK_EPSILON = 1e-6


class CenterNetFeatureExtractor(tf.keras.Model):
  """Base class for feature extractors for the CenterNet meta architecture.

  Child classes are expected to override the _output_model property which will
  return 1 or more tensors predicted by the feature extractor.

  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, name=None, channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.), bgr_ordering=False):
    """Initializes a CenterNet feature extractor.

    Args:
      name: str, the name used for the underlying keras model.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it. If None or empty, we use 0s.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
        If None or empty, we use 1s.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
    """
    super(CenterNetFeatureExtractor, self).__init__(name=name)

    if channel_means is None or len(channel_means) == 0:  # pylint:disable=g-explicit-length-test
      channel_means = [0., 0., 0.]

    if channel_stds is None or len(channel_stds) == 0:  # pylint:disable=g-explicit-length-test
      channel_stds = [1., 1., 1.]

    self._channel_means = channel_means
    self._channel_stds = channel_stds
    self._bgr_ordering = bgr_ordering

  def preprocess(self, inputs):
    """Converts a batch of unscaled images to a scale suitable for the model.

    This method normalizes the image using the given `channel_means` and
    `channels_stds` values at initialization time while optionally flipping
    the channel order if `bgr_ordering` is set.

    Args:
      inputs: a [batch, height, width, channels] float32 tensor

    Returns:
      outputs: a [batch, height, width, channels] float32 tensor

    """

    if self._bgr_ordering:
      red, green, blue = tf.unstack(inputs, axis=3)
      inputs = tf.stack([blue, green, red], axis=3)

    channel_means = tf.reshape(tf.constant(self._channel_means),
                               [1, 1, 1, -1])
    channel_stds = tf.reshape(tf.constant(self._channel_stds),
                              [1, 1, 1, -1])

    return (inputs - channel_means)/channel_stds

  @property
  @abc.abstractmethod
  def out_stride(self):
    """The stride in the output image of the network."""
    pass

  @property
  @abc.abstractmethod
  def num_feature_outputs(self):
    """Ther number of feature outputs returned by the feature extractor."""
    pass


def make_prediction_net(num_out_channels, kernel_size=3, num_filters=256,
                        bias_fill=None):
  """Creates a network to predict the given number of output channels.

  This function is intended to make the prediction heads for the CenterNet
  meta architecture.

  Args:
    num_out_channels: Number of output channels.
    kernel_size: The size of the conv kernel in the intermediate layer
    num_filters: The number of filters in the intermediate conv layer.
    bias_fill: If not None, is used to initialize the bias in the final conv
      layer.

  Returns:
    net: A keras module which when called on an input tensor of size
      [batch_size, height, width, num_in_channels] returns an output
      of size [batch_size, height, width, num_out_channels]
  """

  out_conv = tf.keras.layers.Conv2D(num_out_channels, kernel_size=1)

  if bias_fill is not None:
    out_conv.bias_initializer = tf.keras.initializers.constant(bias_fill)

  net = tf.keras.Sequential(
      [tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size,
                              padding='same'),
       tf.keras.layers.ReLU(),
       out_conv]
  )

  return net


def _to_float32(x):
  return tf.cast(x, tf.float32)


def _get_shape(tensor, num_dims):
  tf.Assert(tensor.get_shape().ndims == num_dims, [tensor])
  return tf.unstack(tf.shape(tensor), axis=0)[:num_dims]


def _flatten_spatial_dimensions(batch_images):
  batch_size, height, width, channels = _get_shape(batch_images, 4)
  return tf.reshape(batch_images, [batch_size, height * width,
                                   channels])


def top_k_feature_map_locations(feature_map, max_pool_kernel_size=3, k=100,
                                per_channel=False):
  """Returns the top k scores and their locations in a feature map.

  Given a feature map, the top k values (based on activation) are returned. If
  `per_channel` is True, the top k values **per channel** are returned.

  The `max_pool_kernel_size` argument allows for selecting local peaks in a
  region. This filtering is done per channel, so nothing prevents two values at
  the same location to be returned.

  Args:
    feature_map: [batch, height, width, channels] float32 feature map.
    max_pool_kernel_size: integer, the max pool kernel size to use to pull off
      peak score locations in a neighborhood (independently for each channel).
      For example, to make sure no two neighboring values (in the same channel)
      are returned, set max_pool_kernel_size=3. If None or 1, will not apply max
      pooling.
    k: The number of highest scoring locations to return.
    per_channel: If True, will return the top k scores and locations per
      feature map channel. If False, the top k across the entire feature map
      (height x width x channels) are returned.

  Returns:
    Tuple of
    scores: A [batch, N] float32 tensor with scores from the feature map in
      descending order. If per_channel is False, N = k. Otherwise,
      N = k * channels, and the first k elements correspond to channel 0, the
      second k correspond to channel 1, etc.
    y_indices: A [batch, N] int tensor with y indices of the top k feature map
      locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
    x_indices: A [batch, N] int tensor with x indices of the top k feature map
      locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
    channel_indices: A [batch, N] int tensor with channel indices of the top k
      feature map locations. If per_channel is False, N = k. Otherwise,
      N = k * channels.
  """
  if not max_pool_kernel_size or max_pool_kernel_size == 1:
    feature_map_peaks = feature_map
  else:
    feature_map_max_pool = tf.nn.max_pool(
        feature_map, ksize=max_pool_kernel_size, strides=1, padding='SAME')

    feature_map_peak_mask = tf.math.abs(
        feature_map - feature_map_max_pool) < PEAK_EPSILON

    # Zero out everything that is not a peak.
    feature_map_peaks = (
        feature_map * _to_float32(feature_map_peak_mask))

  batch_size, _, width, num_channels = _get_shape(feature_map, 4)

  if per_channel:
    single_channel_feat_map_list = tf.unstack(feature_map, axis=3)
    scores_per_channel_list = []
    peak_indices_per_channel_list = []
    for channel_idx, single_channel_feat_map in enumerate(
        single_channel_feat_map_list):
      single_channel_peaks_flat = tf.reshape(
          single_channel_feat_map, [batch_size, -1])
      scores, peak_flat_indices = tf.math.top_k(single_channel_peaks_flat, k=k)
      # Convert the indices such that they represent the location in the full
      # (flattened) feature map of size [batch, height * width * channels].
      peak_flat_indices = num_channels * peak_flat_indices + channel_idx
      scores_per_channel_list.append(scores)
      peak_indices_per_channel_list.append(peak_flat_indices)
    scores = tf.concat(scores_per_channel_list, axis=-1)
    peak_flat_indices = tf.concat(peak_indices_per_channel_list, axis=-1)
  else:
    feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
    scores, peak_flat_indices = tf.math.top_k(feature_map_peaks_flat, k=k)

  # Get x, y and channel indices corresponding to the top indices in the flat
  # array.
  y_indices, x_indices, channel_indices = (
      row_col_channel_indices_from_flattened_indices(
          peak_flat_indices, width, num_channels))
  return scores, y_indices, x_indices, channel_indices


def prediction_tensors_to_boxes(class_center_predictions,
                                height_width_predictions,
                                offset_predictions,
                                num_boxes=100):
  """Converts CenterNet class-center, offset and size predictions to boxes.

  Args:
    class_center_predictions: A float tensor of shape [batch_size, height,
      width, num_classes] representing the per-class heatmaps of bounding box
      centers.
    height_width_predictions: A float tensor of shape [batch_size, height,
      width, 2] representing the height and width of a box centered at each
      pixel.
    offset_predictions: A float tensor of shape [batch_size, height, width, 2]
      representing the y and x offsets of a box centered at each pixel. This
      helps reduce the error from downsampling.
    num_boxes: integer, maximum number of boxes to output.

  Returns:
    detection_boxes: A tensor of shape [batch_size, num_boxes, 4] holding the
      the raw bounding box coordinates of boxes.
    detection_classes: An integer tensor of shape [batch_size, num_boxes]
      indicating the predicted class for each box.
    detection_scores: A float tensor of shape [batch_size, num_boxes] indicating
      the score for each box.
    num_detections: An integer tensor of shape [batch_size,] indicating the
      number of boxes detected for each sample in the batch.

  """
  _, _, width, _ = _get_shape(class_center_predictions, 4)
  # Get x, y and channel indices corresponding to the top indices in the class
  # center predictions.
  detection_scores, y_indices, x_indices, channel_indices = (
      top_k_feature_map_locations(class_center_predictions,
                                  max_pool_kernel_size=3, k=num_boxes))

  peak_spatial_indices = flattened_indices_from_row_col_indices(
      y_indices, x_indices, width)
  y_indices = _to_float32(y_indices)
  x_indices = _to_float32(x_indices)

  height_width_flat = _flatten_spatial_dimensions(height_width_predictions)
  offsets_flat = _flatten_spatial_dimensions(offset_predictions)

  height_width = tf.gather(height_width_flat, peak_spatial_indices,
                           batch_dims=1)
  offsets = tf.gather(offsets_flat, peak_spatial_indices, batch_dims=1)

  heights, widths = tf.unstack(height_width, axis=2)
  y_offsets, x_offsets = tf.unstack(offsets, axis=2)

  detection_classes = channel_indices

  num_detections = tf.reduce_sum(tf.to_int32(detection_scores > 0), axis=1)

  boxes = tf.stack([y_indices + y_offsets - heights / 2.0,
                    x_indices + x_offsets - widths / 2.0,
                    y_indices + y_offsets + heights / 2.0,
                    x_indices + x_offsets + widths / 2.0], axis=2)

  return boxes, detection_classes, detection_scores, num_detections


def prediction_tensors_to_keypoint_candidates(
    keypoint_heatmap_predictions,
    keypoint_heatmap_offsets,
    keypoint_score_threshold=0.1,
    max_pool_kernel_size=1,
    max_candidates=20):
  """Convert keypoint heatmap predictions and offsets to keypoint candidates.

  Args:
    keypoint_heatmap_predictions: A float tensor of shape [batch_size, height,
      width, num_keypoints] representing the per-keypoint heatmaps.
    keypoint_heatmap_offsets: A float tensor of shape [batch_size, height,
      width, 2] representing the per-keypoint offsets.
    keypoint_score_threshold: float, the threshold for considering a keypoint
      a candidate.
    max_pool_kernel_size: integer, the max pool kernel size to use to pull off
      peak score locations in a neighborhood. For example, to make sure no two
      neighboring values for the same keypoint are returned, set
      max_pool_kernel_size=3. If None or 1, will not apply any local filtering.
    max_candidates: integer, maximum number of keypoint candidates per
      keypoint type.

  Returns:
    keypoint_candidates: A tensor of shape
      [batch_size, num_keypoints, max_candidates, 2] holding the
      location of keypoint candidates in [y, x] format (expressed in absolute
      coordinates in the output coordinate frame).
    keypoint_scores: A float tensor of shape
      [batch_size, num_keypoints, max_candidates] with the scores for each
      keypoint candidate. The scores come directly from the heatmap predictions.
    num_keypoint_candidates: An integer tensor of shape
      [batch_size, num_keypoints] with the number of candidates for each
      keypoint type, as it's possible to filter some candidates due to the score
      threshold.
  """
  batch_size, _, width, num_keypoints = _get_shape(
      keypoint_heatmap_predictions, 4)
  # Get x, y and channel indices corresponding to the top indices in the
  # keypoint heatmap predictions.
  # Note that the top k candidates are produced for **each keypoint type**.
  # Might be worth eventually trying top k in the feature map, independent of
  # the keypoint type.
  keypoint_scores, y_indices, x_indices, _ = (
      top_k_feature_map_locations(keypoint_heatmap_predictions,
                                  max_pool_kernel_size=max_pool_kernel_size,
                                  k=max_candidates,
                                  per_channel=True))

  peak_spatial_indices = flattened_indices_from_row_col_indices(
      y_indices, x_indices, width)
  y_indices = _to_float32(y_indices)
  x_indices = _to_float32(x_indices)

  offsets_flat = _flatten_spatial_dimensions(keypoint_heatmap_offsets)

  offsets = tf.gather(offsets_flat, peak_spatial_indices, batch_dims=1)
  y_offsets, x_offsets = tf.unstack(offsets, axis=2)

  keypoint_candidates = tf.stack([y_indices + y_offsets,
                                  x_indices + x_offsets], axis=2)
  keypoint_candidates = tf.reshape(
      keypoint_candidates,
      [batch_size, num_keypoints, max_candidates, 2])
  keypoint_scores = tf.reshape(
      keypoint_scores,
      [batch_size, num_keypoints, max_candidates])
  num_candidates = tf.reduce_sum(
      tf.to_int32(keypoint_scores >= keypoint_score_threshold), axis=2)

  return keypoint_candidates, keypoint_scores, num_candidates


def regressed_keypoints_at_object_centers(regressed_keypoint_predictions,
                                          y_indices, x_indices):
  """Returns the regressed keypoints at specified object centers.

  The original keypoint predictions are regressed relative to each feature map
  location. The returned keypoints are expressed in absolute coordinates in the
  output frame (i.e. the center offsets are added to each individual regressed
  set of keypoints).

  Args:
    regressed_keypoint_predictions: A float tensor of shape
      [batch_size, height, width, 2 * num_keypoints] holding regressed
      keypoints. The last dimension has keypoint coordinates ordered as follows:
      [y0, x0, y1, x1, ..., y{k-1}, x{k-1}] where k is the number of keypoints.
    y_indices: A [batch, num_instances] int tensor holding y indices for object
      centers. These indices correspond to locations in the output feature map.
    x_indices: A [batch, num_instances] int tensor holding x indices for object
      centers. These indices correspond to locations in the output feature map.

  Returns:
    A float tensor of shape [batch_size, num_objects, 2 * num_keypoints] where
    regressed keypoints are gathered at the provided locations, and converted
    to absolute coordinates in the output coordinate frame.
  """
  batch_size, _, width, _ = _get_shape(regressed_keypoint_predictions, 4)
  flattened_indices = flattened_indices_from_row_col_indices(
      y_indices, x_indices, width)
  num_instances = flattened_indices.shape.as_list()[-1]

  regressed_keypoints_flat = _flatten_spatial_dimensions(
      regressed_keypoint_predictions)

  relative_regressed_keypoints = tf.gather(
      regressed_keypoints_flat, flattened_indices, batch_dims=1)
  relative_regressed_keypoints = tf.reshape(
      relative_regressed_keypoints,
      [batch_size, num_instances, -1, 2])
  relative_regressed_keypoints_y, relative_regressed_keypoints_x = tf.unstack(
      relative_regressed_keypoints, axis=3)
  y_indices = _to_float32(tf.expand_dims(y_indices, axis=-1))
  x_indices = _to_float32(tf.expand_dims(x_indices, axis=-1))
  absolute_regressed_keypoints = tf.stack(
      [y_indices + relative_regressed_keypoints_y,
       x_indices + relative_regressed_keypoints_x],
      axis=3)
  return tf.reshape(absolute_regressed_keypoints,
                    [batch_size, num_instances, -1])


def flattened_indices_from_row_col_indices(row_indices, col_indices, num_cols):
  """Get the index in a flattened array given row and column indices."""
  return (row_indices * num_cols) + col_indices


def row_col_channel_indices_from_flattened_indices(indices, num_cols,
                                                   num_channels):
  """Computes row, column and channel indices from flattened indices.

  Args:
    indices: An integer tensor of any shape holding the indices in the flattened
      space.
    num_cols: Number of columns in the image (width).
    num_channels: Number of channels in the image.

  Returns:
    row_indices: The row indices corresponding to each of the input indices.
      Same shape as indices.
    col_indices: The column indices corresponding to each of the input indices.
      Same shape as indices.
    channel_indices. The channel indices corresponding to each of the input
      indices.

  """
  row_indices = (indices // num_channels) // num_cols
  col_indices = (indices // num_channels) % num_cols
  channel_indices = indices % num_channels

  return row_indices, col_indices, channel_indices


def get_valid_anchor_weights_in_flattened_image(true_image_shapes, height,
                                                width):
  """Computes valid anchor weights for an image assuming pixels will be flattened.

  This function is useful when we only want to penalize valid areas in the
  image in the case when padding is used. The function assumes that the loss
  function will be applied after flattening the spatial dimensions and returns
  anchor weights accordingly.

  Args:
    true_image_shapes: An integer tensor of shape [batch_size, 3] representing
      the true image shape (without padding) for each sample in the batch.
    height: height of the prediction from the network.
    width: width of the prediction from the network.

  Returns:
    valid_anchor_weights: a float tensor of shape [batch_size, height * width]
    with 1s in locations where the spatial coordinates fall within the height
    and width in true_image_shapes.
  """

  indices = tf.reshape(tf.range(height * width), [1, -1])
  batch_size = tf.shape(true_image_shapes)[0]
  batch_indices = tf.ones((batch_size, 1), dtype=tf.int32) * indices

  y_coords, x_coords, _ = row_col_channel_indices_from_flattened_indices(
      batch_indices, width, 1)

  max_y, max_x = true_image_shapes[:, 0], true_image_shapes[:, 1]
  max_x = _to_float32(tf.expand_dims(max_x, 1))
  max_y = _to_float32(tf.expand_dims(max_y, 1))

  x_coords = _to_float32(x_coords)
  y_coords = _to_float32(y_coords)

  valid_mask = tf.math.logical_and(x_coords < max_x, y_coords < max_y)

  return _to_float32(valid_mask)


def convert_strided_predictions_to_normalized_boxes(boxes, stride,
                                                    true_image_shapes):
  """Converts predictions in the output space to normalized boxes.

  Boxes falling outside the valid image boundary are clipped to be on the
  boundary.

  Args:
    boxes: A tensor of shape [batch_size, num_boxes, 4] holding the raw
     coordinates of boxes in the model's output space.
    stride: The stride in the output space.
    true_image_shapes: A tensor of shape [batch_size, 3] representing the true
      shape of the input not considering padding.

  Returns:
    boxes: A tensor of shape [batch_size, num_boxes, 4] representing the
      coordinates of the normalized boxes.
  """

  def _normalize_boxlist(args):

    boxes, height, width = args
    boxes = box_list_ops.scale(boxes, stride, stride)
    boxes = box_list_ops.to_normalized_coordinates(boxes, height, width)
    boxes = box_list_ops.clip_to_window(boxes, [0., 0., 1., 1.],
                                        filter_nonoverlapping=False)
    return boxes

  box_lists = [box_list.BoxList(boxes) for boxes in tf.unstack(boxes, axis=0)]
  true_heights, true_widths, _ = tf.unstack(true_image_shapes, axis=1)

  true_heights_list = tf.unstack(true_heights, axis=0)
  true_widths_list = tf.unstack(true_widths, axis=0)

  box_lists = list(map(_normalize_boxlist,
                       zip(box_lists, true_heights_list, true_widths_list)))
  boxes = tf.stack([box_list_instance.get() for
                    box_list_instance in box_lists], axis=0)

  return boxes


class ObjectDetectionParams(
    collections.namedtuple('ObjectDetectionParams', [
        'classification_loss', 'classification_loss_weight',
        'localization_loss', 'scale_loss_weight', 'offset_loss_weight',
        'class_prediction_bias_init', 'min_box_overlap_iou',
        'max_box_predictions'
    ])):
  """Namedtuple to host object detection related parameters.

  This is a wrapper class over the fields that are either the hyper-parameters
  or the loss functions needed for the object detection task. The class is
  immutable after constructed. Please see the __new__ function for detailed
  information for each fields.
  """

  __slots__ = ()

  def __new__(cls,
              classification_loss,
              classification_loss_weight,
              localization_loss,
              scale_loss_weight,
              offset_loss_weight,
              class_prediction_bias_init=-2.19,
              min_box_overlap_iou=0.7,
              max_box_predictions=100):
    """Constructor with default values for ObjectDetectionParams.

    Args:
      classification_loss: an object_detection.core.losses.Loss object to
        compute the loss for the class predictions in CenterNet.
      classification_loss_weight: float, The weight for the classification
        loss.
      localization_loss: a object_detection.core.losses.Loss object to
        compute the loss for the center offset and height/width predictions in
        CenterNet.
      scale_loss_weight: float, The weight for localizing box
        size. Note that the scale loss is dependent on the input image size,
        since we penalize the raw height and width. This constant may need to
        be adjusted depending on the input size.
      offset_loss_weight: float, The weight for localizing center offsets.
      class_prediction_bias_init: float, the initial value of bias in the
        convolutional kernel of the class prediction head. If set to None, the
        bias is initialized with zeros.
      min_box_overlap_iou: float, the minimum IOU overlap that predicted boxes
        need have with groundtruth boxes to not be penalized. This is used for
        computing the class specific center heatmaps.
      max_box_predictions: int, the maximum number of boxes to predict.

    Returns:
      An initialized ObjectDetectionParams namedtuple.
    """
    return super(ObjectDetectionParams,
                 cls).__new__(cls, classification_loss,
                              classification_loss_weight, localization_loss,
                              scale_loss_weight, offset_loss_weight,
                              class_prediction_bias_init, min_box_overlap_iou,
                              max_box_predictions)


class CenterNetMetaArch(model.DetectionModel):
  """The CenterNet meta architecture [1].

  [1]: https://arxiv.org/abs/1904.07850
  """

  def __init__(self,
               is_training,
               add_summaries,
               num_classes,
               feature_extractor,
               image_resizer_fn,
               object_detection_params=None):
    """Initializes a CenterNet model.

    Args:
      is_training: Set to True if this model is being built for training.
      add_summaries: Whether to add tf summaries in the model.
      num_classes: int, The number of classes that the model should predict.
      feature_extractor: A CenterNetFeatureExtractor to use to extract features
        from an image.
      image_resizer_fn: a callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions and
        a 1-D tensor of shape [3] indicating shape of true image within the
        resized image tensor as the resized image tensor could be padded. See
        builders/image_resizer_builder.py.
      object_detection_params: An ObjectDetectionParams namedtuple. This object
        holds the hyper-parameters necessary for object detection. Please see
        the class definition for more details.
    """
    # Shorten the name for convenience and better formatting.
    od_params = object_detection_params
    if od_params:
      assert isinstance(od_params, ObjectDetectionParams)
      # TODO(vighneshb) Explore a size invariant version of scale loss.
      self._is_training = is_training
      self._feature_extractor = feature_extractor
      self._num_feature_outputs = feature_extractor.num_feature_outputs
      self._stride = self._feature_extractor.out_stride
      self._min_box_overlap_iou = od_params.min_box_overlap_iou
      self._image_resizer_fn = image_resizer_fn
      self._classification_loss = od_params.classification_loss
      self._classification_loss_weight = od_params.classification_loss_weight
      self._localization_loss = od_params.localization_loss
      self._offset_loss_weight = od_params.offset_loss_weight
      self._scale_loss_weight = od_params.scale_loss_weight
      self._max_box_predictions = od_params.max_box_predictions

      # The Objects as Points paper attaches loss functions to multiple
      # (`num_feature_outputs`) feature maps in the the backbone. E.g.
      # for the hourglass backbone, `num_feature_outputs` is 2.
      self._num_feature_outputs = feature_extractor.num_feature_outputs
      self._make_prediction_heads(
          num_classes, self._feature_extractor.num_feature_outputs,
          od_params.class_prediction_bias_init)
      self._target_assigner = target_assigner.CenterNetTargetAssigner(
          stride=self._stride, min_overlap=self._min_box_overlap_iou)

    super(CenterNetMetaArch, self).__init__(num_classes)

  def _make_prediction_heads(self, num_classes, num_feature_outputs,
                             class_prediction_bias_init):
    """Creates the prediction heads.

    Args:
      num_classes: int, number of classes to predict.
      num_feature_outputs: The number of feature tensors returned by the
        backbone. This function creates one feature prediction head of each
        type for every predicted feature tensor.
      class_prediction_bias_init: float, the initial value of bias in the
        convolutional kernel of the class prediction head. If set to None,
        the bias is randomly initialized.
    """

    self._class_center_heads = [
        make_prediction_net(num_classes, bias_fill=class_prediction_bias_init)
        for _ in range(num_feature_outputs)]

    self._height_width_heads = [
        make_prediction_net(NUM_SIZE_CHANNELS) for _ in
        range(num_feature_outputs)
    ]

    self._offset_heads = [
        make_prediction_net(NUM_OFFSET_CHANNELS) for _ in
        range(num_feature_outputs)
    ]

  def _compute_class_center_targets(self, input_height, input_width):
    """Computes the class center heatmap target.

    Args:
      input_height: int, the height of the input image.
      input_width: int, the width of the input image.

    Returns:
      class_center_targets: a float tensor of size
        [batch_size, output_height, output_width, num_classes] representing
        the center heatmap.

    """

    groundtruth_boxlists = [
        box_list.BoxList(boxes) for boxes in
        self.groundtruth_lists(fields.BoxListFields.boxes)
    ]
    groundtruth_weights = self.groundtruth_lists(fields.BoxListFields.weights)
    class_center_targets = self._target_assigner.assign_center_targets(
        input_height, input_width, groundtruth_boxlists,
        self.groundtruth_lists(fields.BoxListFields.classes),
        groundtruth_weights)

    return class_center_targets

  def _compute_size_and_offset_targets(self, input_height, input_width):
    """Computes the box size and the box center offset targets.

    Note that the size of first dimension of the tensors returned by this
    function is the total number of boxes in the batch.

    Args:
      input_height: int, the height of the input image.
      input_width: int, the width of the input image.

    Returns:
      batch_indices: an integer tensor of size [num_boxes, 3] holding the
        batch index, y index and x index of each box.
      height_width_targets: a float tensor of size [num_boxes, 2]  holding
        the height and width of each box.
      offset_targets: a float tensor of size [num_boxes, 2]  holding the y and
        x offsets for each box relative to the center.
      batch_weights: a float tensor of size [num_boxes]  holding the weight
        of each groundtruth box.
    """

    groundtruth_boxlists = [
        box_list.BoxList(boxes) for boxes in
        self.groundtruth_lists(fields.BoxListFields.boxes)
    ]
    groundtruth_weights = self.groundtruth_lists(fields.BoxListFields.weights)

    (batch_indices, height_width_targets, offset_targets,
     batch_weights) = self._target_assigner.assign_size_and_offset_targets(
         input_height, input_width, groundtruth_boxlists, groundtruth_weights)

    return batch_indices, height_width_targets, offset_targets, batch_weights

  def preprocess(self, inputs):

    outputs = shape_utils.resize_images_and_return_shapes(
        inputs, self._image_resizer_fn)
    resized_inputs, true_image_shapes = outputs

    return (self._feature_extractor.preprocess(resized_inputs),
            true_image_shapes)

  def predict(self, preprocessed_inputs, _):
    """Predicts CenterNet prediction tensors given an input batch.

    Feature extractors are free to produce predictions from multiple feature
    maps and therefore we return a dictionary mapping strings to lists.
    E.g. the hourglass backbone produces two feature maps.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float32 tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding predicted tensors with
        'preprocessed_inputs' - The input image after being resized and
          preprocessed by the feature extractor.
        'class_center_logits' - A list of size num_feature_outputs containing
          float tensors of size [batch_size, output_height, output_width,
          num_classes] representing the predicted class heatmap logits.
        'height_width' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted box height and width at each output
          location.
        'offset' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted y and x offsets at each output location.

    """

    def _zip_and_predict(features, heads):
      return [head(feature) for (feature, head) in zip(features, heads)]

    features_list = self._feature_extractor(preprocessed_inputs)

    predictions = {
        'class_center_logits': _zip_and_predict(features_list,
                                                self._class_center_heads),
        'height_width': _zip_and_predict(features_list,
                                         self._height_width_heads),
        'offset': _zip_and_predict(features_list,
                                   self._offset_heads),
    }

    predictions['preprocessed_inputs'] = preprocessed_inputs

    return predictions

  def _compute_class_center_loss(self, predictions, targets, per_pixel_weights,
                                 num_boxes):
    """Computes the class center loss.

    Args:
      predictions: A list of size num_feature_outputs containing float tensors
        of size [batch_size, output_height, output_width, num_classes]
        representing the predicted class heatmap logits.
      targets: A float tensor of size [batch_size, output_height, output_width,
        num_classes] representing the groundtruth center heatmap.
      per_pixel_weights: A float tensor of size [batch_size,
        output_height * output_width, 1] representing the flattened weights of
        each pixel in the output space. This tensor has 0s at pixel locations
        corresponding to padded regions of the input.
      num_boxes: int, the total number of boxes in the batch.

    Returns:
      class_center_loss: A scalar tensor representing the class center loss.
    """

    class_center_targets = _flatten_spatial_dimensions(targets)

    loss = 0.0

    for pred in predictions:
      pred = _flatten_spatial_dimensions(pred)
      loss += self._classification_loss(pred, class_center_targets,
                                        weights=per_pixel_weights)

    return (self._classification_loss_weight * tf.reduce_sum(loss) /
            (float(len(predictions)) * num_boxes))

  def _compute_height_width_loss(
      self, batch_indices, batch_weights, height_width_predictions,
      height_width_targets, num_boxes):
    """Computes the height-width loss.

    Args:
      batch_indices: an integer tensor of size [num_boxes, 3] holding the
        batch index, y index and x index of each groundtruth box.
      batch_weights: A float tensor of size [num_boxes] holding the weight
        of each groundtruth box.
      height_width_predictions: A list of size num_feature_outputs holding
        float tensors of size [batch_size, output_height, output_width, 2]
        representing the predicted box height and width at each output
        pixel
      height_width_targets: A float tensor of size [num_boxes, 2]  holding
        the height and width of each groundtruth box.
      num_boxes: int, the total number of boxes in the batch.

    Returns:
      height_width_loss: A scalar tensor representing the height-width loss.
    """

    batch_weights_expanded = tf.expand_dims(batch_weights, -1)

    loss = 0.0
    for prediction in height_width_predictions:
      prediction = target_assigner.get_batch_predictions_from_indices(
          prediction, batch_indices)

      # The dimensions passed are not as per the doc string but the loss
      # still computes the correct value.
      loss += self._localization_loss(prediction, height_width_targets,
                                      weights=batch_weights_expanded)

    return (self._scale_loss_weight * tf.reduce_sum(loss) /
            (float(len(height_width_predictions)) * num_boxes))

  def _compute_offset_loss(
      self, batch_indices, batch_weights, offset_predictions, offset_targets,
      num_boxes):
    """Computes the center offset loss.

    Args:
      batch_indices: an integer tensor of size [num_boxes, 3] holding the
        batch index, y index and x index of each box.
      batch_weights: a float tensor of size [num_boxes]  holding the weight
        of each groundtruth box.
      offset_predictions: A list of size num_feature_outputs holding
        float tensors of size [batch_size, output_height, output_width, 2]
        representing the predicted y and x offsets at each output location.
      offset_targets: a float tensor of size [num_boxes, 2]  holding
        the y and x offset corresponding to each groundtruth box.
      num_boxes: int, the total number of boxes in the batch.

    Returns:
      offset_loss: A scalar tensor representing the height-width loss.

    """

    batch_weights_expanded = tf.expand_dims(batch_weights, -1)

    loss = 0.0
    for prediction in offset_predictions:
      prediction = target_assigner.get_batch_predictions_from_indices(
          prediction, batch_indices)

      loss += self._localization_loss(prediction, offset_targets,
                                      weights=batch_weights_expanded)

    return (self._offset_loss_weight * tf.reduce_sum(loss) /
            (float(len(offset_predictions)) * num_boxes))

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Computes scalar loss tensors with respect to provided groundtruth.

    This function implements the various CenterNet losses.

    Args:
      prediction_dict: a dictionary holding predicted tensors with
        'preprocessed_inputs' - The input image after being resized and
          preprocessed by the feature extractor.
        'class_center_logits' - A list of size num_feature_outputs containing
          float tensors of size [batch_size, output_height, output_width,
          num_classes] representing the predicted class heatmap logits.
        'height_width' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted box height and width at each output
          location.
        'offset' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted y and x offsets at each output location.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping the keys ['class_center_loss', 'height_width_loss',
        'offset_loss'] to scalar tensors corresponding to the class center loss,
        height-width loss and center offset loss respectively.
    """

    _, input_height, input_width, _ = _get_shape(
        prediction_dict['preprocessed_inputs'], 4)

    output_height, output_width = (input_height // self._stride,
                                   input_width // self._stride)
    groundtruth_weights = self.groundtruth_lists(fields.BoxListFields.weights)
    num_boxes = tf.reduce_sum([tf.reduce_sum(w) for w in
                               groundtruth_weights])
    num_boxes = tf.maximum(num_boxes, 1.0)

    class_center_targets = self._compute_class_center_targets(
        input_height, input_width)
    (batch_indices, height_width_targets, offset_targets,
     batch_weights) = self._compute_size_and_offset_targets(
         input_height, input_width)

    # TODO(vighneshb) Explore whether using floor here is safe.
    output_true_image_shapes = tf.ceil(
        tf.to_float(true_image_shapes)/self._stride)
    valid_anchor_weights = get_valid_anchor_weights_in_flattened_image(
        output_true_image_shapes, output_height, output_width)

    valid_anchor_weights = tf.expand_dims(valid_anchor_weights, 2)

    losses = {
        'class_center_loss': self._compute_class_center_loss(
            prediction_dict['class_center_logits'], class_center_targets,
            valid_anchor_weights, num_boxes),
        'height_width_loss': self._compute_height_width_loss(
            batch_indices, batch_weights, prediction_dict['height_width'],
            height_width_targets, num_boxes),
        'offset_loss': self._compute_offset_loss(
            batch_indices, batch_weights, prediction_dict['offset'],
            offset_targets, num_boxes)
    }

    return losses

  def postprocess(self, prediction_dict, true_image_shapes, **params):
    """Produces boxes given a prediction dict returned by predict().

    Although predict returns a list of tensors, only the last tensor in
    each list is used for making box predictions.

    Args:
      prediction_dict: a dictionary holding predicted tensors with
        'preprocessed_inputs' - The input image after being resized and
          preprocessed by the feature extractor.
        'class_center_logits' - A list of size num_feature_outputs containing
          float tensors of size [batch_size, output_height, output_width,
          num_classes] representing the predicted class heatmap logits.
        'height_width' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted box height and width at each output
          location.
        'offset' - A list of size num_feature_outputs holding
          float tensors of size [batch_size, output_height, output_width, 2]
          representing the predicted y and x offsets at each output location.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      **params: Currently ignored.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes - A tensor of shape [batch, max_detections, 4]
          holding the predicted boxes.
        detection_scores: A tensor of shape [batch, max_detections] holding
          the predicted score for each box.
        detection_classes: An integer tensor of shape [batch, max_detections]
          containing the detected class for each box.
        num_detections: An integer tensor of shape [batch] containing the
          number of detected boxes for each sample in the batch.
    """

    boxes, classes, scores, num_detections = prediction_tensors_to_boxes(
        tf.nn.sigmoid(prediction_dict['class_center_logits'][-1]),
        prediction_dict['height_width'][-1],
        prediction_dict['offset'][-1],
        num_boxes=self._max_box_predictions)

    boxes = convert_strided_predictions_to_normalized_boxes(
        boxes, self._stride, true_image_shapes)

    return {
        fields.DetectionResultFields.detection_boxes: boxes,
        fields.DetectionResultFields.detection_scores: scores,
        fields.DetectionResultFields.detection_classes: classes,
        fields.DetectionResultFields.num_detections: num_detections,
    }

  def regularization_losses(self):
    return []

  def restore_map(self, fine_tune_checkpoint_type='classification',
                  load_all_detection_checkpoint_vars=False):

    if fine_tune_checkpoint_type == 'classification':
      return {'feature_extractor': self._feature_extractor.get_base_model()}

    if fine_tune_checkpoint_type == 'detection':
      return {'feature_extractor': self._feature_extractor.get_model()}

    else:
      raise ValueError('Unknown fine tune checkpoint type - {}'.format(
          fine_tune_checkpoint_type))

  def updates(self):
    raise RuntimeError('This model is intended to be used with model_lib_v2 '
                       'which does not support updates()')
