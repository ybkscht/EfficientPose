"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under the Apache License, Version 2.0
"""

# import keras
from tensorflow import keras
import tensorflow as tf
from typeguard import typechecked
from typing import Union, Callable


class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        # return super.call, but set training
        if not training:
            return super(BatchNormalization, self).call(inputs, training=False)
        else:
            return super(BatchNormalization, self).call(inputs, training=(not self.freeze))

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


class wBiFPNAdd(keras.layers.Layer):
    """
    Layer that computes a weighted sum of BiFPN feature maps
    """
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def bbox_transform_inv(boxes, deltas, scale_factors = None):
    """
    Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas of the anchor boxes to the bounding boxes
    Args:
        boxes: Tensor containing the anchor boxes with shape (..., 4)
        deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
        scale_factors: optional scaling factor for the deltas
    Returns:
        Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)

    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


def translation_transform_inv(translation_anchors, deltas, scale_factors = None):
    """ Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors

    Args
        translation_anchors : Tensor of shape (B, N, 3), where B is the batch size, N the number of boxes and 2 values for (x, y) +1 value with the stride.
        deltas: Tensor of shape (B, N, 3). The first 2 deltas (d_x, d_y) are a factor of the stride +1 with Tz.

    Returns
        A tensor of the same shape as translation_anchors, but with deltas applied to each translation_anchors and the last coordinate is the concatenated (untouched) Tz value from deltas.
    """

    stride  = translation_anchors[:, :, -1]

    if scale_factors:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)
        
    Tz = deltas[:, :, 2]

    pred_translations = tf.stack([x, y, Tz], axis = 2) #x,y 2D Image coordinates and Tz

    return pred_translations


class ClipBoxes(keras.layers.Layer):
    """
    Layer that clips 2D bounding boxes so that they are inside the image
    """
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class RegressBoxes(keras.layers.Layer):
    """ 
    Keras layer for applying regression offset values to anchor boxes to get the 2D bounding boxes.
    """
    def __init__(self, *args, **kwargs):
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config
    
    
class RegressTranslation(keras.layers.Layer):
    """ 
    Keras layer for applying regression offset values to translation anchors to get the 2D translation centerpoint and Tz.
    """

    def __init__(self, *args, **kwargs):
        """Initializer for the RegressTranslation layer.
        """
        super(RegressTranslation, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        translation_anchors, regression_offsets = inputs
        return translation_transform_inv(translation_anchors, regression_offsets)

    def compute_output_shape(self, input_shape):
        # return input_shape[0]
        return input_shape[1]

    def get_config(self):
        config = super(RegressTranslation, self).get_config()

        return config
    
    
class CalculateTxTy(keras.layers.Layer):
    """ Keras layer for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """

    def __init__(self, *args, **kwargs):
        """ Initializer for an CalculateTxTy layer.
        """
        super(CalculateTxTy, self).__init__(*args, **kwargs)

    def call(self, inputs, fx = 572.4114, fy = 573.57043, px = 325.2611, py = 242.04899, tz_scale = 1000.0, image_scale = 1.6666666666666667, **kwargs):
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy
        
        fx = tf.expand_dims(fx, axis = -1)
        fy = tf.expand_dims(fy, axis = -1)
        px = tf.expand_dims(px, axis = -1)
        py = tf.expand_dims(py, axis = -1)
        tz_scale = tf.expand_dims(tz_scale, axis = -1)
        image_scale = tf.expand_dims(image_scale, axis = -1)
        
        x = inputs[:, :, 0] / image_scale
        y = inputs[:, :, 1] / image_scale
        tz = inputs[:, :, 2] * tz_scale
        
        x = x - px
        y = y - py
        
        tx = tf.math.multiply(x, tz) / fx
        ty = tf.math.multiply(y, tz) / fy
        
        output = tf.stack([tx, ty, tz], axis = -1)
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CalculateTxTy, self).get_config()

        return config


def filter_detections(
        boxes,
        classification,
        rotation,
        translation,
        num_rotation_parameters,
        num_translation_parameters = 3,
        class_specific_filter = True,
        nms = True,
        score_threshold = 0.01,
        max_detections = 100,
        nms_threshold = 0.5,
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        rotation: Tensor of shape (num_boxes, num_rotation_parameters) containing the rotations.
        translation: Tensor of shape (num_boxes, 3) containing the translation vectors.
        num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
        num_translation_parameters: Number of translation parameters, usually 3 
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, rotation, translation].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        rotation is shaped (max_detections, num_rotation_parameters) and contains the rotations of the non-suppressed predictions.
        translation is shaped (max_detections, num_translation_parameters) and contains the translations of the non-suppressed predictions.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)
    rotation = keras.backend.gather(rotation, indices)
    translation = keras.backend.gather(translation, indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')
    rotation = tf.pad(rotation, [[0, pad_size], [0, 0]], constant_values=-1)
    translation = tf.pad(translation, [[0, pad_size], [0, 0]], constant_values=-1)

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    rotation.set_shape([max_detections, num_rotation_parameters])
    translation.set_shape([max_detections, num_translation_parameters])

    return [boxes, scores, labels, rotation, translation]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            num_rotation_parameters,
            num_translation_parameters = 3,
            nms = True,
            class_specific_filter = True,
            nms_threshold = 0.5,
            score_threshold = 0.01,
            max_detections = 100,
            parallel_iterations = 32,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
            num_translation_parameters: Number of translation parameters, usually 3 
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, rotation, translation] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        rotation = inputs[2]
        translation = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            rotation_ = args[2]
            translation_ = args[3]

            return filter_detections(
                boxes_,
                classification_,
                rotation_,
                translation_,
                self.num_rotation_parameters,
                self.num_translation_parameters,
                nms = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold = self.score_threshold,
                max_detections = self.max_detections,
                nms_threshold = self.nms_threshold,
            )

        # call filter_detections on each batch item
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, rotation, translation],
            dtype=['float32', 'float32', 'int32', 'float32', 'float32'],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, rotation, translation].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_rotation.shape, filtered_translation.shape]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[2][0], self.max_detections, self.num_rotation_parameters),
            (input_shape[3][0], self.max_detections, self.num_translation_parameters),
        ]

    def compute_mask(self, inputs, mask = None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
            'num_rotation_parameters': self.num_rotation_parameters,
            'num_translation_parameters': self.num_translation_parameters,
        })

        return config
    
    
#copied from tensorflow addons source because tensorflow addons needs tf 2.x https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/layers/normalizations.py#L26-L279
class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    @typechecked
    def __init__(
        self,
        groups: int = 2,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: Union[None, dict, str, Callable] = "zeros",
        gamma_initializer: Union[None, dict, str, Callable] = "ones",
        beta_regularizer: Union[None, dict, str, Callable] = None,
        gamma_regularizer: Union[None, dict, str, Callable] = None,
        beta_constraint: Union[None, dict, str, Callable] = None,
        gamma_constraint: Union[None, dict, str, Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape
