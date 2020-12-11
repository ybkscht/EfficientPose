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
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from tensorflow import keras
import tensorflow as tf
import math


def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args:
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns:
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args:
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns:
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args:
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns:
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args:
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns:
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def transformation_loss(model_3d_points_np, num_rotation_parameter):
    """
    Create a transformation loss functor as described in https://arxiv.org/abs/2011.04307
    Args:
        model_3d_points_np: numpy array containing the 3D model points of all classes for calculating the transformed point distances.
                            The shape is (num_classes, num_points, 3)
        num_rotation_parameter: The number of rotation parameters, usually 3 for axis angle representation
    Returns:
        A functor for computing the transformation loss given target data and predicted data.
    """
    model_3d_points = tf.convert_to_tensor(value = model_3d_points_np)
    num_points = tf.shape(model_3d_points)[1]    
    
    def _transformation_loss(y_true, y_pred):
        """ Compute the transformation loss of y_pred w.r.t. y_true using the model_3d_points tensor.
        Args:
            y_true: Tensor from the generator of shape (B, N, num_rotation_parameter + num_translation_parameter + is_symmetric_flag + class_label + anchor_state).
                    num_rotation_parameter is 3 for axis angle representation and num_translation parameter is also 3
                    is_symmetric_flag is a Boolean indicating if the GT object is symmetric or not, used to calculate the correct loss
                    class_label is the class of the GT object, used to take the correct 3D model points from the model_3d_points tensor for the transformation
                    The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, num_rotation_parameter + num_translation_parameter).
        Returns:
            The transformation loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression_rotation = y_pred[:, :, :num_rotation_parameter]
        regression_translation = y_pred[:, :, num_rotation_parameter:]
        regression_target_rotation = y_true[:, :, :num_rotation_parameter]
        regression_target_translation = y_true[:, :, num_rotation_parameter:-3]
        is_symmetric = y_true[:, :, -3]
        class_indices = y_true[:, :, -2]
        anchor_state      = tf.cast(tf.math.round(y_true[:, :, -1]), tf.int32)
    
        # filter out "ignore" anchors
        indices           = tf.where(tf.equal(anchor_state, 1))
        regression_rotation = tf.gather_nd(regression_rotation, indices) * math.pi
        regression_translation = tf.gather_nd(regression_translation, indices)
        
        regression_target_rotation = tf.gather_nd(regression_target_rotation, indices) * math.pi
        regression_target_translation = tf.gather_nd(regression_target_translation, indices)
        is_symmetric = tf.gather_nd(is_symmetric, indices)
        is_symmetric = tf.cast(tf.math.round(is_symmetric), tf.int32)
        class_indices = tf.gather_nd(class_indices, indices)
        class_indices = tf.cast(tf.math.round(class_indices), tf.int32)
        
        axis_pred, angle_pred = separate_axis_from_angle(regression_rotation)
        axis_target, angle_target = separate_axis_from_angle(regression_target_rotation)
        
        #rotate the 3d model points with target and predicted rotations        
        #select model points according to the class indices
        selected_model_points = tf.gather(model_3d_points, class_indices, axis = 0)
        #expand dims of the rotation tensors to rotate all points along the dimension via broadcasting
        axis_pred = tf.expand_dims(axis_pred, axis = 1)
        angle_pred = tf.expand_dims(angle_pred, axis = 1)
        axis_target = tf.expand_dims(axis_target, axis = 1)
        angle_target = tf.expand_dims(angle_target, axis = 1)
        
        #also expand dims of the translation tensors to translate all points along the dimension via broadcasting
        regression_translation = tf.expand_dims(regression_translation, axis = 1)
        regression_target_translation = tf.expand_dims(regression_target_translation, axis = 1)
        
        transformed_points_pred = rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
        transformed_points_target = rotate(selected_model_points, axis_target, angle_target) + regression_target_translation
        
        #distinct between symmetric and asymmetric objects
        sym_indices = tf.where(keras.backend.equal(is_symmetric, 1))
        asym_indices = tf.where(keras.backend.not_equal(is_symmetric, 1))
        
        sym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, sym_indices), (-1, num_points, 3))
        asym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, asym_indices), (-1, num_points, 3))
        
        sym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, sym_indices), (-1, num_points, 3))
        asym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, asym_indices), (-1, num_points, 3))
        
        # # compute transformed point distances
        sym_distances = calc_sym_distances(sym_points_pred, sym_points_target)
        asym_distances = calc_asym_distances(asym_points_pred, asym_points_target)

        distances = tf.concat([sym_distances, asym_distances], axis = 0)
        
        loss = tf.math.reduce_mean(distances)
        #in case of no annotations the loss is nan => replace with zero
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        
        return loss
        
    return _transformation_loss


def separate_axis_from_angle(axis_angle_tensor):
    """ Separates the compact 3-dimensional axis_angle representation in the rotation axis and a rotation angle
        Args:
            axis_angle_tensor: tensor with a shape of 3 in the last dimension.
        Returns:
            axis: Tensor of the same shape as the input axis_angle_tensor but containing only the rotation axis and not the angle anymore
            angle: Tensor of the same shape as the input axis_angle_tensor except the last dimension is 1 and contains the rotation angle
        """
    squared = tf.math.square(axis_angle_tensor)
    summed = tf.math.reduce_sum(squared, axis = -1)
    angle = tf.expand_dims(tf.math.sqrt(summed), axis = -1)
    
    axis = tf.math.divide_no_nan(axis_angle_tensor, angle)
    
    return axis, angle

def calc_sym_distances(sym_points_pred, sym_points_target):
    """ Calculates the average minimum point distance for symmetric objects
        Args:
            sym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            sym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average minimum point distance between both transformed 3D models
        """
    sym_points_pred = tf.expand_dims(sym_points_pred, axis = 2)
    sym_points_target = tf.expand_dims(sym_points_target, axis = 1)
    distances = tf.reduce_min(tf.norm(sym_points_pred - sym_points_target, axis = -1), axis = -1)
    
    return tf.reduce_mean(distances, axis = -1)
    
def calc_asym_distances(asym_points_pred, asym_points_target):
    """ Calculates the average pairwise point distance for asymmetric objects
        Args:
            asym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            asym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average point distance between both transformed 3D models
        """
    distances = tf.norm(asym_points_pred - asym_points_target, axis = -1)
    
    return tf.reduce_mean(distances, axis = -1)


#copied and adapted the following functions from tensorflow graphics source because they did not work with unknown shape
#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def cross(vector1, vector2, name=None):
  """Computes the cross product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    vector2: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    axis: The dimension along which to compute the cross product.
    name: A name for this op which defaults to "vector_cross".
  Returns:
    A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
    represents the result of the cross product.
  """
  with tf.compat.v1.name_scope(name, "vector_cross", [vector1, vector2]):
    vector1_x = vector1[:, :, 0]
    vector1_y = vector1[:, :, 1]
    vector1_z = vector1[:, :, 2]
    vector2_x = vector2[:, :, 0]
    vector2_y = vector2[:, :, 1]
    vector2_z = vector2[:, :, 2]
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return tf.stack((n_x, n_y, n_z), axis = -1)

#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def dot(vector1, vector2, axis=-1, keepdims=True, name=None):
  """Computes the dot product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    vector2: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    axis: The dimension along which to compute the dot product.
    keepdims: If True, retains reduced dimensions with length 1.
    name: A name for this op which defaults to "vector_dot".
  Returns:
    A tensor of shape `[A1, ..., Ai = 1, ..., An]`, where the dimension i = axis
    represents the result of the dot product.
  """
  with tf.compat.v1.name_scope(name, "vector_dot", [vector1, vector2]):
    return tf.reduce_sum(
        input_tensor=vector1 * vector2, axis=axis, keepdims=keepdims)

#copied from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py
def rotate(point, axis, angle, name=None):
  r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.
  Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
  $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:
  $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
  +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to rotate.
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_rotate".
  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.
  Raises:
    ValueError: If `point`, `axis`, or `angle` are of different shape or if
    their respective shape is not supported.
  """
  with tf.compat.v1.name_scope(name, "axis_angle_rotate", [point, axis, angle]):
    cos_angle = tf.cos(angle)
    axis_dot_point = dot(axis, point)
    return point * cos_angle + cross(
        axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)   