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

import numpy as np
from tensorflow import keras

from utils.compute_overlap import compute_overlap


class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self,
                 sizes = (32, 64, 128, 256, 512),
                 strides = (8, 16, 32, 64, 128),
                 ratios = (1, 0.5, 2),
                 scales = (2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios = np.array([1, 0.5, 2], keras.backend.floatx()),
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
        num_rotation_parameters,
        num_translation_parameters,
        translation_anchors,
        negative_overlap = 0.4,
        positive_overlap = 0.5,
):
    """
    Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        num_rotation_parameters: Number of rotation parameters to regress (e.g. 3 for axis angle representation).
        num_translation_parameters: Number of translation parameters to regress (usually 3).
        translation_anchors: np.array of annotations of shape (N, 2) for (x, y).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state
                      (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states
                      (np.array of shape (batch_size, N, 4 + 1), where N is the number of anchors for an image,
                      the first 4 columns define regression targets for (x1, y1, x2, y2) and the last column defines
                      anchor states (-1 for ignore, 0 for bg, 1 for fg).
        transformation_batch: batch that contains 6D pose regression targets for an image & anchor states
                      (np.array of shape (batch_size, N, num_rotation_parameters + num_translation_parameters + 1),
                      where N is the number of anchors for an image,
                      the first num_rotation_parameters columns define regression targets for the rotation,
                      the next num_translation_parameters columns define regression targets for the translation,
                      and the last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert (len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert (len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."
        assert('transformation_targets' in annotations), "Annotations should contain transformation_targets."

    batch_size = len(image_group)

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=np.float32)
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=np.float32)
    transformation_batch  = np.zeros((batch_size, anchors.shape[0], num_rotation_parameters + num_translation_parameters + 1), dtype = np.float32)

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            # argmax_overlaps_inds: id of ground truth box has greatest overlap with anchor
            # (N, ), (N, ), (N, ) N is num_anchors
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors,
                                                                                            annotations['bboxes'],
                                                                                            negative_overlap,
                                                                                            positive_overlap)
            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1
            
            transformation_batch[index, ignore_indices, -1]   = -1
            transformation_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :4] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])
                
            transformation_batch[index, :, :-1] = annotations['transformation_targets'][argmax_overlaps_inds, :]
                
        # ignore anchors outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1
            transformation_batch[index, indices, -1] = -1

    return labels_batch, regression_batch, transformation_batch


def compute_gt_annotations(
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """
    Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (K, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors, (N, )
        ignore_indices: indices of ignored anchors, (N, )
        argmax_overlaps_inds: ordered overlaps indices, (N, )
    """
    # (N, K)
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # (N, )
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    # (N, )
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    # (N, )
    positive_indices = max_overlaps >= positive_overlap
    
    #get the max overlapping anchor indices for each gt box and set them to true so that each gt box has at least one positive anchor box
    max_overlapping_anchor_box_indices = np.argmax(overlaps, axis = 0)
    positive_indices[max_overlapping_anchor_box_indices] = True

    # (N, )
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """
    Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            input_shapes = [shape[inbound_layer.name] for inbound_layer in node.inbound_layers]
            if not input_shapes:
                continue
            shape[layer.name] = layer.compute_output_shape(input_shapes[0] if len(input_shapes) == 1 else input_shapes)

    return shape


def make_shapes_callback(model):
    """
    Make a function for getting the shape of the pyramid levels.
    """

    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.

    Args
          image_shape: The shape of the image.
          pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels = None,
    anchor_params = None,
    shapes_callback = None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        anchors np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
        translation anchors np.array of shape (N, 3) containing the (x, y, stride) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    all_translation_anchors = np.zeros((0, 3))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        translation_anchors = np.zeros(shape = (len(anchor_params.ratios) * len(anchor_params.scales), 2))
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        shifted_translation_anchors = translation_shift(image_shapes[idx], anchor_params.strides[idx], translation_anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        all_translation_anchors = np.append(all_translation_anchors, shifted_translation_anchors, axis = 0)

    return all_anchors.astype(np.float32), all_translation_anchors.astype(np.float32)


def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def translation_shift(shape, stride, translation_anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        translation_anchors: The translation anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
    )).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = translation_anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (translation_anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))
    
    #append stride to anchors
    stride_array = np.full((all_anchors.shape[0], 1), stride)
    all_anchors = np.concatenate([all_anchors, stride_array], axis = -1)

    return all_anchors


def generate_anchors(base_size = 16, ratios = None, scales = None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size: The base size of the anchor boxes
        ratios: Tuple containing the aspect ratios of the anchor boxes
        scales: Tuple containing the scales of the anchor boxes

    Returns:
        anchors: numpy array (num_anchors, 4) containing the anchor boxes (x1, y1, x2, y2) created with the given size, scales and ratios
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, scale_factors = None):
    """
    Computes the 2D bbox regression targets using the anchor boxes and the grdound truth 2D bounding boxes

    Args:
        anchors: np.array of anchor boxes with shape (N, 4) for (x1, y1, x2, y2).
        gt_boxes: np.array of the ground truth 2D bounding boxes with shape (N, 4) for (x1, y1, x2, y2)
        scale_factors: Optional scale factors

    Returns:
        targets: numpy array (N, 4) containing the 2D bounding box targets
    """
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.
    cya = anchors[:, 1] + ha / 2.

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.
    cy = gt_boxes[:, 1] + h / 2.
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]
    targets = np.stack([ty, tx, th, tw], axis = 1)
    return targets


def translation_transform(translation_anchors, gt_translations, scale_factors = None):
    """
    Computes the translation regression targets for an image using the translation anchors and ground truth translations.

    Args:
        translation_anchors: np.array of translation anchors with shape (N, 3) for (x, y, stride).
        gt_translations: np.array of the ground truth translations with shape (N, 3) for (x_2D, y_2D, z_3D)
        scale_factors: Optional scale factors

    Returns:
        targets: numpy array (N, 3) containing the translation regression targets
    """

    strides  = translation_anchors[:, -1]

    targets_dx = (gt_translations[:, 0] - translation_anchors[:, 0]) / strides
    targets_dy = (gt_translations[:, 1] - translation_anchors[:, 1]) / strides
    
    if scale_factors:
        targets_dx /= scale_factors[0]
        targets_dy /= scale_factors[1]
    
    targets_tz = gt_translations[:, 2]

    targets = np.stack((targets_dx, targets_dy, targets_tz), axis = 1)

    return targets
