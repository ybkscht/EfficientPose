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

import cv2
import numpy as np

from utils.colors import label_color


def draw_box(image, box, color, thickness = 2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness = 2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)
        
def draw_bbox_8_2D(draw_img, bbox_8_2D, color = (0, 255, 0), thickness = 2):
    """ Draws the 2D projection of a 3D model's cuboid on an image with a given color.

    # Arguments
        draw_img     : The image to draw on.
        bbox_8_2D    : A [8 or 9, 2] matrix containing the 8 corner points (x, y) and maybe also the centerpoint.
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    #convert bbox to int and tuple
    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))
    
    #lower level
    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    #upper level
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    #sides
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)
    
    #check if centerpoint is also available to draw
    if len(bbox) == 9:
        #draw centerpoint
        cv2.circle(draw_img, bbox[8], 3, color, -1)
    
    
def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, append_centerpoint = True):
    """ Projects the 3D model's cuboid onto a 2D image plane with the given rotation, translation and camera matrix.

    Arguments:
        points_bbox_3D: numpy array with shape (8, 3) containing the 8 (x, y, z) corner points of the object's 3D model cuboid 
        rotation_vector: numpy array containing the rotation vector with shape (3,)
        translation_vector: numpy array containing the translation vector with shape (3,)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        append_centerpoint: Boolean indicating wheter to append the centerpoint or not
    Returns:
        points_bbox_2D: numpy array with shape (8 or 9, 2) with the 2D projections of the object's 3D cuboid
    """
    if append_centerpoint:
        points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape = (1, 3))], axis = 0)
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, None)
    points_bbox_2D = np.squeeze(points_bbox_2D)
    
    return points_bbox_2D
    


def draw_detections(image, boxes, scores, labels, rotations, translations, class_to_bbox_3D, camera_matrix, color = None, label_to_name = None, score_threshold = 0.5, draw_bbox_2d = False, draw_name = False):
    """ Draws detections in an image.

    # Arguments
        image: The image to draw on.
        boxes: A [N, 4] matrix (x1, y1, x2, y2).
        scores: A list of N classification scores.
        labels: A list of N labels.
        rotations: A list of N rotations
        translations: A list of N translations
        class_to_bbox_3D: A dictionary mapping the class labels to the object's 3D bboxes (cuboids)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        color: The color of the boxes. By default the color from utils.colors.label_color will be used.
        label_to_name: (optional) Functor or dictionary for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
        draw_bbox_2d: Boolean indicating wheter to draw the 2D bounding boxes or not
        draw_name: Boolean indicating wheter to draw the class names or not
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        if color is None:
            c = label_color(int(labels[i]))
        if draw_bbox_2d:
            draw_box(image, boxes[i, :], color = c)
        translation_vector = translations[i, :]
        points_bbox_2D = project_bbox_3D_to_2D(class_to_bbox_3D[labels[i]], rotations[i, :], translation_vector, camera_matrix, append_centerpoint = True)
        draw_bbox_8_2D(image, points_bbox_2D, color = c)
        if draw_name:
            if isinstance(label_to_name, dict):
                name = label_to_name[labels[i]] if label_to_name else labels[i]
            else:
                name = label_to_name(labels[i]) if label_to_name else labels[i]
            caption = name + ': {0:.2f}'.format(scores[i])
            draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, class_to_bbox_3D, camera_matrix, color = (0, 255, 0), label_to_name = None, draw_bbox_2d = False, draw_name = False):
    """ Draws annotations in an image.

    # Arguments
        image: The image to draw on.
        annotations: A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]) and rotations (shaped [N, 3]) and translations (shaped [N, 4]).
        class_to_bbox_3D: A dictionary mapping the class labels to the object's 3D bboxes (cuboids)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        color: The color of the boxes. By default the color from utils.colors.label_color will be used.
        label_to_name: (optional) Functor or dictionary for mapping a label to a name.
        draw_bbox_2d: Boolean indicating wheter to draw the 2D bounding boxes or not
        draw_name: Boolean indicating wheter to draw the class names or not
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert('rotations' in annotations)
    assert('translations' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        if color is None:
            color = (0, 255, 0)
        if draw_bbox_2d:
            draw_box(image, annotations['bboxes'][i], color = (0, 127, 0))
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        points_bbox_2D = project_bbox_3D_to_2D(class_to_bbox_3D[annotations["labels"][i]], annotations['rotations'][i, :3], annotations['translations'][i, :], camera_matrix, append_centerpoint = True)
        draw_bbox_8_2D(image, points_bbox_2D, color = color)
        if draw_name:
            if isinstance(label_to_name, dict):
                caption = label_to_name[int(label)] if label_to_name else int(label)
            else:
                caption = label_to_name(int(label)) if label_to_name else int(label)
            draw_caption(image, annotations['bboxes'][i], caption)
