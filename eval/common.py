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

from utils.compute_overlap import compute_overlap, wrapper_c_min_distances
from utils.visualization import draw_detections, draw_annotations

import tensorflow as tf
import numpy as np
import os
import math
from tqdm import tqdm

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold = 0.05, max_detections = 100, save_path = None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (boxes+classes = detections[num_detections, 4 + num_classes], rotations = detections[num_detections, num_rotation_parameters], translations = detections[num_detections, num_translation_parameters)

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image, scale = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)
        camera_matrix = generator.load_camera_matrix(i)
        camera_input = generator.get_camera_parameter_input(camera_matrix, scale, generator.translation_scale_norm)

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels, rotations, translations = model.predict_on_batch([np.expand_dims(image, axis=0), np.expand_dims(camera_input, axis=0)])[:5]
        
        if tf.version.VERSION >= '2.0.0':
            boxes = boxes.numpy()
            scores = scores.numpy()
            labels = labels.numpy()
            rotations = rotations.numpy()
            translations = translations.numpy()

        # correct boxes for image scale
        boxes /= scale
        
        #rescale rotations and translations
        rotations *= math.pi
        height, width, _ = raw_image.shape

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_rotations  = rotations[0, indices[scores_sort], :]
        image_translations = translations[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            draw_annotations(raw_image, generator.load_annotations(i), class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations, class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = (image_detections[image_detections[:, -1] == label, :-1], image_rotations[image_detections[:, -1] == label, :], image_translations[image_detections[:, -1] == label, :])

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (bboxes = annotations[num_detections, 5], rotations = annotations[num_detections, num_rotation_parameters], translations = annotations[num_detections, num_translation_parameters])

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = (annotations['bboxes'][annotations['labels'] == label, :].copy(), annotations['rotations'][annotations['labels'] == label, :].copy(), annotations['translations'][annotations['labels'] == label, :].copy())

    return all_annotations


def check_6d_pose_2d_reprojection(model_3d_points, rotation_gt, translation_gt, rotation_pred, translation_pred, camera_matrix, pixel_threshold = 5.0):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 2D reprojection metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        pixel_threshold: Threshold in pixels when a prdicted 6D pose in considered to be correct
    # Returns
        Boolean indicating wheter the predicted 6D pose is correct or not
    """
    #transform points into camera coordinate system with gt and prediction transformation parameters respectively
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred
    
    #project the points on the 2d image plane
    points_2D_gt, _ = np.squeeze(cv2.projectPoints(transformed_points_gt, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))
    points_2D_pred, _ = np.squeeze(cv2.projectPoints(transformed_points_pred, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))
    
    distances = np.linalg.norm(points_2D_gt - points_2D_pred, axis = -1)
    mean_distances = np.mean(distances)
    
    if mean_distances <= pixel_threshold:
        is_correct = True
    else:
        is_correct = False
        
    return is_correct


def check_6d_pose_add(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred, translation_pred, diameter_threshold = 0.1):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
        transformed_points_gt: numpy array with shape (num_3D_points, 3) containing the object's 3D points transformed with the ground truth 6D pose
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred

    distances = np.linalg.norm(transformed_points_gt - transformed_points_pred, axis = -1)
    mean_distances = np.mean(distances)
    
    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, mean_distances, transformed_points_gt


def check_6d_pose_add_s(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred, translation_pred, diameter_threshold = 0.1, max_points = 1000):    
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD-S metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
        max_points: Max number of 3D points to calculate the distances (The computed distance between all points to all points can be very memory consuming)
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred
    #calc all distances between all point pairs and get the minimum distance for every point
    num_points = transformed_points_gt.shape[0]
    
    #approximate the add-s metric and use max max_points of the 3d model points to reduce computational time
    step = num_points // max_points + 1
    
    min_distances = wrapper_c_min_distances(transformed_points_gt[::step, :], transformed_points_pred[::step, :])
    mean_distances = np.mean(min_distances)
    
    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, mean_distances


def calc_translation_diff(translation_gt, translation_pred):
    """ Computes the distance between the predicted and ground truth translation

    # Arguments
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        The translation distance
    """
    return np.linalg.norm(translation_gt - translation_pred)


def calc_rotation_diff(rotation_gt, rotation_pred):
    """ Calculates the distance between two rotations in degree
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
    # Returns
        the rotation distance in degree
    """  
    rotation_diff = np.dot(rotation_pred, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = (trace - 1.) / 2.
    if trace < -1.:
        trace = -1.
    elif trace > 1.:
        trace = 1.
    angular_distance = np.rad2deg(np.arccos(trace))
    
    return abs(angular_distance)


def check_6d_pose_5cm_5degree(rotation_gt, translation_gt, rotation_pred, translation_pred):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 5cm 5 degree metric
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py def cm_degree_5_metric(self, pose_pred, pose_targets):
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        translation_distance: the translation distance
        rotation_distance: the rotation distance
    """    
    translation_distance = calc_translation_diff(translation_gt, translation_pred)
    
    rotation_distance = calc_rotation_diff(rotation_gt, rotation_pred)
    
    if translation_distance <= 50 and rotation_distance <= 5:
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, translation_distance, rotation_distance


def test_draw(image, camera_matrix, points_3d):
    """ Projects and draws 3D points onto a 2D image and shows the image for debugging purposes

    # Arguments
        image: The image to draw on
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        points_3d: numpy array with shape (num_3D_points, 3) containing the 3D points to project and draw (usually the object's 3D points transformed with the ground truth 6D pose)
    """
    points_2D, jacobian = cv2.projectPoints(points_3d, np.zeros((3,)), np.zeros((3,)), camera_matrix, None)
    points_2D = np.squeeze(points_2D)
    points_2D = np.copy(points_2D).astype(np.int32)
    
    tuple_points = tuple(map(tuple, points_2D))
    for point in tuple_points:
        cv2.circle(image, point, 2, (255, 0, 0), -1)
        
    cv2.imshow('image', image)
    cv2.waitKey(0)


def evaluate(
    generator,
    model,
    iou_threshold = 0.5,
    score_threshold = 0.05,
    max_detections = 100,
    save_path = None,
    diameter_threshold = 0.1,
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        Several dictionaries mapping class names to the computed metrics.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    all_3d_models      = generator.get_models_3d_points_dict()
    all_3d_model_diameters = generator.get_objects_diameter_dict()
    average_precisions = {}
    add_metric = {}
    add_s_metric = {}
    metric_5cm_5degree = {}
    translation_diff_metric = {}
    rotation_diff_metric = {}
    metric_2d_projection = {}
    mixed_add_and_add_s_metric = {}
    average_point_distance_error_metric = {}
    average_sym_point_distance_error_metric = {}
    mixed_average_point_distance_error_metric = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        true_positives_add  = np.zeros((0,))
        true_positives_add_s  = np.zeros((0,))
        model_3d_points = all_3d_models[label]
        model_3d_diameter = all_3d_model_diameters[label]
        true_positives_5cm_5degree  = np.zeros((0,))
        translation_diffs = np.zeros((0,))
        rotation_diffs = np.zeros((0,))
        true_positives_2d_projection  = np.zeros((0,))
        point_distance_errors = np.zeros((0,))
        point_sym_distance_errors = np.zeros((0,))

        for i in tqdm(range(generator.size())):
            detections           = all_detections[i][label][0]
            detections_rotations = all_detections[i][label][1]
            detections_translations = all_detections[i][label][2]
            annotations          = all_annotations[i][label][0]
            annotations_rotations = all_annotations[i][label][1]
            annotations_translations = all_annotations[i][label][2]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d, d_rotation, d_translation in zip(detections, detections_rotations, detections_translations):
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
                assigned_rotation = annotations_rotations[assigned_annotation, :3]
                assigned_translation = annotations_translations[assigned_annotation, :]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    #correct 2d object detection => check if the 6d pose is also correct
                    is_correct_6d_pose_add, mean_distances_add, transformed_points_gt = check_6d_pose_add(model_3d_points,
                                                                                                        model_3d_diameter,
                                                                                                        rotation_gt = generator.axis_angle_to_rotation_mat(assigned_rotation),
                                                                                                        translation_gt = np.squeeze(assigned_translation),
                                                                                                        rotation_pred = generator.axis_angle_to_rotation_mat(d_rotation),
                                                                                                        translation_pred = d_translation,
                                                                                                        diameter_threshold = diameter_threshold)
                    
                    is_correct_6d_pose_add_s, mean_distances_add_s = check_6d_pose_add_s(model_3d_points,
                                                                                       model_3d_diameter,
                                                                                       rotation_gt = generator.axis_angle_to_rotation_mat(assigned_rotation),
                                                                                       translation_gt = np.squeeze(assigned_translation),
                                                                                       rotation_pred = generator.axis_angle_to_rotation_mat(d_rotation),
                                                                                       translation_pred = d_translation,
                                                                                       diameter_threshold = diameter_threshold)
                    
                    is_correct_6d_pose_5cm_5degree, translation_distance, rotation_distance = check_6d_pose_5cm_5degree(rotation_gt = generator.axis_angle_to_rotation_mat(assigned_rotation),
                                                                                                                         translation_gt = np.squeeze(assigned_translation),
                                                                                                                         rotation_pred = generator.axis_angle_to_rotation_mat(d_rotation),
                                                                                                                         translation_pred = d_translation)
                    
                    is_correct_2d_projection = check_6d_pose_2d_reprojection(model_3d_points,
                                                                             rotation_gt = generator.axis_angle_to_rotation_mat(assigned_rotation),
                                                                             translation_gt = np.squeeze(assigned_translation),
                                                                             rotation_pred = generator.axis_angle_to_rotation_mat(d_rotation),
                                                                             translation_pred = d_translation,
                                                                             camera_matrix = generator.load_camera_matrix(i),
                                                                             pixel_threshold = 5.0)
                    
                    # #draw transformed gt points in image to test the transformation
                    # test_draw(generator.load_image(i), generator.load_camera_matrix(i), transformed_points_gt)
                    
                    if is_correct_6d_pose_add:
                        true_positives_add  = np.append(true_positives_add, 1)
                    if is_correct_6d_pose_add_s:
                        true_positives_add_s  = np.append(true_positives_add_s, 1)
                    if is_correct_6d_pose_5cm_5degree:
                        true_positives_5cm_5degree = np.append(true_positives_5cm_5degree, 1)
                    if is_correct_2d_projection:
                        true_positives_2d_projection = np.append(true_positives_2d_projection, 1)
                        
                    translation_diffs = np.append(translation_diffs, translation_distance)
                    rotation_diffs = np.append(rotation_diffs, rotation_distance)
                    point_distance_errors = np.append(point_distance_errors, mean_distances_add)
                    point_sym_distance_errors = np.append(point_sym_distance_errors, mean_distances_add_s)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        
        #compute add accuracy
        add_accuracy = np.sum(true_positives_add) / num_annotations
        add_metric[label] = add_accuracy, num_annotations
        
        #compute add-s accuracy
        add_s_accuracy = np.sum(true_positives_add_s) / num_annotations
        add_s_metric[label] = add_s_accuracy, num_annotations
        
        #compute 5cm 5degree accuracy
        accuracy_5cm_5degree = np.sum(true_positives_5cm_5degree) / num_annotations
        metric_5cm_5degree[label] = accuracy_5cm_5degree, num_annotations
        
        #compute the mean and std of the translation- and rotation differences
        mean_translations = np.mean(translation_diffs)
        std_translations = np.std(translation_diffs)
        translation_diff_metric[label] = mean_translations, std_translations
        
        mean_rotations = np.mean(rotation_diffs)
        std_rotations = np.std(rotation_diffs)
        rotation_diff_metric[label] = mean_rotations, std_rotations
        
        #compute 2d projection accuracy
        accuracy_2d_projection = np.sum(true_positives_2d_projection) / num_annotations
        metric_2d_projection[label] = accuracy_2d_projection, num_annotations
        
        #compute the mean and std of the transformed point errors
        mean_point_distance_errors = np.mean(point_distance_errors)
        std_point_distance_errors = np.std(point_distance_errors)
        average_point_distance_error_metric[label] = mean_point_distance_errors, std_point_distance_errors
        
        #compute the mean and std of the symmetric transformed point errors
        mean_point_sym_distance_errors = np.mean(point_sym_distance_errors)
        std_point_sym_distance_errors = np.std(point_sym_distance_errors)
        average_sym_point_distance_error_metric[label] = mean_point_sym_distance_errors, std_point_sym_distance_errors
        
    
    #fill in the add values for asymmetric objects and add-s for symmetric objects
    for label, add_tuple in add_metric.items():
        add_s_tuple = add_s_metric[label]
        if generator.class_labels_to_object_ids[label] in generator.symmetric_objects:
            mixed_add_and_add_s_metric[label] = add_s_tuple
        else:
            mixed_add_and_add_s_metric[label] = add_tuple
            
    #fill in the average point distance values for asymmetric objects and the corresponding average sym point distances for symmetric objects
    for label, asym_tuple in average_point_distance_error_metric.items():
        sym_tuple = average_sym_point_distance_error_metric[label]
        if generator.class_labels_to_object_ids[label] in generator.symmetric_objects:
            mixed_average_point_distance_error_metric[label] = sym_tuple
        else:
            mixed_average_point_distance_error_metric[label] = asym_tuple
        

    return average_precisions, add_metric, add_s_metric, metric_5cm_5degree, translation_diff_metric, rotation_diff_metric, metric_2d_projection, mixed_add_and_add_s_metric, average_point_distance_error_metric, average_sym_point_distance_error_metric, mixed_average_point_distance_error_metric
