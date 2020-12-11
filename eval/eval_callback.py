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

import tensorflow as tf
from tensorflow import keras
from eval.common import evaluate


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        model,
        iou_threshold = 0.5,
        score_threshold = 0.05,
        max_detections = 100,
        diameter_threshold = 0.1,
        save_path = None,
        tensorboard = None,
        weighted_average = False,
        verbose = 1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator: The generator that represents the dataset to evaluate.
            model: The model to evaluate.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.
            diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
            save_path: The path to save images with visualized detections to.
            tensorboard: Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.diameter_threshold = diameter_threshold
        self.active_model = model

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}

        # run evaluation
        average_precisions, add_metric, add_s_metric, metric_5cm_5degree, translation_diff_metric, rotation_diff_metric, metric_2d_projection, mixed_add_and_add_s_metric, average_point_distance_error_metric, average_sym_point_distance_error_metric, mixed_average_point_distance_error_metric = evaluate(
            self.generator,
            self.active_model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path,
            diameter_threshold = self.diameter_threshold
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
            
        # compute per class ADD Accuracy
        total_instances_add = []
        add_accuracys = []
        for label, (add_acc, num_annotations) in add_metric.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with ADD accuracy: {:.4f}'.format(add_acc))
            total_instances_add.append(num_annotations)
            add_accuracys.append(add_acc)
        if self.weighted_average:
            self.mean_add = sum([a * b for a, b in zip(total_instances_add, add_accuracys)]) / sum(total_instances_add)
        else:
            self.mean_add = sum(add_accuracys) / sum(x > 0 for x in total_instances_add)
            
        #same for add-s metric
        total_instances_add_s = []
        add_s_accuracys = []
        for label, (add_s_acc, num_annotations) in add_s_metric.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with ADD-S-Accuracy: {:.4f}'.format(add_s_acc))
            total_instances_add_s.append(num_annotations)
            add_s_accuracys.append(add_s_acc)
        if self.weighted_average:
            self.mean_add_s = sum([a * b for a, b in zip(total_instances_add_s, add_s_accuracys)]) / sum(total_instances_add_s)
        else:
            self.mean_add_s = sum(add_s_accuracys) / sum(x > 0 for x in total_instances_add_s)
            
        #same for 5cm 5degree metric
        total_instances_5cm_5degree = []
        accuracys_5cm_5degree = []
        for label, (acc_5cm_5_degree, num_annotations) in metric_5cm_5degree.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with 5cm-5degree-Accuracy: {:.4f}'.format(acc_5cm_5_degree))
            total_instances_5cm_5degree.append(num_annotations)
            accuracys_5cm_5degree.append(acc_5cm_5_degree)
        if self.weighted_average:
            self.mean_5cm_5degree = sum([a * b for a, b in zip(total_instances_5cm_5degree, accuracys_5cm_5degree)]) / sum(total_instances_5cm_5degree)
        else:
            self.mean_5cm_5degree = sum(accuracys_5cm_5degree) / sum(x > 0 for x in total_instances_5cm_5degree)
            
        #same for translation diffs
        translation_diffs_mean = []
        translation_diffs_std = []
        for label, (t_mean, t_std) in translation_diff_metric.items():
            print('class', self.generator.label_to_name(label), 'with Translation Differences in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
            translation_diffs_mean.append(t_mean)
            translation_diffs_std.append(t_std)
        self.mean_translation_mean = sum(translation_diffs_mean) / len(translation_diffs_mean)
        self.mean_translation_std = sum(translation_diffs_std) / len(translation_diffs_std)
            
        #same for rotation diffs
        rotation_diffs_mean = []
        rotation_diffs_std = []
        for label, (r_mean, r_std) in rotation_diff_metric.items():
            if self.verbose == 1:
                print('class', self.generator.label_to_name(label), 'with Rotation Differences in degree: Mean: {:.4f} and Std: {:.4f}'.format(r_mean, r_std))
            rotation_diffs_mean.append(r_mean)
            rotation_diffs_std.append(r_std)
        self.mean_rotation_mean = sum(rotation_diffs_mean) / len(rotation_diffs_mean)
        self.mean_rotation_std = sum(rotation_diffs_std) / len(rotation_diffs_std)
            
        #same for 2d projection metric
        total_instances_2d_projection = []
        accuracys_2d_projection = []
        for label, (acc_2d_projection, num_annotations) in metric_2d_projection.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with 2d-projection-Accuracy: {:.4f}'.format(acc_2d_projection))
            total_instances_2d_projection.append(num_annotations)
            accuracys_2d_projection.append(acc_2d_projection)
        if self.weighted_average:
            self.mean_2d_projection = sum([a * b for a, b in zip(total_instances_2d_projection, accuracys_2d_projection)]) / sum(total_instances_2d_projection)
        else:
            self.mean_2d_projection = sum(accuracys_2d_projection) / sum(x > 0 for x in total_instances_2d_projection)
            
        #same for mixed_add_and_add_s_metric
        total_instances_mixed_add_and_add_s_metric = []
        accuracys_mixed_add_and_add_s_metric = []
        for label, (acc_mixed_add_and_add_s_metric, num_annotations) in mixed_add_and_add_s_metric.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with ADD(-S)-Accuracy: {:.4f}'.format(acc_mixed_add_and_add_s_metric))
            total_instances_mixed_add_and_add_s_metric.append(num_annotations)
            accuracys_mixed_add_and_add_s_metric.append(acc_mixed_add_and_add_s_metric)
        if self.weighted_average:
            self.mean_mixed_add_and_add_s_metric = sum([a * b for a, b in zip(total_instances_mixed_add_and_add_s_metric, accuracys_mixed_add_and_add_s_metric)]) / sum(total_instances_mixed_add_and_add_s_metric)
        else:
            self.mean_mixed_add_and_add_s_metric = sum(accuracys_mixed_add_and_add_s_metric) / sum(x > 0 for x in total_instances_mixed_add_and_add_s_metric)
            
        #same for average transformed point distances
        transformed_diffs_mean = []
        transformed_diffs_std = []
        for label, (t_mean, t_std) in average_point_distance_error_metric.items():
            print('class', self.generator.label_to_name(label), 'with Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
            transformed_diffs_mean.append(t_mean)
            transformed_diffs_std.append(t_std)
        self.mean_transformed_mean = sum(transformed_diffs_mean) / len(transformed_diffs_mean)
        self.mean_transformed_std = sum(transformed_diffs_std) / len(transformed_diffs_std)
        
        #same for average symmetric transformed point distances
        transformed_sym_diffs_mean = []
        transformed_sym_diffs_std = []
        for label, (t_mean, t_std) in average_sym_point_distance_error_metric.items():
            print('class', self.generator.label_to_name(label), 'with Transformed Symmetric Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
            transformed_sym_diffs_mean.append(t_mean)
            transformed_sym_diffs_std.append(t_std)
        self.mean_transformed_sym_mean = sum(transformed_sym_diffs_mean) / len(transformed_sym_diffs_mean)
        self.mean_transformed_sym_std = sum(transformed_sym_diffs_std) / len(transformed_sym_diffs_std)
        
        #same for mixed average transformed point distances for symmetric and asymmetric objects
        mixed_transformed_diffs_mean = []
        mixed_transformed_diffs_std = []
        for label, (t_mean, t_std) in mixed_average_point_distance_error_metric.items():
            print('class', self.generator.label_to_name(label), 'with Mixed Transformed Point Distances in mm: Mean: {:.4f} and Std: {:.4f}'.format(t_mean, t_std))
            mixed_transformed_diffs_mean.append(t_mean)
            mixed_transformed_diffs_std.append(t_std)
        self.mean_mixed_transformed_mean = sum(mixed_transformed_diffs_mean) / len(mixed_transformed_diffs_mean)
        self.mean_mixed_transformed_std = sum(mixed_transformed_diffs_std) / len(mixed_transformed_diffs_std)

        if self.tensorboard is not None:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                summary = tf.Summary()
                #mAP
                summary_value_map = summary.value.add()
                summary_value_map.simple_value = self.mean_ap
                summary_value_map.tag = "mAP"
                #ADD
                summary_value_add = summary.value.add()
                summary_value_add.simple_value = self.mean_add
                summary_value_add.tag = "ADD"
                #ADD-S
                summary_value_add_s = summary.value.add()
                summary_value_add_s.simple_value = self.mean_add_s
                summary_value_add_s.tag = "ADD-S"
                #5cm 5degree
                summary_value_5cm_5degree = summary.value.add()
                summary_value_5cm_5degree.simple_value = self.mean_5cm_5degree
                summary_value_5cm_5degree.tag = "5cm_5degree"
                #translation
                summary_value_translation_mean = summary.value.add()
                summary_value_translation_mean.simple_value = self.mean_translation_mean
                summary_value_translation_mean.tag = "TranslationErrorMean_in_mm"
                summary_value_translation_std = summary.value.add()
                summary_value_translation_std.simple_value = self.mean_translation_std
                summary_value_translation_std.tag = "TranslationErrorStd_in_mm"
                #rotation
                summary_value_rotation_mean = summary.value.add()
                summary_value_rotation_mean.simple_value = self.mean_rotation_mean
                summary_value_rotation_mean.tag = "RotationErrorMean_in_degree"
                summary_value_rotation_std = summary.value.add()
                summary_value_rotation_std.simple_value = self.mean_rotation_std
                summary_value_rotation_std.tag = "RotationErrorStd_in_degree"
                #2d projection
                summary_value_2d_projection = summary.value.add()
                summary_value_2d_projection.simple_value = self.mean_2d_projection
                summary_value_2d_projection.tag = "2D_Projection"
                #summed translation and rotation errors for lr scheduling
                summary_value_summed_error = summary.value.add()
                summary_value_summed_error.simple_value = self.mean_translation_mean + self.mean_translation_std + self.mean_rotation_mean + self.mean_rotation_std
                summary_value_summed_error.tag = "Summed_Translation_Rotation_Error"
                #ADD(-S)
                summary_value_mixed_add_and_add_s_metric = summary.value.add()
                summary_value_mixed_add_and_add_s_metric.simple_value = self.mean_mixed_add_and_add_s_metric
                summary_value_mixed_add_and_add_s_metric.tag = "ADD(-S)"
                #average point distances
                summary_value_transformed_sym_mean = summary.value.add()
                summary_value_transformed_sym_mean.simple_value = self.mean_transformed_sym_mean
                summary_value_transformed_sym_mean.tag = "AverageSymmetricPointDistanceMean_in_mm"
                summary_value_transformed_sym_std = summary.value.add()
                summary_value_transformed_sym_std.simple_value = self.mean_transformed_sym_std
                summary_value_transformed_sym_std.tag = "AverageSymmetricPointDistanceStd_in_mm"
                #average point distances
                summary_value_transformed_mean = summary.value.add()
                summary_value_transformed_mean.simple_value = self.mean_transformed_mean
                summary_value_transformed_mean.tag = "AveragePointDistanceMean_in_mm"
                summary_value_transformed_std = summary.value.add()
                summary_value_transformed_std.simple_value = self.mean_transformed_std
                summary_value_transformed_std.tag = "AveragePointDistanceStd_in_mm"
                #average point distances
                summary_value_mixed_transformed_mean = summary.value.add()
                summary_value_mixed_transformed_mean.simple_value = self.mean_mixed_transformed_mean
                summary_value_mixed_transformed_mean.tag = "MixedAveragePointDistanceMean_in_mm"
                summary_value_mixed_transformed_std = summary.value.add()
                summary_value_mixed_transformed_std.simple_value = self.mean_mixed_transformed_std
                summary_value_mixed_transformed_std.tag = "MixedAveragePointDistanceStd_in_mm"
                
                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                tf.summary.scalar('mAP', self.mean_ap, epoch)
                tf.summary.scalar("ADD", self.mean_add, epoch)
                tf.summary.scalar("ADD-S", self.mean_add_s, epoch)
                tf.summary.scalar("5cm_5degree", self.mean_5cm_5degree, epoch)
                tf.summary.scalar("TranslationErrorMean_in_mm", self.mean_translation_mean, epoch)
                tf.summary.scalar("TranslationErrorStd_in_mm", self.mean_translation_std, epoch)
                tf.summary.scalar("RotationErrorMean_in_degree", self.mean_rotation_mean, epoch)
                tf.summary.scalar("RotationErrorStd_in_degree", self.mean_rotation_std, epoch)
                tf.summary.scalar("2D_Projection", self.mean_2d_projection, epoch)
                tf.summary.scalar("Summed_Translation_Rotation_Error", self.mean_translation_mean + self.mean_translation_std + self.mean_rotation_mean + self.mean_rotation_std, epoch)
                tf.summary.scalar("ADD(-S)", self.mean_mixed_add_and_add_s_metric, epoch)
                tf.summary.scalar("AverageSymmetricPointDistanceMean_in_mm", self.mean_transformed_sym_mean, epoch)
                tf.summary.scalar("AverageSymmetricPointDistanceStd_in_mm", self.mean_transformed_sym_std, epoch)
                tf.summary.scalar("AveragePointDistanceMean_in_mm", self.mean_transformed_mean, epoch)
                tf.summary.scalar("AveragePointDistanceStd_in_mm", self.mean_transformed_std, epoch)
                tf.summary.scalar("MixedAveragePointDistanceMean_in_mm", self.mean_mixed_transformed_mean, epoch)
                tf.summary.scalar("MixedAveragePointDistanceStd_in_mm", self.mean_mixed_transformed_std, epoch)

        logs['mAP'] = self.mean_ap
        logs['ADD'] = self.mean_add
        logs['ADD-S'] = self.mean_add_s
        logs['5cm_5degree'] = self.mean_5cm_5degree
        logs['TranslationErrorMean_in_mm'] = self.mean_translation_mean
        logs['TranslationErrorStd_in_mm'] = self.mean_translation_std
        logs['RotationErrorMean_in_degree'] = self.mean_rotation_mean
        logs['RotationErrorStd_in_degree'] = self.mean_rotation_std
        logs['2D-Projection'] = self.mean_2d_projection
        logs['Summed_Translation_Rotation_Error'] = self.mean_translation_mean + self.mean_translation_std + self.mean_rotation_mean + self.mean_rotation_std
        logs['ADD(-S)'] = self.mean_mixed_add_and_add_s_metric
        logs['AveragePointDistanceMean_in_mm'] = self.mean_transformed_mean
        logs['AveragePointDistanceStd_in_mm'] = self.mean_transformed_std
        logs['AverageSymmetricPointDistanceMean_in_mm'] = self.mean_transformed_sym_mean
        logs['AverageSymmetricPointDistanceStd_in_mm'] = self.mean_transformed_sym_std
        logs['MixedAveragePointDistanceMean_in_mm'] = self.mean_mixed_transformed_mean
        logs['MixedAveragePointDistanceStd_in_mm'] = self.mean_mixed_transformed_std

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
            print('ADD: {:.4f}'.format(self.mean_add))
            print('ADD-S: {:.4f}'.format(self.mean_add_s))
            print('5cm_5degree: {:.4f}'.format(self.mean_5cm_5degree))
            print('TranslationErrorMean_in_mm: {:.4f}'.format(self.mean_translation_mean))
            print('TranslationErrorStd_in_mm: {:.4f}'.format(self.mean_translation_std))
            print('RotationErrorMean_in_degree: {:.4f}'.format(self.mean_rotation_mean))
            print('RotationErrorStd_in_degree: {:.4f}'.format(self.mean_rotation_std))
            print('2D-Projection: {:.4f}'.format(self.mean_2d_projection))
            print('Summed_Translation_Rotation_Error: {:.4f}'.format(self.mean_translation_mean + self.mean_translation_std + self.mean_rotation_mean + self.mean_rotation_std))
            print('ADD(-S): {:.4f}'.format(self.mean_mixed_add_and_add_s_metric))
            print('AveragePointDistanceMean_in_mm: {:.4f}'.format(self.mean_transformed_mean))
            print('AveragePointDistanceStd_in_mm: {:.4f}'.format(self.mean_transformed_std))
            print('AverageSymmetricPointDistanceMean_in_mm: {:.4f}'.format(self.mean_transformed_sym_mean))
            print('AverageSymmetricPointDistanceStd_in_mm: {:.4f}'.format(self.mean_transformed_sym_std))
            print('MixedAveragePointDistanceMean_in_mm: {:.4f}'.format(self.mean_mixed_transformed_mean))
            print('MixedAveragePointDistanceStd_in_mm: {:.4f}'.format(self.mean_mixed_transformed_std))
