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
import random
import warnings
import cv2
import math
from tensorflow import keras

from utils.anchors import anchors_for_shape, anchor_targets_bbox, AnchorParameters
from generators.randaug import RandAugment


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            phi = 0,
            image_sizes = (512, 640, 768, 896, 1024, 1280, 1408),
            train = True,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            scale_6DoF_augmentation = (0.7, 1.3),
            chance_no_augmentation = 0.02,
            translation_scale_norm = 1000.0,
            points_for_shape_match_loss = 500,
            batch_size = 1,
            rotation_representation = "axis_angle",
            group_method='random',  # one of 'none', 'random', 'ratio'
            shuffle_groups = True,
    ):
        """
        Initialize Generator object.

        Args:
            phi: EfficientPose scaling hyperparameter phi
            image_sizes: Tuple of different input image resolutions for every phi
            train: Boolean indicating wheter the generator loads training data or not
            use_colorspace_augmentation: Boolean indicating wheter to use augmentation in the color space or not
            use_6DoF_augmentation: Boolean indicating wheter to use 6D augmentation or not
            chance_no_augmentation: Probability to skip augmentation for an image
            translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
            points_for_shape_match_loss: Number of the objects 3D model points that are used in the loss function
            batch_size: The size of the batches to generate.
            rotation_representation: String which representation of rotation should be used. Currently only axis_angle is supported
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups: If True, shuffles the groups each epoch.
        """
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_size = image_sizes[phi]
        self.groups = None
        self.anchor_parameters = AnchorParameters.default
        self.anchors, self.translation_anchors = anchors_for_shape((self.image_size, self.image_size), anchor_params = self.anchor_parameters)
        self.num_anchors = self.anchor_parameters.num_anchors()
        
        self.train = train
        self.use_colorspace_augmentation = use_colorspace_augmentation
        self.use_6DoF_augmentation = use_6DoF_augmentation
        self.chance_no_augmentation = chance_no_augmentation
        self.translation_scale_norm = translation_scale_norm
        self.points_for_shape_match_loss = points_for_shape_match_loss
        self.scale_6DoF_augmentation = scale_6DoF_augmentation
        if self.use_colorspace_augmentation:
            self.rand_aug = RandAugment(n = (1, 3), m = (1, 14))
        else:
            self.rand_aug = None

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)
            
        self.all_3d_model_points_array_for_loss = self.create_all_3d_model_points_array_for_loss(self.class_to_model_3d_points, self.points_for_shape_match_loss)
        
        
    def __getitem__(self, index):
        """
        Keras sequence method for generating the input and annotation batches for EfficientPose.
        Args:
            index: The index of the element in the sequence
        Returns:
            inputs: List with the input batches for EfficientPose [batch_images, batch_camera_parameters]
            targets: List with the target batches for EfficientPose
        """
        index = index % len(self.groups)
        group = self.groups[index]
        inputs, targets = self.compute_inputs_targets(group)
        return inputs, targets
    
    
    def compute_inputs_targets(self, group, debug = False):
        """
        Compute inputs and target outputs for the network.
        Args:
            group: The index of the group/batch of data in the generator
        Returns:
            inputs: List with the input batches for EfficientPose [batch_images, batch_camera_parameters]
            targets: List with the target batches for EfficientPose
        """

        # load images and annotations
        image_group = self.load_image_group(group)
        mask_group       = self.load_mask_group(group)
        annotations_group = self.load_annotations_group(group)
        camera_matrix_group = self.load_camera_matrix_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        #randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group, mask_group, camera_matrix_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group, camera_matrix_group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        if debug:
            return inputs, targets, annotations_group

        return inputs, targets
    
    
    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        Args:
            group: The index of the group/batch of data in the generator
        Returns:
            annotations_group: List with the annotations of the group/batch
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\', \'bboxes\', \'rotations\', \'translations\' and \'translations_x_y_2D\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\', \'bboxes\', \'rotations\', \'translations\' and \'translations_x_y_2D\'.'
            assert('rotations' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\', \'bboxes\', \'rotations\', \'translations\' and \'translations_x_y_2D\'.'
            assert('translations' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\', \'bboxes\', \'rotations\', \'translations\' and \'translations_x_y_2D\'.'
            assert('translations_x_y_2D' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\', \'bboxes\', \'rotations\', \'translations\' and \'translations_x_y_2D\'.'


        return annotations_group


    def load_image_group(self, group):
        """
        Load images for all images in a group.
        Args:
            group: The index of the group/batch of data in the generator
        Returns:
            List with the images of the group/batch
        """
        return [self.load_image(image_index) for image_index in group]
    
    
    def load_mask_group(self, group):
        """ Load masks for all images in a group.
        Args:
            group: The index of the group/batch of data in the generator
        Returns:
            List with the segmentation masks of the group/batch
        """
        return [self.load_mask(image_index) for image_index in group]
    
    
    def load_camera_matrix_group(self, group):
        """ Load intrinsic camera matrix for all images in a group.
        Args:
            group: The index of the group/batch of data in the generator
        Returns:
            List with the intrinsic camera parameters of the group/batch
        """
        return [self.load_camera_matrix(image_index) for image_index in group]
    
    
    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group and the camera parameters from the annotations_group.
        Args:
            image_group: List with images of a group/batch
            annotations_group: List with annotations of a group/batch
        Returns:
            List with the input batches for EfficientPose [batch_images, batch_camera_parameters]
        """
        batch_images = np.array(image_group).astype(np.float32)
        #model needs also the camera parameters to compute the final translation vector
        batch_camera_parameters = np.array([anno['camera_parameters'] for anno in annotations_group]).astype(np.float32)
        
        return [batch_images, batch_camera_parameters]
        

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        Args:
            image_group: List with images of a group/batch
            annotations_group: List with annotations of a group/batch
        Returns:
            List with the target batches for EfficientPose
        """

        batches_targets = anchor_targets_bbox(
            self.anchors,
            image_group,
            annotations_group,
            num_classes=self.num_classes(),
            num_rotation_parameters = self.rotation_parameter + 2, #+1 for the is_symmetric flag and +1 for the class idx to choose the correct model 3d points
            num_translation_parameters = self.translation_parameter, #x,y in 2D and Tz
            translation_anchors = self.translation_anchors,
        )
        return list(batches_targets)
    
    
    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        Args:
            image_group: List with images of a group/batch
            annotations_group: List with annotations of a group/batch
            group: Sequence containing the example id's contained in this group/batch
        Returns:
            image_group: List with the filtered images of a group/batch
            annotations_group: List with the filtered annotations of a group/batch
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            # if annotations['bboxes'].shape[0] == 0:
            #     warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
            #         group[index],
            #         image.shape,
            #     ))
        return image_group, annotations_group
    
    
    def random_transform_group(self, image_group, annotations_group, mask_group, camera_matrix_group):
        """ Randomly transforms each image and its annotations.
        Args:
            image_group: List with images of a group/batch
            annotations_group: List with annotations of a group/batch
            mask_group: List with segmentation masks of a group/batch
            camera_matrix_group: List with intrinsic camera parameters of a group/batch
        Returns:
            image_group: List with the transformed/augmented images of a group/batch
            annotations_group: List with the transformed/augmented annotations of a group/batch
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index], mask_group[index], camera_matrix_group[index])

        return image_group, annotations_group
    
    
    def random_transform_group_entry(self, image, annotations, mask, camera_matrix, transform = None):
        """ Randomly transforms image and annotation.
        Args:
            image: The image to transform/augment
            annotations: The annotations to transform/augment
            mask: The mask to transform/augment
            camera_matrix: The camera matrix of the example
        Returns:
            image: The transformed/augmented image
            annotations: The transformed/augmented annotations
        """
        #chance to skip augmentation
        chance_no_augmentation = random.random()
        if chance_no_augmentation >= self.chance_no_augmentation:
            
            # randomly transform both image and annotations
            if self.use_colorspace_augmentation:
                # apply transformation to image
                #expand image to batch and switch from BGR to RGB
                image = np.expand_dims(image[:, :, ::-1], axis = 0)
                #apply color space augmentation
                image = self.rand_aug(images = image)
                #squeeze dims again and flip back to BGR
                image = np.squeeze(image)[:, :, ::-1]
                
            if self.use_6DoF_augmentation:
                image, annotations = self.augment_6DoF_image_and_annotations(image, annotations, mask, camera_matrix)

        return image, annotations
    
    
    def augment_6DoF_image_and_annotations(self, img, annotations, mask, camera_matrix):
        """ Randomly transforms image and annotation using 6D augmentation.
        Args:
            img: The image to augment
            annotations: The annotations to augment
            mask: The segmentation mask of the image
            camera_matrix: The camera matrix of the example
        Returns:
            augmented_img: The transformed/augmented image
            annotations: The transformed/augmented annotations
        """
        num_annos = annotations["rotations"].shape[0]
        rotation_matrix_annos = np.zeros((num_annos, 3, 3), dtype = np.float32)
        translation_vector_annos = np.zeros((num_annos, 3), dtype = np.float32)
        mask_values = np.zeros((num_annos,), dtype = np.uint8)
        for i in range(num_annos):
            rotation_matrix_annos[i, :, :] = self.axis_angle_to_rotation_mat(annotations["rotations"][i, :3])
            translation_vector_annos[i, :] = annotations["translations"][i, :]
            mask_values[i] = self.name_to_mask_value[self.class_to_name[annotations["labels"][i]]]
        
        #generate random scale and angle
        scale_range, min_scale = self.get_scale_6DoF_augmentation_parameter()
        scale = random.random() * scale_range + min_scale #standard is scale between [0.7, 1.3]
        angle = random.random() * 360
        
        augmented_img, augmented_rotation_vector, augmented_translation_vector, augmented_bbox, still_valid_annos, is_valid_augmentation = self.augmentation_6DoF(img = img,
                                                                                                                                                mask = mask,
                                                                                                                                                rotation_matrix_annos = rotation_matrix_annos,
                                                                                                                                                translation_vector_annos = translation_vector_annos,
                                                                                                                                                angle = angle,
                                                                                                                                                scale = scale,
                                                                                                                                                camera_matrix = camera_matrix,
                                                                                                                                                mask_values = mask_values)
        if is_valid_augmentation:
            for i in range(num_annos):
                annotations["bboxes"][i, :] = augmented_bbox[i, :]
                annotations["rotations"][i, :3] = augmented_rotation_vector[i, :]
                annotations["translations"][i, :] = augmented_translation_vector[i, :]
                annotations["translations_x_y_2D"][i, :] = self.project_points_3D_to_2D(points_3D = np.zeros(shape = (1, 3)), #transform the object coordinate system origin point which is the centerpoint
                                                                                        rotation_vector = augmented_rotation_vector[i, :],
                                                                                        translation_vector = augmented_translation_vector[i, :],
                                                                                        camera_matrix = camera_matrix)
                
            #delete invalid annotations
            annos_to_delete = np.argwhere(still_valid_annos == False)
            annotations["labels"] = np.delete(annotations["labels"], annos_to_delete, axis = 0)
            annotations["rotations"] = np.delete(annotations["rotations"], annos_to_delete, axis = 0)
            annotations["bboxes"] = np.delete(annotations["bboxes"], annos_to_delete, axis = 0)
            annotations["translations"] = np.delete(annotations["translations"], annos_to_delete, axis = 0)
            annotations["translations_x_y_2D"] = np.delete(annotations["translations_x_y_2D"], annos_to_delete, axis = 0)
            # print("deleted {} annos".format(annos_to_delete.shape[0]))
            
        else:
            #invalid augmentation where the object probably got rotated out of the image
            augmented_img = np.array(img)
            # print("complete invalid")
        
        return augmented_img, annotations
    
    
    def augmentation_6DoF(self, img, mask, rotation_matrix_annos, translation_vector_annos, angle, scale, camera_matrix, mask_values):
        """ Computes the 6D augmentation.
        Args:
            img: The image to augment
            mask: The segmentation mask of the image
            rotation_matrix_annos: numpy array with shape (num_annotations, 3, 3) which contains the ground truth rotation matrix for each annotated object in the image
            translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the ground truth translation vectors for each annotated object in the image
            angle: rotate the image with the given angle
            scale: scale the image with the given scale
            camera_matrix: The camera matrix of the example
            mask_values: numpy array of shape (num_annotations,) containing the segmentation mask value of each annotated object
        Returns:
            augmented_img: The augmented image
            augmented_rotation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth rotation vectors for each annotated object in the image
            augmented_translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth translation vectors for each annotated object in the image
            augmented_bbox_annos: numpy array with shape (num_annotations, 4) which contains the augmented ground truth 2D bounding boxes for each annotated object in the image
            still_valid_annos: numpy boolean array of shape (num_annotations,) indicating if the augmented annotation of each object is still valid or not (object rotated out of the image for example)
            is_valid_augmentation: Boolean indicating wheter there is at least one valid annotated object after the augmentation
        """
        #get the center point from the intrinsic camera matrix
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        height, width, _ = img.shape
        #rotate and scale image
        rot_2d_mat = cv2.getRotationMatrix2D((cx, cy), -angle, scale)
        augmented_img = cv2.warpAffine(img, rot_2d_mat, (width, height))
        #append the affine transformation also to the mask to extract the augmented bbox afterwards
        augmented_mask = cv2.warpAffine(mask, rot_2d_mat, (width, height), flags = cv2.INTER_NEAREST) #use nearest neighbor interpolation to keep valid mask values
        #check if complete mask is zero
        _, is_valid_augmentation = self.get_bbox_from_mask(augmented_mask)
        if not is_valid_augmentation:
            #skip augmentation because all objects are out of bounds
            return None, None, None, None, None, False
        
        num_annos = rotation_matrix_annos.shape[0]
        
        augmented_rotation_vector_annos = np.zeros((num_annos, 3), dtype = np.float32)
        augmented_translation_vector_annos = np.zeros((num_annos, 3), dtype = np.float32)
        augmented_bbox_annos = np.zeros((num_annos, 4), dtype = np.float32)
        still_valid_annos = np.zeros((num_annos,), dtype = bool) #flag for the annotations if they are still in the image and usable after augmentation or not
        
        for i in range(num_annos):
            augmented_bbox, is_valid_augmentation = self.get_bbox_from_mask(augmented_mask, mask_value = mask_values[i])
            
            if not is_valid_augmentation:
                still_valid_annos[i] = False
                continue
        
            #create additional rotation vector representing the rotation of the given angle around the z-axis in the camera coordinate system
            tmp_rotation_vector = np.zeros((3,))
            tmp_rotation_vector[2] = angle / 180. * math.pi
            tmp_rotation_matrix, _ = cv2.Rodrigues(tmp_rotation_vector)
            #get the final augmentation rotation
            augmented_rotation_matrix = np.dot(tmp_rotation_matrix, rotation_matrix_annos[i, :, :])
            augmented_rotation_vector, _ = cv2.Rodrigues(augmented_rotation_matrix)
            
            #also rotate the gt translation vector first and then adjust Tz with the given augmentation scale
            augmented_translation_vector = np.dot(np.copy(translation_vector_annos[i, :]), tmp_rotation_matrix.T)
            augmented_translation_vector[2] /= scale
            
            #fill in augmented annotations
            augmented_rotation_vector_annos[i, :] = np.squeeze(augmented_rotation_vector)
            augmented_translation_vector_annos[i, :] = augmented_translation_vector
            augmented_bbox_annos[i, :] = augmented_bbox
            still_valid_annos[i] = True
        
        return augmented_img, augmented_rotation_vector_annos, augmented_translation_vector_annos, augmented_bbox_annos, still_valid_annos, True
    
    
    def get_scale_6DoF_augmentation_parameter(self):
        """ Returns the 6D augmentation config parameter.
        Returns:
            scale_range: Float representing the range of the 6D augmentation scale
            min_scale: Float representing the minimum scale of the 6D augmentation
        """
        min_scale, max_scale = self.scale_6DoF_augmentation
        
        if max_scale < min_scale:
            print("Warning: the given 6DoF Augmentation scale parameter max_scale {} is smaller than the min_scale parameter {}".format(max_scale, min_scale))
            return 0., 1.
        
        scale_range = max_scale - min_scale
        
        return scale_range, min_scale
    
    
    def get_bbox_from_mask(self, mask, mask_value = None):
        """ Computes the 2D bounding box from the input mask
        Args:
            mask: The segmentation mask of the image
            mask_value: The integer value of the object in the segmentation mask
        Returns:
            numpy array with shape (4,) containing the 2D bounding box
            Boolean indicating if the object is found in the given mask or not
        """
        if mask_value is None:
            seg = np.where(mask != 0)
        else:
            seg = np.where(mask == mask_value)
        #check if mask is empty
        if seg[0].size <= 0 or seg[1].size <= 0:
            return np.zeros((4,), dtype = np.float32), False
        min_x = np.min(seg[1])
        min_y = np.min(seg[0])
        max_x = np.max(seg[1])
        max_y = np.max(seg[0])
        
        return np.array([min_x, min_y, max_x, max_y], dtype = np.float32), True
    
    
    def preprocess_group(self, image_group, annotations_group, camera_matrix_group):
        """
        Preprocess each image and its annotations in its group.
        Args:
            image_group: List with images of a group/batch
            annotations_group: List with annotations of a group/batch
            camera_matrix_group: List with intrinsic camera parameters of a group/batch
        Returns:
            image_group: List with the preprocessed images of a group/batch
            annotations_group: List with the preprocessed annotations of a group/batch
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index],
                                                                                       camera_matrix_group[index])

        return image_group, annotations_group
    
    
    def preprocess_group_entry(self, image, annotations, camera_matrix):
        """
        Preprocess image and its annotations.
        Args:
            image: The image to preprocess
            annotations: The annotations to preprocess
            camera_matrix: The camera matrix of the example
        Returns:
            image: The preprocessed image
            annotations: The preprocessed annotations
        """
        
        # preprocess and resize the image
        image, image_scale = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale
        
        #normalize rotation from [-pi, +pi] to [-1, +1]
        annotations['rotations'][:, :self.rotation_parameter] /= math.pi
        
        #apply resizing to translation 2D centerpoint
        annotations["translations_x_y_2D"] *= image_scale
        #concat rotation and translation annotations to transformation targets because keras accepts only a single prediction tensor in a loss function, so this is a workaround to combine them both in the loss function
        annotations['transformation_targets'] = np.concatenate([annotations["rotations"][:, :self.rotation_parameter], annotations['translations'], annotations["rotations"][:, self.rotation_parameter:]], axis = -1)
        
        annotations['camera_parameters'] = self.get_camera_parameter_input(camera_matrix, image_scale, self.translation_scale_norm)

        return image, annotations
    
    
    def preprocess_image(self, image):
        """
        Preprocess image
        Args:
            image: The image to preprocess
        Returns:
            image: The preprocessed image
            scale: The factor with which the image was scaled to match the EfficientPose input resolution
        """
        # image, RGB
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        pad_h = self.image_size - resized_height
        pad_w = self.image_size - resized_width
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        
        return image, scale
    
    
    def get_camera_parameter_input(self, camera_matrix, image_scale, translation_scale_norm):
        """
        Returns the input vector containing the needed intrinsic camera parameters, image scale and the translation_scale_norm
        Args:
            camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
            image_scale: The factor with which the image was scaled to match the EfficientPose input resolution
            translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        Returns:
            input_vector: numpy array of shape (6,) containing [fx, fy, px, py, translation_scale_norm, image_scale]
        """
        #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
        input_vector = np.zeros((6,), dtype = np.float32)
        
        input_vector[0] = camera_matrix[0, 0]
        input_vector[1] = camera_matrix[1, 1]
        input_vector[2] = camera_matrix[0, 2]
        input_vector[3] = camera_matrix[1, 2]
        input_vector[4] = translation_scale_norm
        input_vector[5] = image_scale
        
        return input_vector
    
    
    def init_num_rotation_parameters(self, **kwargs):
        """
        Initializes the rotation representation and it's number of parameters. Currently only axis_angle is supported
        """
        self.possible_rotation_representations = {"axis_angle": 3, "rotation_matrix": 9, "quaternion": 4}
        
        rotation_representation = None
        if "rotation_representation" in kwargs:
            rotation_representation = kwargs["rotation_representation"]
        
        if rotation_representation in self.possible_rotation_representations:
            self.rotation_representation = rotation_representation
            self.rotation_parameter = self.possible_rotation_representations[self.rotation_representation]
        else:
            print("\n\nError: Invalid given rotation representation {}. Choose one of the following: {}. Continuing using 'axis_angle' representation".format(rotation_representation, self.possible_rotation_representations.keys()))
            self.rotation_representation = "axis_angle"
            self.rotation_parameter = self.possible_rotation_representations[self.rotation_representation]
    
        
    def get_translation_scale_norm(self):
        """
        Returns the translation_scale_norm parameter
        """
        return self.translation_scale_norm
    
    
    def get_all_3d_model_points_array_for_loss(self):
        """
        Returns the numpy array with shape (num_classes, num_3D_points, 3) containing the 3D model points for every object in the dataset
        """
        return self.all_3d_model_points_array_for_loss
    
        
    def create_all_3d_model_points_array_for_loss(self, class_to_model_3d_points, num_model_points):
        """
        Creates and returns the numpy array with shape (num_classes, num_3D_points, 3) containing the 3D model points for every object in the dataset
        Args:
            class_to_model_3d_points: Dictionary mapping the object class to the object's 3D model points
            num_model_points: The number of 3D points to use for each object
        Returns:
            all_model_points: numpy array with shape (num_classes, num_model_points, 3) containing the 3D model points (x, y, z) for every object in the dataset
        """
        all_model_points = np.zeros(shape = (self.num_classes(), num_model_points, 3), dtype = np.float32)
        
        for i in range(self.num_classes()):
            all_model_points[i, :, :] = self.get_model_3d_points_for_loss(class_to_model_3d_points, i, num_model_points, flatten = False)
            
        return all_model_points
    
    
    def get_model_3d_points_for_loss(self, class_to_model_3d_points, class_idx, points_for_shape_match_loss, flatten = True):
        """
        Creates and returns the numpy array with shape (points_for_shape_match_loss, 3) containing the 3D model points of a single object in the dataset.
        Subsamples 3D points if there are more than needed or use zero padding if there are less than needed.
        Args:
            class_to_model_3d_points: Dictionary mapping the object class to the object's 3D model points
            class_idx: The class index of the object
            points_for_shape_match_loss: The number of 3D points to use for each object
            flatten: Boolean indicating wheter to reshape the output array to a single dimension
        Returns:
            numpy array with shape (num_model_points, 3) or (num_model_points * 3,) containing the 3D model points (x, y, z) of an object
        """
        if class_idx in class_to_model_3d_points:
            all_model_points = class_to_model_3d_points[class_idx]
        else:
            print("Error: Unkown class idx {}".format(class_idx))
            return None
        
        num_points = all_model_points.shape[0]
        
        if num_points == points_for_shape_match_loss:
            #just return the flattened array
            if flatten:
                return np.reshape(all_model_points, (-1,))
            else:
                return all_model_points
        elif num_points < points_for_shape_match_loss:
            #use zero padding
            points = np.zeros((points_for_shape_match_loss, 3))
            points[:num_points, :] = all_model_points
            if flatten:
                return np.reshape(points, (-1,))
            else:
                return points
        else:
            #return a subsample from all 3d points
            step_size = (num_points // points_for_shape_match_loss) - 1
            if step_size < 1:
                step_size = 1
            points = all_model_points[::step_size, :]
            if flatten:
                return np.reshape(points[:points_for_shape_match_loss, :], (-1, ))
            else:
                return points[:points_for_shape_match_loss, :]
    
    
    def project_points_3D_to_2D(self, points_3D, rotation_vector, translation_vector, camera_matrix):
        """
        Transforms and projects the input 3D points onto the 2D image plane using the given rotation, translation and camera matrix    
        Arguments:
            points_3D: numpy array with shape (num_points, 3) containing 3D points (x, y, z)
            rotation_vector: numpy array containing the rotation vector with shape (3,)
            translation_vector: numpy array containing the translation vector with shape (3,)
            camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        Returns:
            points_2D: numpy array with shape (num_points, 2) with the 2D projections of the given 3D points
        """
        points_2D, jacobian = cv2.projectPoints(points_3D, rotation_vector, translation_vector, camera_matrix, None)
        points_2D = np.squeeze(points_2D)
    
        return points_2D
    
    
    def rotation_mat_to_axis_angle(self, rotation_matrix):
        """
        Computes an axis angle rotation vector from a rotation matrix 
        Arguments:
            rotation_matrix: numpy array with shape (3, 3) containing the rotation
        Returns:
            axis_angle: numpy array with shape (3,) containing the rotation
        """
        axis_angle, jacobian = cv2.Rodrigues(rotation_matrix)
        
        return np.squeeze(axis_angle)
    
    
    def axis_angle_to_rotation_mat(self, rotation_vector):
        """
        Computes a rotation matrix from an axis angle rotation vector
        Arguments:
            rotation_vector: numpy array with shape (3,) containing the rotation
        Returns:
            rotation_mat: numpy array with shape (3, 3) containing the rotation
        """
        rotation_mat, jacobian = cv2.Rodrigues(np.expand_dims(rotation_vector, axis = -1))
        
        return rotation_mat
    
    
    def transform_rotation(self, rotation_matrix, rotation_representation):
        """
        Transforms the input rotation matrix into the given rotation representation. Currently only axis_angle is supported.     
        Arguments:
            rotation_matrix: numpy array with shape (3, 3) containing the rotation
            rotation_representation: String with the rotation representation. Currently only 'axis_angle' is supported
        Returns:
            numpy array containing the rotation in the given representation
        """
        #possible rotation representations: "axis_angle", "rotation_matrix", "quaternion"
        if rotation_representation == "rotation_matrix":
            return rotation_matrix
        elif rotation_representation == "axis_angle":
            reshaped_rot_mat = np.reshape(rotation_matrix, newshape = (3, 3))
            return self.rotation_mat_to_axis_angle(reshaped_rot_mat)
        elif rotation_representation == "quaternion":
            print("Error: quaternion representation is currently not supported.")
            return None
        else:
            print("Error: Unkown rotation representation {}".format(rotation_representation))
            return None


    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]


    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.groups)
    

    def on_epoch_end(self):
        """
        Shuffles the dataset on the end of each epoch
        """
        if self.shuffle_groups:
            random.shuffle(self.groups)


    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')


    def get_anchors(self):
        """
        loads the anchors from a txt file
        """
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # (N, 2), wh
        return np.array(anchors).reshape(-1, 2)


    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')


    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')


    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')


    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')


    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')


    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')


    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')


    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')