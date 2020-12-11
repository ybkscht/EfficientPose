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

import os
import numpy as np
import cv2
import yaml
import random
import copy
from plyfile import PlyData

from generators.common import Generator


#Generator for the Occlusion Dataset (Linemod because Occlusion is a subset of linemod) downloaded from here: https://github.com/j96w/DenseFusion and the valid poses from here https://github.com/chensong1995/HybridPose
class OcclusionGenerator(Generator):
    """
    Generator for the Occlusion dataset

    """
    def __init__(self,
                 dataset_base_path,
                 object_ids = {"ape": 1, "can": 5, "cat": 6, "driller": 8, "duck": 9, "eggbox": 10, "glue": 11, "holepuncher": 12}, #dictionary with the names and object ids of the occlusion dataset
                 image_extension = ".png",
                 shuffle_dataset = True,
                  symmetric_objects = {"glue", 11, "eggbox", 10}, #set with names and indices of symmetric objects
                 **kwargs):
        """
        Initializes a Occlusion generator
        Args:
            dataset_base_path: path to the Linemod dataset (Occlusion is a subset of Linemod)
            object_ids: Dictionary mapping the Occlusion object names to their object id's
            image_extension: String containing the image filename extension
            shuffle_dataset: Boolean wheter to shuffle the dataset or not
             symmetric_objects: set with names and indices of symmetric objects
        
        """
        self.dataset_base_path = dataset_base_path
        self.dataset_path = os.path.join(self.dataset_base_path, "data")
        self.model_path = os.path.join(self.dataset_base_path, "models")
        self.image_extension = image_extension
        self.shuffle_dataset = shuffle_dataset
        self.translation_parameter = 3
        self.symmetric_objects = symmetric_objects
        self.object_ids = object_ids
        self.object_id = 2 #hardcored for occlusion
        
        #set the class and name dict for mapping each other
        self.class_to_name = {0: "ape", 1: "can", 2: "cat", 3: "driller", 4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher"}
        self.name_to_class = {val: key for key, val in self.class_to_name.items()}
        self.object_ids_to_class_labels, self.class_labels_to_object_ids = self.map_object_ids_to_class_labels(self.object_ids, self.name_to_class)
        self.name_to_mask_value = {"ape": 21,
                                   "can": 106,
                                   "cat": 128,
                                   "driller": 170,
                                   "duck": 191,
                                   "eggbox": 213,
                                   "glue": 234,
                                   "holepuncher": 255}
        
        #check and set the rotation representation and the number of parameters to use
        self.init_num_rotation_parameters(**kwargs)
        
        #check if both paths exist
        if not self.check_path(self.dataset_base_path) or not self.check_path(self.dataset_path) or not self.check_path(self.model_path):
            return None
        
        #get dict with object ids as keys and object subdirs as values
        self.object_paths_and_ids = {int(subdir): os.path.join(self.dataset_path, subdir) for subdir in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, subdir))}
        
        if not self.object_id in self.object_paths_and_ids:
            print("The given object id {} was not found in the dataset dir {}".format(self.object_id, self.dataset_path))
            return None
        
        #get path containing the data for the given object
        self.object_path = self.object_paths_and_ids[self.object_id]
        
        #get path containing infos about the valid examples. valid_poses subfolder needs to be copied manually in this dir
        self.valid_annotations_path = os.path.join(self.object_path, "valid_poses")
        if not self.check_path(self.valid_annotations_path):
            print("\n\nError: The path {} containing the information about the valid annotations of the objects in the occlusion dataset was not found! please download and copy it in this dir {}".format(self.valid_annotations_path, self.object_path))
            return None
        
        self.class_to_valid_examples, self.name_to_valid_examples = self.parse_valid_examples(self.valid_annotations_path, self.name_to_class)
        
        #get all train or test data examples from the dataset in the given split
        if not "train" in kwargs or kwargs["train"]:
            self.data_file = os.path.join(self.object_path, "train.txt")
        else:
            self.data_file = os.path.join(self.object_path, "test.txt")
            
        self.data_examples = self.parse_examples(data_file = self.data_file)
        
        #parse yaml files with ground truth annotations and infos about camera intrinsics and 3D BBox
        self.gt_dict = self.parse_yaml(self.object_path)
        self.info_dict = self.parse_yaml(self.object_path, filename = "info.yml")
        self.all_models_dict = self.parse_yaml(self.model_path, filename = "models_info.yml")
        
        #create dict with the class indices/names as keys and 3d model diameters as values
        self.class_to_model_3d_diameters, self.name_to_model_3d_diameters = self.create_model_3d_diameters_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)
        
        #create dict with the class indices/names as keys and model 3d bboxes as values
        self.class_to_model_3d_bboxes, self.name_to_model_3d_bboxes = self.create_model_3d_bboxes_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)
        
        #load 3d points for all models and store it in two dicts: one with class idx as keys and the other one with the model names as keys
        self.class_to_model_3d_points, self.name_to_model_3d_points = self.load_model_3d_points(self.model_path, self.object_ids, self.name_to_class)
        
        #get the final input and annotation infos for the base generator
        self.image_paths, self.mask_paths, self.depth_paths, self.annotations, self.infos = self.prepare_dataset(self.object_path, self.data_examples, self.gt_dict, self.info_dict, self.class_to_valid_examples)
        
        #shuffle dataset
        if self.shuffle_dataset:
            self.image_paths, self.mask_paths, self.depth_paths, self.annotations, self.infos = self.shuffle_sequences(self.image_paths, self.mask_paths, self.depth_paths, self.annotations, self.infos)
        
        #init base class
        Generator.__init__(self, **kwargs)
        
        
    def map_object_ids_to_class_labels(self, object_ids_dict, name_to_class_dict):
        """
       Creates two dictionaries containing the Occlusion dataset object id as key and the EfficientPose class label as value and vice versa
        Args:
            object_ids_dict: Dictionary mapping the Occlusion object names to their object id's
            name_to_class_dict: Dictionary mapping the Occlusion object name to the EfficientPose class label
        Returns:
            Dictionaries containing the Occlusion dataset object id as key and the EfficientPose class label as value and vice versa
    
        """
        object_ids_to_class_labels = {object_id: name_to_class_dict[object_name] for object_name, object_id in object_ids_dict.items()}
        class_labels_to_object_ids = {class_label: object_id for object_id, class_label in object_ids_to_class_labels.items()}
        return object_ids_to_class_labels, class_labels_to_object_ids
    
    
    def create_model_3d_diameters_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
       Creates two dictionaries containing the class idx and the model name as key and the 3D model diameters as values
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes and diameters in the Linemod dataset format
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the EfficientPose classes
            class_to_name: Dictionary mapping the EfficientPose classes to their names
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values
    
        """
        class_to_model_3d_diameters = dict()
        name_to_model_3d_diameters = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            class_to_model_3d_diameters[class_label] = all_models_dict[object_id]["diameter"]
            name_to_model_3d_diameters[class_to_name[class_label]] = all_models_dict[object_id]["diameter"]
            
        return class_to_model_3d_diameters, name_to_model_3d_diameters
    
    
    def create_model_3d_bboxes_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
       Creates two dictionaries which are mapping the class indices, respectively the class names to the 3D model cuboids
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes in the Linemod dataset format (min_x, min_y, min_z, size_x, size_y, size_z)
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the EfficientPose classes
            class_to_name: Dictionary mapping the EfficientPose classes to their names
        Returns:
            Two dictionaries containing the EfficientPose class indices or the class names as keys and the 3D model cuboids as values
    
        """
        class_to_model_3d_bboxes = dict()
        name_to_model_3d_bboxes = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            model_bbox = self.get_bbox_3d(all_models_dict[object_id])
            class_to_model_3d_bboxes[class_label] = model_bbox
            name_to_model_3d_bboxes[class_to_name[class_label]] = model_bbox
            
        return class_to_model_3d_bboxes, name_to_model_3d_bboxes
    
    
    def parse_valid_example_id(self, path_to_file):
        """
       Reads the given file containing a valid example id's of an object
        Args:
            path_to_file: Path to the file
        Returns:
            The example id of the valid pose in the given file
    
        """
        with open(path_to_file) as fid:
            lines = [line.strip() for line in fid if line != ""]
        
        example_id = int(lines[-1])
        
        return example_id
        
        
    def parse_valid_examples(self, valid_annotations_path, name_to_class_dict):
        """
        Reads and parses all files containing the valid poses of all object subdirs in the given valid annotations path
        Args:
            valid_annotations_path: Path to the subdir containing all valid example ids for all object
            name_to_class_dict: Dictionary mapping the Occlusion object name to the EfficientPose class label
        Returns:
            class_to_valid_examples: Dictionary mapping the object class to a tuple of all valid data examples of this object.
                                    In Occlusion there are usually all objects annotated even if they are not visible at all.
                                    So filter those annotations out.
            name_to_valid_examples: The same as class_to_valid_examples but with the class names as keys
        """
        #init dicts with None
        class_to_valid_examples = {value: None for value in name_to_class_dict.values()}
        name_to_valid_examples = {key: None for key in name_to_class_dict.keys()}
        
        for object_name in name_to_valid_examples.keys():
            object_path = os.path.join(valid_annotations_path, object_name)
            if not self.check_path(object_path):
                return None, None #make sure to crash to avoid missing object annotations in training/testing
            
            txt_files = [os.path.join(object_path, filename) for filename in os.listdir(object_path) if ".txt" in filename]
            valid_object_ids = tuple()
            for txt_file in txt_files:
                valid_object_ids += (self.parse_valid_example_id(txt_file), )
                
            name_to_valid_examples[object_name] = valid_object_ids
            class_to_valid_examples[name_to_class_dict[object_name]] = valid_object_ids
            
        return class_to_valid_examples, name_to_valid_examples
        
        
    def load_model_3d_points(self, all_models_path, name_to_object_id, name_to_class):
        """
        Loads all objects 3D models and creates dictionaries mapping the object class and name respectively to the corresponding 3D model points
        Args:
            all_models_path: Path where all model ply files are stored
            name_to_object_id: Dictionary mapping the Occlusion object name to the object id
            name_to_class: Dictionary mapping the Occlusion object name to the EfficientPose class label
        Returns:
            class_to_model_3d_points: Dictionary mapping the object class to the object's 3D model points
            name_to_model_3d_points: Dictionary mapping the object name to the object's 3D model points
        """
        class_to_model_3d_points = {}
        name_to_model_3d_points = {}
        
        for model_name, class_idx in name_to_object_id.items():
            filename = "obj_{:02}.ply".format(class_idx)
            point_file = os.path.join(all_models_path, filename)
            
            if not os.path.isfile(point_file):
                print("\n\nError: File {} with 3D model points for model {} does not exist!\n\n".format(point_file, model_name))
                continue
            
            model_3d_points = self.load_model_ply(point_file)
            
            #add to dicts
            class_to_model_3d_points[name_to_class[model_name]] = model_3d_points
            name_to_model_3d_points[model_name] = model_3d_points
            
        return class_to_model_3d_points, name_to_model_3d_points
        
        
    def get_bbox_3d(self, model_dict):
        """
        Converts the 3D model cuboid from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
        Args:
            model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
        Returns:
            bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.
    
        """
        #get infos from model dict
        min_point_x = model_dict["min_x"]
        min_point_y = model_dict["min_y"]
        min_point_z = model_dict["min_z"]
        
        size_x = model_dict["size_x"]
        size_y = model_dict["size_y"]
        size_z = model_dict["size_z"]
        
        bbox = np.zeros(shape = (8, 3))
        #lower level
        bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
        bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
        bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
        bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
        #upper level
        bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
        bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
        bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
        bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
        
        return bbox
    
    
    def get_bbox_3d_dict(self, class_idx_as_key = True):
        """
       Returns a dictionary which either maps the class indices or the class names to the 3D model cuboids
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model cuboids as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_bboxes
        else:
            return self.name_to_model_3d_bboxes
    
    
    def is_symmetric_object(self, name_or_object_id):
        """
       Check if the given object is considered to be symmetric or not
        Args:
            name_or_object_id: The name of the object or the id of the object
        Returns:
            Boolean indicating wheter the object is symmetric or not
    
        """
        return name_or_object_id in self.symmetric_objects
    
    
    def get_models_3d_points_dict(self, class_idx_as_key = True):
        """
       Returns either the 3d model points dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model points as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_points
        else:
            return self.name_to_model_3d_points
    
    
    def get_objects_diameter_dict(self, class_idx_as_key = True):
        """
       Returns either the diameter dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values
    
        """
        if class_idx_as_key:
            return self.class_to_model_3d_diameters
        else:
            return self.name_to_model_3d_diameters
    
        
    def get_num_rotation_parameters(self):
        """
       Returns the number of rotation parameters. For axis angle representation there are 3 parameters used
    
        """
        return self.rotation_parameter
    
    
    def get_num_translation_parameters(self):
        """
       Returns the number of translation parameters. Usually 3 
    
        """
        return self.translation_parameter
            
        
    def shuffle_sequences(self, image_paths, mask_paths, depth_paths, annotations, infos):
        """
       Takes sequences (e.g. lists) containing the dataset and shuffle them so that the corresponding entries still match
    
        """
        concatenated = list(zip(image_paths, mask_paths, depth_paths, annotations, infos))
        random.shuffle(concatenated)
        image_paths, mask_paths, depth_paths, annotations, infos = zip(*concatenated)
        
        return image_paths, mask_paths, depth_paths, annotations, infos
    
    
    def load_model_ply(self, path_to_ply_file):
        """
       Loads a 3D model from a plyfile
        Args:
            path_to_ply_file: Path to the ply file containing the object's 3D model
        Returns:
            points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points
    
        """
        model_data = PlyData.read(path_to_ply_file)
                                  
        vertex = model_data['vertex']
        points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1).astype(np.float32)
        
        return points_3d
        
    
    def prepare_dataset(self, object_path, data_examples, gt_dict, info_dict, class_to_valid_examples):
        """
       Prepares the Occlusion dataset and converts the data from the Linemod format to the EfficientPose format
        Args:
            object_path: path to the Linemod subscene containing Occlusion (This is a hardcoded parameter in the init function)
            data_examples: List containing all data examples of the used dataset split (train or test)
            gt_dict: Dictionary mapping the example id's to the corresponding ground truth data
            info_dict: Dictionary mapping the example id's to the intrinsic camera parameters
            class_to_valid_examples: Dictionary mapping the object class to a tuple of all valid data examples of this object.
                                    In Occlusion there are usually all objects annotated even if they are not visible at all.
                                    So filter those annotations out.
        Returns:
            image_paths: List with all rgb image paths in the dataset split
            mask_paths: List with all segmentation mask paths in the dataset split
            depth_paths: List with all depth image paths in the dataset split (Currently not used in EfficientPose)
            annotations: List with all annotation dictionaries in the dataset split
            infos: List with all info dictionaries (intrinsic camera parameters) in the dataset split
    
        """
        all_images_path = os.path.join(object_path, "rgb")
        
        #load all images which are in the dataset split (train/test)
        all_filenames = [filename for filename in os.listdir(all_images_path) if self.image_extension in filename and filename.replace(self.image_extension, "") in data_examples]
        image_paths = [os.path.join(all_images_path, filename) for filename in all_filenames]
        mask_paths = [img_path.replace("rgb", "merged_masks") for img_path in image_paths] #this dir is generated from generate_masks.py script in occlusion_labels and needs also to be copied into the subdir
        depth_paths = [img_path.replace("rgb", "depth") for img_path in image_paths]
        
        #parse the example ids for the gt dict from filenames
        example_ids = [int(filename.split(".")[0]) for filename in all_filenames]
        filtered_gt_lists = [(gt_dict[key], key) for key in example_ids]#creates a list containing tuples of the example id and list of all annotations per image. usually one element but at object id 2 is also the occlusion dataset included
        #filer out invalid annotations and annotations from objects not included in the occlusion dataset
        filtered_gts = [[anno for anno in gt_list if anno["obj_id"] in self.object_ids.values() and example_id in class_to_valid_examples[self.object_ids_to_class_labels[anno["obj_id"]]]] for gt_list, example_id in filtered_gt_lists]
                
        filtered_infos = [info_dict[key] for key in example_ids] #filter info dicts containing camera calibration etc analogue to gts
        
        #insert camera calibration as 3x3 numpy array in the infos
        infos = self.insert_np_cam_calibration(filtered_infos)
        
        #convert the gt into the base generator format
        annotations = self.convert_gt(filtered_gts, infos, mask_paths)
        
        # max_angle = max(annotations, key = lambda dic: np.max(dic["rotations"]))
        # min_angle = min(annotations, key = lambda dic: np.min(dic["rotations"]))
        # print("\n\n\nmax angle: ", max_angle, "\nmin angle: ", min_angle, "\n\n")
        
        # max_t = max(annotations, key = lambda dic: np.max(dic["translations"]))
        # min_t = min(annotations, key = lambda dic: np.min(dic["translations"]))
        # print("\n\n\nmax translation: ", max_t, "\nmin translation: ", min_t, "\n\n")
        
        return image_paths, mask_paths, depth_paths, annotations, infos
    
    
    def insert_np_cam_calibration(self, filtered_infos):
        """
       Converts the intrinsic camera parameters in each dict of the given list into a numpy (3, 3) camera matrix
        Args:
            filtered_infos: List with all dictionaries containing the intrinsic camera parameters
        Returns:
            filtered_infos: List with all dictionaries containing the intrinsic camera parameters also as a numpy (3, 3) array
    
        """
        for info in filtered_infos:
            info["cam_K_np"] = np.reshape(np.array(info["cam_K"]), newshape = (3, 3))
        
        return filtered_infos
    
    
    def convert_gt(self, gt_list, info_list, mask_paths):
        """
       Prepares the annotations from the Linemod dataset format into the EfficientPose format
        Args:
            gt_list: List with all ground truth dictionaries in the dataset split
            info_list: List with all info dictionaries (intrinsic camera parameters) in the dataset split
            mask_paths: List with all segmentation mask paths in the dataset split
        Returns:
            all_annotations: List with the converted ground truth dictionaries
    
        """
        all_annotations = []
        for single_gt_list, info, mask_path in zip(gt_list, info_list, mask_paths):
            num_annos = len(single_gt_list)
            #init annotations in the correct base format.
            num_all_rotation_parameters = self.rotation_parameter + 2 #+1 for class id and +1 for is_symmetric flag
            annotations = {'labels': np.zeros((num_annos,)),
                           'bboxes': np.zeros((num_annos, 4)),
                           'rotations': np.zeros((num_annos, num_all_rotation_parameters)),
                           'translations': np.zeros((num_annos, self.translation_parameter)),
                           'translations_x_y_2D': np.zeros((num_annos, 2))}
            
            mask = cv2.imread(mask_path)

            for i, gt in enumerate(single_gt_list):
                #fill in the values
                #get the class label for the occlusion object id
                annotations["labels"][i] = self.object_ids_to_class_labels[gt["obj_id"]]
                #get bbox from mask
                annotations["bboxes"][i, :], found_object = self.get_bbox_from_mask(mask, mask_value = self.name_to_mask_value[self.class_to_name[self.object_ids_to_class_labels[gt["obj_id"]]]])
                if not found_object:
                    print("\nWarning: Did not find object in mask!")
                    # print(mask_path)
                    # print(gt["obj_id"])
                #transform rotation into the needed representation
                annotations["rotations"][i, :-2] = self.transform_rotation(np.array(gt["cam_R_m2c"]), self.rotation_representation)
                annotations["rotations"][i, -2] = float(self.is_symmetric_object(gt["obj_id"]))
                annotations["rotations"][i, -1] = float(self.object_ids_to_class_labels[gt["obj_id"]])
                
                annotations["translations"][i, :] = np.array(gt["cam_t_m2c"])
                annotations["translations_x_y_2D"][i, :] = self.project_points_3D_to_2D(points_3D = np.zeros(shape = (1, 3)), #transform the object origin point which is the centerpoint
                                                                                        rotation_vector = self.transform_rotation(np.array(gt["cam_R_m2c"]), "axis_angle"),
                                                                                        translation_vector = np.array(gt["cam_t_m2c"]),
                                                                                        camera_matrix = info["cam_K_np"])
            
            all_annotations.append(annotations)
        
        return all_annotations
    
    
    def convert_bboxes(self, bbox):
        """
       Convert bbox from (x1, y1, width, height) to (x1, y1, x2, y2) format
        Args:
            bbox: numpy array (x1, y1, width, height)
        Returns:
            new_bbox: numpy array (x1, y1, x2, y2)
    
        """
        #convert bbox from (x1, y1, width, height) to (x1, y1, x2, y2) format
        new_bbox = np.copy(bbox)
        new_bbox[2] += new_bbox[0]
        new_bbox[3] += new_bbox[1]
        
        return new_bbox
    
        
    def parse_yaml(self, object_path, filename = "gt.yml"):
        """
       Reads a yaml file
        Args:
            object_path: Path to the yaml file
            filename: filename of the yaml file
        Returns:
            yaml_dic: Dictionary containing the yaml file content
    
        """
        yaml_path = os.path.join(object_path, filename)
        
        if not os.path.isfile(yaml_path):
            print("Error: file {} does not exist!".format(yaml_path))
            return None
        
        with open(yaml_path) as fid:
            yaml_dic = yaml.safe_load(fid)
            
        return yaml_dic
        
    
    def check_path(self, path):
        """
        Check if the given path exists
        """
        if not os.path.exists(path):
            print("Error: path {} does not exist!".format(path))
            return False
        else:
            return True
        
        
    def parse_examples(self, data_file):
        """
       Reads the Linemod dataset split (train or test) txt file containing the examples of this split
        Args:
            data_file: Path to the dataset split file
        Returns:
            data_examples: List containing all data example id's of the used dataset split
    
        """
        if not os.path.isfile(data_file):
            print("Error: file {} does not exist!".format(data_file))
            return None
        
        with open(data_file) as fid:
            data_examples = [example.strip() for example in fid if example != ""]
            
        return data_examples
        
        
    #needed function implementations of the generator base class
    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_paths)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.class_to_name)

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        return label in self.class_to_name

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.name_to_class

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.name_to_class[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.class_to_name[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        #image aspect ratio is fixed on Occlusion dataset
        return 640. / 480.

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #load image and switch BGR to RGB
        image = cv2.imread(self.image_paths[image_index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_mask(self, image_index):
        """ Load mask at the image_index.
        """
        return cv2.imread(self.mask_paths[image_index])

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        return copy.deepcopy(self.annotations[image_index])
    
    def load_camera_matrix(self, image_index):
        """ Load intrinsic camera parameter for an image_index.
        """
        return np.copy(self.infos[image_index]["cam_K_np"])
        
    

if __name__ == "__main__":
    #test linemod generator
    train_gen = OcclusionGenerator("/Datasets/Linemod_preprocessed/", use_6DoF_augmentation = True, use_colorspace_augmentation = True)
    #test_gen = OcclusionGenerator("/Datasets/Linemod_preprocessed/", train = False, shuffle_dataset = False, shuffle_groups = False)
    
    for i in range(100):
        img, anno = train_gen[0]
    # img, anno = test_gen[0]
    
    