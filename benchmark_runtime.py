"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
"""

import numpy as np
import os
import time
from tqdm import tqdm
import math

import tensorflow as tf

from model import build_EfficientPose
from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator


def main():
    """
    Measures EfficientPose runtime on your machine.
    input_params:
            "phi": EfficientPose scaling hyperparameter phi,
            "dataset": On which dataset should the runtime be measured. Use one of the following ("linemod", "occlusion", "complete_linemod", "occlusion_different_number_instances")
                        "linemod": a single object of Linemod is used.
                        "occlusion": the occlusion dataset including all 8 objects is used
                        "complete_linemod": Benchmark the complete Linemod dataset. Therefore you need all weight files stored as follows model_path/object_X/phi_Y_linemod_best_ADD{-S if the object is symmetric}).h5
                        "occlusion_different_number_instances": iteratively measures the runtime from 1 to 8 objects on the Occlusion dataset via deleting objects in the image using the segmentation masks to match the right number of objects per image
            "object_id": in case of Linemod this is the id of the Linemod object. If not you can ignore this parameter
            "dataset_path": Path to the dataset
            "model_path": Path to the EfficientPose weight file

    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()
    
    possible_datasets = ("linemod", "occlusion", "complete_linemod", "occlusion_different_number_instances")
    
    #input parameter
    input_params = {"phi": 0,
                    "dataset": "occlusion_different_number_instances",
                    "object_id": 8, #this parameter is not used if you use Occlusion
                    "dataset_path": "/Datasets/Linemod_preprocessed/",
                    # "model_path": "./weights/phi_3/occlusion/phi_3_occlusion_best_ADD(-S).h5"
                    "model_path": "./weights/phi_0_occlusion_best_ADD(-S).h5"
                    # "model_path": "./weights/phi_3/"
        }
    
    if input_params["dataset"] not in possible_datasets:
        print("Error: given dataset {} is not a valid dataset. Choose one of the following: {}".format(input_params["dataset"], possible_datasets))
        return
    
    #start runtime benchmark on the chosen dataset
    benchmark_dataset(**input_params)
    
    
def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)
    
    
def benchmark_dataset(**kwargs):
    """
    Get the right generator and start the given benchmark
    input_params:
            "phi": EfficientPose scaling hyperparameter phi,
            "dataset": On which dataset should the runtime be measured. Use one of the following ("linemod", "occlusion", "complete_linemod", "occlusion_different_number_instances")
                        "linemod": a single object of Linemod is used.
                        "occlusion": the occlusion dataset including all 8 objects is used
                        "complete_linemod": Benchmark the complete Linemod dataset. Therefore you need all weight files stored as follows model_path/object_X/phi_Y_linemod_best_ADD{-S if the object is symmetric}).h5
                        "occlusion_different_number_instances": iteratively measures the runtime from 1 to 8 objects on the Occlusion dataset via deleting objects in the image using the segmentation masks to match the right number of objects per image
            "object_id": in case of Linemod this is the id of the Linemod object. If not you can ignore this parameter
            "dataset_path": Path to the dataset
            "model_path": Path to the EfficientPose weight file

    """
    phi = kwargs["phi"]
    dataset = kwargs["dataset"]
    object_id = kwargs["object_id"]
    dataset_path = kwargs["dataset_path"]
    model_path = kwargs["model_path"]
    
    if dataset == "linemod":
        generator = create_linemod_generator(phi, object_id, dataset_path)
    elif dataset == "occlusion":
        generator = create_occlusion_generator(phi, dataset_path)
    elif dataset == "complete_linemod":
        benchmark_complete_linemod(phi, dataset_path, model_path)
        return
    elif dataset == "occlusion_different_number_instances":
        benchmark_occlusion_diff_num_instances(phi, dataset_path, model_path)
        return
    else:
        print("\nError: Unkown dataset {}".format(dataset))
        return
    
    model = build_model(phi, model_path, generator)
    
    #perform a few predictions to make sure everything is initialized to measure the real inference times later
    warmup(generator, model)
    
    results = benchmark(generator, model)


def create_linemod_generator(phi, object_id, dataset_path):
    """
    Create Linemod generator
    Args:
        phi: EfficientPose scaling hyperparameter phi
        object_id: ID of the Linemod object
        dataset_path: Path to the dataset
    
    Returns:
        The generator
    """
    common_args = {
        'batch_size': 1,
        'phi': phi,
    }
    
    generator = LineModGenerator(
            dataset_path,
            object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = "axis_angle",
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    
    return generator


def create_occlusion_generator(phi, dataset_path):
    """
    Create Occlusion generator
    Args:
        phi: EfficientPose scaling hyperparameter phi
        dataset_path: Path to the dataset
    
    Returns:
        The generator
    """
    common_args = {
        'batch_size': 1,
        'phi': phi,
    }
    
    generator = OcclusionGenerator(
            dataset_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = "axis_angle",
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    
    return generator


def benchmark_complete_linemod(phi, dataset_path, all_models_path):
    """
    Measures the runtime of EfficientPose iteratively on all Linemod objects
    Args:
        phi: EfficientPose scaling hyperparameter phi
        dataset_path: Path to the dataset
        all_models_path: Path to all weight files stored as follows all_models_path/object_X/phi_Y_linemod_best_ADD{-S if the object is symmetric}).h5

    """
    linemod_object_ids = (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)
    benchmark_results = {}
    
    for idx, object_id in enumerate(linemod_object_ids):
        generator = create_linemod_generator(phi, object_id, dataset_path)
        
        if generator.is_symmetric_object(object_id):
            weight_sub_path = "object_{}/phi_{}_linemod_best_ADD-S.h5".format(object_id, phi)
        else:
            weight_sub_path = "object_{}/phi_{}_linemod_best_ADD.h5".format(object_id, phi)
        model_path = os.path.join(all_models_path, weight_sub_path)
        
        if idx <= 0:
            model = build_model(phi, model_path, generator)
        else:
            model.load_weights(model_path, by_name = True)
        
        #perform a few predictions to make sure everything is initialized to measure the real inference times later
        warmup(generator, model)
        print("\n\nBenchmarking object {}...\n".format(object_id))
        mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time = benchmark(generator, model)
        
        benchmark_results[object_id] = {"mean_preprocessing_time": mean_preprocessing_time, 
                                        "std_preprocessing_time": std_preprocessing_time,
                                        "mean_network_time": mean_network_time, 
                                        "std_network_time": std_network_time, 
                                        "mean_end_to_end_time": mean_end_to_end_time,
                                        "std_end_to_end_time": std_end_to_end_time}
        
    mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time = calc_complete_linemod_results(benchmark_results)
    print("\nAverage results on complete Linemod dataset:\n")
    print_results(mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time)


def benchmark_occlusion_diff_num_instances(phi, dataset_path, model_path):
    """
    Measures the runtime of EfficientPose iteratively on Occlusion from 1 to 8 objects per image
    Args:
        phi: EfficientPose scaling hyperparameter phi
        dataset_path: Path to the dataset
        model_path: Path to the weight file
    
    """
    max_objects = 8
    benchmark_results = {}
    
    generator = create_occlusion_generator(phi, dataset_path)
    model = build_model(phi, model_path, generator)
    
    for num_objects in range(1, max_objects + 1):
        #perform a few predictions to make sure everything is initialized to measure the real inference times later
        warmup(generator, model)
        
        print("\n\nBenchmarking {} objects...\n".format(num_objects))
        mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time = benchmark(generator, model, number_of_objects = num_objects)
        
        benchmark_results[num_objects] = {"mean_preprocessing_time": mean_preprocessing_time, 
                                        "mean_network_time": mean_network_time, 
                                        "mean_end_to_end_time": mean_end_to_end_time}
        

def build_model(phi, model_path, generator):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        phi: EfficientPose scaling hyperparameter
        model_path: Path to the weight file
        generator: Dataset generator
        
    Returns:
        model: EfficientPose model

    """
    _, model, _ = build_EfficientPose(phi,
                                      num_classes = generator.num_classes(),
                                      num_anchors = generator.num_anchors,
                                      freeze_bn = True,
                                      score_threshold = 0.5,
                                      num_rotation_parameters = generator.get_num_rotation_parameters(),
                                      print_architecture = False)
    
    model.load_weights(model_path, by_name = True)
    
    return model


def warmup(generator, model):
    """
    Perform a few predictions to make sure everythin is initialized so we really measure the correct time later
    Args:
        generator: Dataset generator
        model: EfficientPose model

    """
    num_warmup_iterations = 10
    for i in range(num_warmup_iterations):
        _ = single_prediction(model, generator, 0)
        
        
def benchmark(generator, model, number_of_objects = None):
    """
    Benchmark the given model on the given dataset generator
    Args:
        generator: Dataset generator
        model: EfficientPose model
        number_of_objects: In case of "occlusion_different_number_instances" the generator deletes all objects needed to match this given number.
        
    Returns:
        The measured mean and std times

    """
    print("\nStarting benchmark...\n")
    
    preprocessing_times = []
    network_times = []
    end_to_end_times = []
    
    for i in tqdm(range(generator.size())):
        pre_time, net_time, end_time = single_prediction(model, generator, i, number_of_objects)
        
        preprocessing_times.append(pre_time)
        network_times.append(net_time)
        end_to_end_times.append(end_time)
        
    mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time  = calc_results(preprocessing_times, network_times, end_to_end_times)
    print_results(mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time)
    
    return mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time


def single_prediction(model, generator, i, number_of_objects = None):
    """
    Perform a single inference step and measure the time
    Args:
        model: EfficientPose model
        generator: Dataset generator
        i: The generator iteration step
        number_of_objects: In case of "occlusion_different_number_instances" the generator deletes all objects needed to match this given number.
        
    Returns:
        The measured times of this single inference step

    """
    score_threshold = 0.5
    image = generator.load_image(i)
    camera_matrix = generator.load_camera_matrix(i)
    
    if number_of_objects is not None:
        image, not_enough_objects = fix_object_number_in_image(image, generator, i, number_of_objects)
        if not_enough_objects:
            return None, None, None
    
    start_end_to_end = time.time()

    image, scale = generator.preprocess_image(image)
    camera_input = generator.get_camera_parameter_input(camera_matrix, scale, generator.translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    preprocessing_time = time.time() - start_end_to_end
    
    # run network
    start_network = time.time()   
    boxes, scores, labels, rotations, translations = model.predict_on_batch(input_list)[:5]
    network_time = time.time() - start_network
    
    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    
    # correct boxes for image scale
    boxes /= scale   
    #rescale rotations
    rotations *= math.pi
    
    indices = np.where(scores[:] > score_threshold)
    # select those scores
    scores = scores[indices]

    # select detections
    image_boxes      = boxes[indices, :]
    image_rotations  = rotations[indices, :]
    image_translations = translations[indices, :]
    image_labels     = labels[indices]
    
    end_to_end_time = time.time() - start_end_to_end
    
    if number_of_objects is not None:
        #check if the expected number of objects were detected
        num_detected_objects = image_labels.size
        if num_detected_objects != number_of_objects:
            #print("There are {} objects on the image but only {} were detected. Skipping time.".format(number_of_objects, num_detected_objects))
            return None, None, None
                
    
    return preprocessing_time, network_time, end_to_end_time


def calc_results(preprocessing_times, network_times, end_to_end_times):
    """
    Calculates the mean and std of the measured times
    Args:
        preprocessing_times: List containing all preprocessing times
        network_times: List containing all network forward propagation times
        end_to_end_times: List containing all end-to-end times including preprocessing, network forward propagation and postprocessing
    Returns:
        The mean and std of the measured times

    """
    print("\nAll end-to-end-times: ", len(end_to_end_times))
    preprocessing_times = [t for t in preprocessing_times if t is not None]
    network_times = [t for t in network_times if t is not None]
    end_to_end_times = [t for t in end_to_end_times if t is not None]
    print("After filtering out times with wrong number of detected objects: ", len(end_to_end_times))
    
    mean_preprocessing_time = sum(preprocessing_times) / len(preprocessing_times)
    std_preprocessing_time = sum([abs(t - mean_preprocessing_time) for t in preprocessing_times]) / len(preprocessing_times)
    
    mean_network_time = sum(network_times) / len(network_times)
    std_network_time = sum([abs(t - mean_network_time) for t in network_times]) / len(network_times)
    
    mean_end_to_end_time = sum(end_to_end_times) / len(end_to_end_times)  
    std_end_to_end_time = sum([abs(t - mean_end_to_end_time) for t in end_to_end_times]) / len(end_to_end_times)

    return mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time      


def print_results(mean_preprocessing_time, std_preprocessing_time, mean_network_time, std_network_time, mean_end_to_end_time, std_end_to_end_time):    
    """
    Print the benchmark results

    """
    print("\n\n\nMean time for preprocessing: {}s".format(mean_preprocessing_time))
    print("Mean FPS for preprocessing: {}".format(1. / mean_preprocessing_time))
    print("Std time for preprocessing: {}s".format(std_preprocessing_time))
    print("Mean time for network forward propagation: {}s".format(mean_network_time))
    print("Mean FPS for network forward propagation: {}".format(1. / mean_network_time))
    print("Std time for network forward propagation: {}s".format(std_network_time))
    print("Mean time for end-to-end: {}s".format(mean_end_to_end_time))
    print("Mean FPS for end-to-end: {}".format(1. / mean_end_to_end_time))
    print("Std time for end-to-end: {}s".format(std_end_to_end_time))


def calc_complete_linemod_results(benchmark_results):
    """
    Calculates the overall mean and std of all separate Linemod object benchmarks
    Args:
        benchmark_results: Dictionary containing the mean and std times of the single Linemod object benchmarks
    Returns:
        The overall mean and std of the measured times

    """
    mean_preprocessing_times = [result["mean_preprocessing_time"] for result in benchmark_results.values()]
    std_preprocessing_times = [result["std_preprocessing_time"] for result in benchmark_results.values()]
    
    mean_network_times = [result["mean_network_time"] for result in benchmark_results.values()]
    std_network_times = [result["std_network_time"] for result in benchmark_results.values()]
    
    mean_end_to_end_times = [result["mean_end_to_end_time"] for result in benchmark_results.values()]
    std_end_to_end_times = [result["std_end_to_end_time"] for result in benchmark_results.values()]
    
    mean_preprocess = sum(mean_preprocessing_times) / len(mean_preprocessing_times)
    std_preprocess = sum(std_preprocessing_times) / len(std_preprocessing_times)
    
    mean_network = sum(mean_network_times) / len(mean_network_times)
    std_network = sum(std_network_times) / len(std_network_times)
    
    mean_end_to_end = sum(mean_end_to_end_times) / len(mean_end_to_end_times)
    std_end_to_end = sum(std_end_to_end_times) / len(std_end_to_end_times)
    
    return mean_preprocess, std_preprocess, mean_network, std_network, mean_end_to_end, std_end_to_end


def delete_object_from_image(image, mask, bbox, mask_value):
    """
    Removes an object from the image using it's segmentation mask
    Args:
        image: The image
        mask: The segmentation mask
        bbox: numpy array [4] of the object's 2D bounding box
        mask_value: The segmentation mask value of the object to remove
    Returns:
        image: The image without the object
        mask: The mask without the object

    """
    bbox = list(map(int, [bbox[i] for i in range(bbox.size)]))
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    original_image = image.copy()
    #delete complete object bbox from image
    image[y1 : y2, x1 : x2, :] = 0
    
    #restore deleted pixels from other objects
    image[mask != 0] = original_image[mask != 0]
    image[mask == mask_value] = 0
    
    #delete object from mask
    mask[mask == mask_value] = 0
    
    #test
    # cv2.imshow("original", original_image)
    # cv2.imshow("deleted", image)
    # cv2.waitKey(0)
    
    return image, mask


def fix_object_number_in_image(image, generator, i, number_of_objects):
    """
    Removes all objects in the given image to match a given number of objects per image
    Args:
        image: The image
        generator: The dataset generator
        i: The dataset generator iteration step
        number_of_objects: The number of objects an image should contain
    Returns:
        image: The image with the given number of objects
        not_enough_objects: Boolean indicating if the image did not contain enough objects per image to match the given number of objects.

    """
    annotations = generator.load_annotations(i)
    mask = generator.load_mask(i)
    class_to_name = generator.class_to_name
    name_to_mask = generator.name_to_mask_value
    
    classes = annotations["labels"]
    num_classes = classes.size
    
    not_enough_objects = False
    if num_classes < number_of_objects:
        not_enough_objects = True
        return image, not_enough_objects
    elif num_classes == number_of_objects:
        return image, not_enough_objects
    
    num_objects_to_delete = num_classes - number_of_objects
    
    for i in range(num_objects_to_delete):
        object_to_delete = classes[i]
        bbox = annotations["bboxes"][i, :]
        mask_value = name_to_mask[class_to_name[object_to_delete]]
        
        image, mask = delete_object_from_image(image, mask, bbox, mask_value)
        
    return image, not_enough_objects
   

if __name__ == '__main__':
    main()
