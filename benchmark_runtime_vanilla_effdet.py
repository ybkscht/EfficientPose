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

import tensorflow as tf

from model_vanilla_effdet import efficientdet
from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator


def main():
    """
    Measures the vanilla EfficientDet runtime on your machine.
    input_params:
            "phi": EfficientDet scaling hyperparameter phi,
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
    
    possible_datasets = ("linemod", "occlusion", "complete_linemod")
    
    #input parameter
    input_params = {"phi": 0,
                    #"dataset": "complete_linemod",
                    "dataset": "occlusion",
                    "object_id": 8,
                    "dataset_path": "/Datasets/Linemod_preprocessed/",
                    # "model_path": "./weights/phi_3/occlusion/phi_3_occlusion_best_ADD(-S).h5"
                    # "model_path": "./weights/phi_0/object_8/phi_0_linemod_best_ADD.h5"
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
            "phi": EfficientDet scaling hyperparameter phi,
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
        phi: EfficientDet scaling hyperparameter phi
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
        phi: EfficientDet scaling hyperparameter phi
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
    Measures the runtime of EfficientDet iteratively on all Linemod objects
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
        mean_preprocessing_time, mean_network_time, mean_end_to_end_time = benchmark(generator, model)
        
        benchmark_results[object_id] = {"mean_preprocessing_time": mean_preprocessing_time, 
                                        "mean_network_time": mean_network_time, 
                                        "mean_end_to_end_time": mean_end_to_end_time}
        
    mean_preprocessing_time, mean_network_time, mean_end_to_end_time = calc_complete_linemod_results(benchmark_results)
    print("\nAverage results on complete Linemod dataset:\n")
    print_results(mean_preprocessing_time, mean_network_time, mean_end_to_end_time)


def build_model(phi, model_path, generator):
    """
    Builds an EfficientDet model and init it with a given weight file
    Args:
        phi: EfficientDet scaling hyperparameter
        model_path: Path to the weight file
        generator: Dataset generator
        
    Returns:
        model: EfficientPose model

    """
    _, model, _ = efficientdet(phi,
                                num_classes = generator.num_classes(),
                                num_anchors = generator.num_anchors,
                                weighted_bifpn = True,
                                freeze_bn = True,
                                detect_quadrangle = False,
                                score_threshold = 0.5,
                                )
    
    model.load_weights(model_path, by_name = True)
    
    return model


def warmup(generator, model):
    """
    Perform a few predictions to make sure everythin is initialized so we really measure the correct time later
    Args:
        generator: Dataset generator
        model: EfficientDet model

    """
    num_warmup_iterations = 10
    for i in range(num_warmup_iterations):
        _ = single_prediction(model, generator, 0)
        
        
def benchmark(generator, model):
    """
    Benchmark the given model on the given dataset generator
    Args:
        generator: Dataset generator
        model: EfficientDet model
        
    Returns:
        The measured mean and std times

    """
    print("\nStarting benchmark...\n")
    
    preprocessing_times = []
    network_times = []
    end_to_end_times = []
    
    for i in tqdm(range(generator.size())):
        pre_time, net_time, end_time = single_prediction(model, generator, i)
        
        preprocessing_times.append(pre_time)
        network_times.append(net_time)
        end_to_end_times.append(end_time)
        
    mean_preprocessing_time, mean_network_time, mean_end_to_end_time  = calc_results(preprocessing_times, network_times, end_to_end_times)
    print_results(mean_preprocessing_time, mean_network_time, mean_end_to_end_time)
    
    return mean_preprocessing_time, mean_network_time, mean_end_to_end_time
    
    
def single_prediction(model, generator, i):
    """
    Perform a single inference step and measure the time
    Args:
        model: EfficientPose model
        generator: Dataset generator
        i: The generator iteration step
        
    Returns:
        The measured times of this single inference step

    """
    score_threshold = 0.5
    image = generator.load_image(i)
    
    start_end_to_end = time.time()

    image, scale = generator.preprocess_image(image)
    
    image_batch = np.expand_dims(image, axis=0)
    input_list = [image_batch]
    
    preprocessing_time = time.time() - start_end_to_end
    
    # run network
    start_network = time.time()   
    boxes, scores, labels = model.predict_on_batch(input_list)[:3]
    network_time = time.time() - start_network
    
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    
    # correct boxes for image scale
    boxes /= scale   
    
    indices = np.where(scores[:] > score_threshold)
    # select those scores
    scores = scores[indices]

    # select detections
    image_boxes      = boxes[indices, :]
    image_labels     = labels[indices]
    
    end_to_end_time = time.time() - start_end_to_end
    
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
    mean_preprocessing_time = sum(preprocessing_times) / len(preprocessing_times)
    mean_network_time = sum(network_times) / len(network_times)
    mean_end_to_end_time = sum(end_to_end_times) / len(end_to_end_times)  

    return mean_preprocessing_time, mean_network_time, mean_end_to_end_time      


def print_results(mean_preprocessing_time, mean_network_time, mean_end_to_end_time):
    """
    Print the benchmark results

    """    
    print("\n\n\nMean time for preprocessing: {}s".format(mean_preprocessing_time))
    print("Mean time for network forward propagation: {}s".format(mean_network_time))
    print("Mean FPS for network forward propagation: {}".format(1. / mean_network_time))
    print("Mean time for end-to-end: {}s".format(mean_end_to_end_time))
    print("Mean FPS for end-to-end: {}".format(1. / mean_end_to_end_time))


def calc_complete_linemod_results(benchmark_results):
    """
    Calculates the overall mean and std of all separate Linemod object benchmarks
    Args:
        benchmark_results: Dictionary containing the mean and std times of the single Linemod object benchmarks
    Returns:
        The overall mean and std of the measured times

    """
    preprocessing_times = [result["mean_preprocessing_time"] for result in benchmark_results.values()]
    network_times = [result["mean_network_time"] for result in benchmark_results.values()]
    end_to_end_times = [result["mean_end_to_end_time"] for result in benchmark_results.values()]
    
    return calc_results(preprocessing_times, network_times, end_to_end_times)
    

if __name__ == '__main__':
    main()
