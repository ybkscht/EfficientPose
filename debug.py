#!/usr/bin/env python

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

import argparse
import sys
import cv2

from generators.linemod import LineModGenerator
from generators.occlusion import OcclusionGenerator
from utils.visualization import draw_annotations, draw_boxes
from utils.anchors import anchors_for_shape, compute_gt_annotations


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Dataset debug script.')
    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True
    
    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help = 'Path to dataset directory (ie. /tmp/linemod).')
    linemod_parser.add_argument('--object-id', help = 'ID of the LINEMOD Object to train on', type = int, default = 8)

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help = 'Path to dataset directory (ie. /tmp/occlusion).')
    
    
    parser.add_argument('--rotation-representation', help = 'Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"', default = 'axis_angle')
    parser.add_argument('--anchors', help = 'Show positive anchors on the image.', action = 'store_true')
    parser.add_argument('--annotations', help = 'Show annotations on the image.', action = 'store_true')
    parser.add_argument('--draw-class-names', help = 'Show class annotations on the image.', action = 'store_true')
    parser.add_argument('--draw_2d-bboxes', help = 'Show 2D bounding box annotations on the image.', action = 'store_true')
    parser.add_argument('--no-color-augmentation', help = 'Do not use colorspace augmentation', action = 'store_true')
    parser.add_argument('--no-6dof-augmentation', help = 'Do not use 6DoF augmentation', action = 'store_true')
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))

    return parser.parse_args(args)


def main(args = None):
    """
    Creates dataset generator with the parsed input arguments and starts the dataset visualization
    Args:
        args: command line arguments
    
    """
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    run(generator, args)


def create_generator(args):
    """ Create the data generators.

    Args:
        args: parseargs arguments object.
        
    Returns:
        Generator
        
    """

    if args.dataset_type == 'linemod':
        generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = True,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            phi = args.phi,
        )
    elif args.dataset_type == 'occlusion':
        generator = OcclusionGenerator(
            args.occlusion_path,
            train = True,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            phi = args.phi,
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def run(generator, args):
    """ Main loop in which data is provided by the generator and then displayed

    Args:
        generator: The generator to debug.
        args: parseargs args object.
    """
    while True:
        # display images, one at a time
        for i in range(generator.size()):
            # load the data
            image       = generator.load_image(i)
            annotations = generator.load_annotations(i)
            mask = generator.load_mask(i)
            camera_matrix = generator.load_camera_matrix(i)
            if len(annotations['labels']) > 0 :
                # apply random transformations
                image, annotations = generator.random_transform_group_entry(image, annotations, mask, camera_matrix)
    
                anchors = anchors_for_shape(image.shape, anchor_params = None)
                positive_indices, _, max_indices = compute_gt_annotations(anchors[0], annotations['bboxes'])
                
                #switch image RGB to BGR again
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
                # draw anchors on the image
                if args.anchors:
                    draw_boxes(image, anchors[0][positive_indices], (255, 255, 0), thickness=1)
    
                # draw annotations on the image
                if args.annotations:
                    draw_annotations(image,
                                     annotations,
                                     class_to_bbox_3D = generator.get_bbox_3d_dict(),
                                     camera_matrix = camera_matrix,
                                     label_to_name = generator.label_to_name,
                                     draw_bbox_2d = args.draw_2d_bboxes,
                                     draw_name = args.draw_class_names)
        
                print("Generator idx: {}".format(i))
                
            cv2.imshow('Image', image)
            if cv2.waitKey() == ord('q'):
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    main()
