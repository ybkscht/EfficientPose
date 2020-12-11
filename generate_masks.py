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

import cv2
import os
import numpy as np
from tqdm import tqdm

#Please note that you do not need this script if you downloaded the ready-to-use dataset from here https://drive.google.com/drive/folders/1Q8ezhQdKKTymDaN1BBiwGoiW_L3RZY9o?usp=sharing

#This script is needed to prepare the Occlusion dataset masks for the EfficientPose generator.
#Download the Linemod dataset from here https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7 and unzip it.
#Download the Occlusion labels from here https://drive.google.com/file/d/1PItmDj7Go0OBnC1Lkvagz3RRB9qdJUIG/view?usp=sharing and unzip them.
#copy the 'masks' and 'valid_poses' folders from the unzipped Occlusion labels to your unzipped Linemod dataset in Linemod_preprocessed/data/02.
#copy this script in Linemod_preprocessed/data/02 and execute it

masks_path = "./masks"
merged_masks_path = "./merged_masks"
os.makedirs(merged_masks_path, exist_ok = True)

name_to_mask_value = {"ape": 21,
                        "can": 106,
                        "cat": 128,
                        "driller": 170,
                        "duck": 191,
                        "eggbox": 213,
                        "glue": 234,
                        "holepuncher": 255}

file_list = [(filename, int(filename.replace(".png", ""))) for filename in os.listdir(os.path.join(masks_path, "ape")) if ".png" in filename]

for filename, example_id in tqdm(file_list):
    merged_mask = np.zeros((480, 640, 3), dtype = np.uint8)
    for object_name, mask_value in name_to_mask_value.items():
        subdir_path = os.path.join(masks_path, object_name)
        single_mask_path = os.path.join(subdir_path, filename)
        single_mask = cv2.imread(single_mask_path)
        merged_mask[single_mask != 0] = mask_value
        
    merged_filename = "{:04d}.png".format(example_id)
    cv2.imwrite(os.path.join(merged_masks_path, merged_filename), merged_mask)
        
        


