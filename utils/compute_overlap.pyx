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

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------
"""

cimport cython
import numpy as np
cimport numpy as np

cdef extern from "calc_min_distances.h":
    void c_min_distances(float *points_gt, float *points_pred, float *min_distances, int num_points_gt, int num_points_pred)


def compute_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def calc_min_distance_between_all_point_pairs(
    np.ndarray[float, ndim=2] points_gt,
    np.ndarray[float, ndim=2] points_pred
):
    """
    Args:
        points_gt: (N, 3) ndarray of float
        points_pred: (K, 3) ndarray of float

    Returns
        min_distances: (N,) ndarray of the minimal distances between a point of a and any other point of b
    """
    cdef unsigned int N = points_gt.shape[0]
    cdef unsigned int K = points_pred.shape[0]
    cdef np.ndarray[float, ndim = 1] min_distances = np.zeros((N,), dtype=np.float32)
    cdef int i
    cdef np.ndarray[float, ndim = 1] point_gt = np.zeros((3,), dtype = np.float32)
    cdef np.ndarray[float, ndim = 2] gt_row = np.zeros((K, 3), dtype = np.float32)
    
    for i in range(N):
        point_gt[:] = points_gt[i, :]
        gt_row[:, :] = np.tile(point_gt, (K, 1))
        min_distances[i] = np.min(np.linalg.norm(gt_row - points_pred, axis = -1))
        
    return min_distances


def wrapper_c_min_distances(points_gt, points_pred):
    """
    Python wrapper function to call the C calculation function
    Args:
        points_gt: (N, 3) ndarray of float containing the 3D points of an object, transformed with the ground truth 6D pose
        points_pred: (K, 3) ndarray of float containing the 3D points of an object, transformed with the predicted 6D pose

    Returns
        min_distances: (N,) ndarray of the minimal distances between a point of a and any other point of b
    """
    cdef int num_points_gt = points_gt.shape[0]
    cdef int num_points_pred = points_pred.shape[0]
    cdef np.ndarray[float, ndim=1, mode="c"] min_distances = np.zeros((num_points_gt,), dtype = np.float32, order = 'c')
    cdef np.ndarray[float, ndim=2, mode="c"] points_gt_c = points_gt.astype(np.float32, order = 'c')
    cdef np.ndarray[float, ndim=2, mode="c"] points_pred_c = points_pred.astype(np.float32, order = 'c')
    
    c_min_distances(&points_gt_c[0,0], &points_pred_c[0,0], &min_distances[0], num_points_gt, num_points_pred)
    
    return min_distances
