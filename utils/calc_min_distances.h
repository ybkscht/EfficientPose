/*
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
*/

#include <math.h>

float distance(float *p1, float *p2){
    float d1 = p1[0] - p2[0];
    float d2 = p1[1] - p2[1];
    float d3 = p1[2] - p2[2];
    
    return (float) sqrt(d1*d1 + d2*d2 + d3*d3);
}

void c_min_distances(float *points_gt, float *points_pred, float *min_distances, int num_points_gt, int num_points_pred){
    for(int idx_gt = 0; idx_gt < num_points_gt; idx_gt++){
        float *p_gt = points_gt + (idx_gt * 3);
        float current_min_distance = distance(p_gt, points_pred);
        for(int idx_pred = 1; idx_pred < num_points_pred; idx_pred++){
            float tmp_dist = distance(p_gt, points_pred + (idx_pred * 3));
            if(tmp_dist < current_min_distance){
                current_min_distance = tmp_dist;
            }
        }
        min_distances[idx_gt] = current_min_distance;
    }
}