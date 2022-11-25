
import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm

from data.dataset_wrapper import generate_segmentation_line


def thickness_metric(distances, threshold=0.2):
    mean = np.mean(distances)
    std_dev = np.std(distances)
    if threshold is None:
        return mean, std_dev
    num_threshold = np.count_nonzero(np.asarray(distances) <= threshold)
    pct = num_threshold / len(distances)
    return mean, std_dev, pct


def thickness_percentage_differences(predictions_thick, annotations_thick, num_std_points, bodypart_dict, bodypart_segmentation_func, bbox_func, ids, thickness_vectors, bodyparts_no_center=[]):
    """
    Thickness accuracy
    @param predictions_thick: shape [num_examples, num_keypoints, 3]
    @param annotations_thick: shape [num_examples, num_keypoints, 3]
    @param num_std_points: first points are treated as standard and not included in this metric
    @param bodypart_dict: dictionary bodypart_segmentation index -> keypoint indices from bodypart between these points
    @param bodypart_segmentation_func: function returning segmentation masks with id as input
    @param bbox_func: function returning bounding box for segmentation mask in x1, y1, x2, y2
    @param ids: image ids in same order than predictions and annotations
    @return:    - the distance between the generated and the detected point in the image
                - the distance between the desired thickness of the thickness vector and the detected point
                -
    """
    distances = []
    distances_vec = []
    for index in tqdm(range(predictions_thick.shape[0]), desc="Thickness Errors", position=0):  # sadly, this is not vectorizable
        pred, anno = predictions_thick[index], annotations_thick[index]

        p_round = np.asarray(np.round(pred[:, :2]), dtype=np.int32)
        a_round = np.asarray(np.round(anno[num_std_points:, :2]), dtype=np.int32)

        if bbox_func is not None:
            bbr = bbox_func(ids[index])
            bbr = np.array(bbr).astype(int)
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            bodypart_map = bodypart_segmentation_func(ids[index], (int(x2 - x1), int(y2 - y1)))

            a_round[:, 0] -= x1
            a_round[:, 1] -= y1

            a_round = a_round[anno[num_std_points:, 2] != 0]

            p_round[:, 0] -= x1
            p_round[:, 1] -= y1
        else:
            bodypart_map = bodypart_segmentation_func(ids[index])
            x1, y1 = 0, 0
            y2, x2 = bodypart_map.shape

        bodypart_ind = bodypart_map[a_round[:, 1], a_round[:, 0]]

        for i in range(num_std_points, predictions_thick.shape[1]):
            if np.sum(annotations_thick[index, i]) <= 0:
                continue
            bodypart_id = bodypart_ind[i - num_std_points]
            if bodypart_id not in bodypart_dict:
                bodypart_area = bodypart_map[a_round[i-num_std_points, 1] - 5:a_round[i-num_std_points, 1] + 6, a_round[i-num_std_points, 0] - 5:a_round[i-num_std_points, 0] + 6]
                counts = np.bincount(bodypart_area.flatten())
                count_max = np.argmax(counts)
                if count_max != 0:
                    bodypart_ind[i] = count_max
                else:
                    raise RuntimeError("did not find bodypart from annotation")
            bodypart_mask = np.zeros_like(bodypart_map)
            bodypart_mask[np.where(bodypart_map == bodypart_ind[i-num_std_points])] = 1
            bodypart = bodypart_dict[bodypart_ind[i-num_std_points]]
            keypoints = anno[[bodypart[0], bodypart[1]]]
            keypoints_box = keypoints.copy()
            keypoints_box[:, 0] -= x1
            keypoints_box[:, 1] -= y1
            if (keypoints[0][1] - keypoints[1][1]) != 0:
                m1 = - (keypoints[0][0] - keypoints[1][0]) / (keypoints[0][1] - keypoints[1][1])
                t1_a = anno[i, 1] - m1 * anno[i, 0]
                t1_p = pred[i, 1] - m1 * pred[i, 0]
                if np.abs(m1) > 1e-4:
                    m2 = - 1 / m1
                    t2 = keypoints[0][1] - m2 * keypoints[0][0]

                    projection_a = [(t1_a - t2) / (m2 - m1)]
                    projection_a.append(m1 * projection_a[0] + t1_a)
                    projection_p = [(t1_p - t2) / (m2 - m1)]
                    projection_p.append(m1 * projection_p[0] + t1_p)

                else:
                    projection_a = [keypoints[0][0], anno[i, 1]]
                    projection_p = [keypoints[0][0], pred[i, 1]]
            else:
                projection_a = [anno[i, 0], keypoints[0][1]]
                projection_p = [pred[i, 0], keypoints[0][1]]
                m1 = float('inf')
                t1_a, t1_p = 0, 0

            x_a, y_a = generate_segmentation_line(m1, t1_a, (x1, y1, x2, y2), np.asarray(projection_a))
            x_p, y_p = generate_segmentation_line(m1, t1_p, (x1, y1, x2, y2), np.asarray(projection_p))
            x_a -= x1
            y_a -= y1
            x_p -= x1
            y_p -= y1

            projection_a[0] -= x1
            projection_a[1] -= y1

            no_center = bodypart_id in bodyparts_no_center

            mask_out_a = np.where(bodypart_mask[y_a, x_a])
            if len(mask_out_a[0]) == 0:  # might happen because of rounding issues, then we cannot calculate this metric
                continue

            p1_a = [x_a[mask_out_a[0][0]], y_a[mask_out_a[0][0]]]
            p2_a = [x_a[mask_out_a[0][-1]], y_a[mask_out_a[0][-1]]]
            test_points = np.asarray([p1_a, projection_a, p2_a])
            anno_point = anno[i][:2].copy()
            anno_point[0] -= x1
            anno_point[1] -= y1
            dist_a = distance_matrix(test_points, anno_point[None])[:, 0]
            if no_center:
                dist_total_a = dist_a[0] + dist_a[2]
                thickness_a = dist_a[0] / (dist_total_a + 1e-10)
                thickness_vec = thickness_vectors[index][i, 2] if thickness_vectors is not None else None
            else:
                dist_total_a = min(dist_a[0], dist_a[2]) + dist_a[1]
                thickness_a = dist_a[1] / (dist_total_a + 1e-10)
                thickness_vec = 1 - thickness_vectors[index][i, 1] if thickness_vectors is not None else None

            if len(x_p) == 0:
                dist, dist_vec = 2, 2
            else:
                mask_out_p = np.where(bodypart_mask[y_p, x_p])

                if len(mask_out_p[0]) == 0:
                    dist = 2
                    dist_vec = 2
                else:
                    p1_p = [x_p[mask_out_p[0][0]] , y_p[mask_out_p[0][0]] ]
                    p2_p = [x_p[mask_out_p[0][-1]] , y_p[mask_out_p[0][-1]] ]
                    projection_p[0] -= x1
                    projection_p[1] -= y1

                    pred_point = pred[i][:2].copy()
                    pred_point[0] -= x1
                    pred_point[1] -= y1

                    test_points = np.asarray([p1_p, projection_p, p2_p])
                    dist_p = distance_matrix(test_points, pred_point[None, :])[:, 0]
                    if no_center:
                        dist_total_p = dist_p[0] + dist_p[2]
                    else:
                        dist_total_p = min(dist_p[0], dist_p[2]) + dist_p[1]

                    if no_center:
                        thickness_p = dist_p[0] / dist_total_p
                        dist = np.abs(thickness_a - thickness_p) * 2
                        dist_vec = np.abs(thickness_p - thickness_vec) * 2 if thickness_vectors is not None else None
                    elif dist_a[0] < dist_a[2] and dist_p[0] > dist_p[2] or dist_a[0] > dist_a[2] and dist_p[0] < dist_p[2]:
                        dist = thickness_a + dist_p[1] / dist_total_p
                        dist_vec = thickness_vec + dist_p[1] / dist_total_p if thickness_vectors is not None else None
                    else:
                        thickness_p = dist_p[1] / dist_total_p
                        dist = np.abs(thickness_a - thickness_p)
                        dist_vec = np.abs(thickness_p - thickness_vec) if thickness_vectors is not None else None
            distances.append(dist)
            if thickness_vectors is not None:
                distances_vec.append(dist_vec)
    return distances, distances_vec


def format_thickness_result(mean, std, pct, prefix=""):
    print("{} - Thickness results: mean {:.2f}, std deviation {:.2f}, pct {:.2f}".format(prefix, mean, std, pct))
    print("{:.1f} & {:.1f} & {:.1f}".format(mean*100, pct*100, std))