
import pickle

import cv2
import numpy as np

from utils.keypoint_generator import get_intermediate_point


class DataWrapper:

    def __init__(self, params):
        self.annotations = None
        self.image_paths = None
        self.has_annotations = True
        self.image_ids = None
        self.name = None
        self.bboxes = None
        if "seg_representation_type" in params:
            self.used_bodyparts = None
            self.bodypart_dict = None
            self.bodypart_masks = None

    def load_keypoints(self, image_id, generate_on_segmentations=None):
        annos = self.annotations[image_id] if getattr(self, "has_annotations", True) else np.zeros((1, self.get_num_keypoints(), 3))
        if generate_on_segmentations is not None:
            annos = self.generate_additional_kps_on_segmentations(image_id, generate_on_segmentations)
        return annos

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        path = self.image_paths[image_id]
        image = cv2.imread(path)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_bbox(self, image_id):
        if hasattr(self, "bboxes"):
            return self.bboxes[image_id]
        else:
            raise NotImplementedError

    def load_bodypart_mask(self, image_id, size=None, scale_mask=1):
        segmentation_masks = self.bodypart_masks[image_id]
        segmentation_masks = cv2.imread(segmentation_masks, cv2.IMREAD_UNCHANGED) * scale_mask
        return segmentation_masks

    def get_num_keypoints(self):
        raise NotImplementedError

    def generate_additional_kps_on_segmentations(self, image_id, num):
        keypoints = self.annotations[image_id]
        image = self.load_image(image_id)
        bbr = self.bboxes[image_id]
        bbr = np.array(bbr).astype(int)

        # Indicates if the mask has the size of the whole image or the size of the bounding box
        full_image_mask = getattr(self, "full_image_mask", False)
        # Ids of body parts that are not split into left and right
        bodypart_ids_no_center = getattr(self, "bodypart_ids_no_center", [])

        if not full_image_mask:
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min([x2, image.shape[1]])
            y2 = min([y2, image.shape[0]])
        else:
            x1, y1 = 0, 0
            y2, x2 = image.shape[:2]
        if num > 0:
            resized_masks = self.load_bodypart_mask(image_id, (int(x2 - x1), int(y2 - y1)))

        p = [1. / len(self.used_bodyparts) for _ in range(len(self.used_bodyparts))]
        count = 0
        tries = 0
        existent_bodyparts = self.used_bodyparts.copy()

        new_annotations = np.zeros((num, 3), dtype=np.float32)
        new_annotations[:, 2] = 1

        kp_vector = np.zeros((num, self.get_num_keypoints()), dtype=np.float32)
        thickness_vector = np.zeros((num, 3))

        while count < num and tries < num * 2:
            tries += 1
            r = np.random.choice(range(len(existent_bodyparts)), 1, replace=True, p=p)
            bodypart = existent_bodyparts[r[0]]
            mask_ind = np.where(resized_masks == bodypart[2])
            if keypoints[bodypart[0]][2] == 0 or keypoints[bodypart[1]][2] == 0 or len(mask_ind[0]) == 0:
                existent_bodyparts.remove(bodypart)
                p = [1. / len(existent_bodyparts) for _ in range(len(existent_bodyparts))]
                if len(existent_bodyparts) == 0:
                    break
                continue

            percentage_projection = np.random.rand()
            percentage_thickness = np.random.normal(0, 0.3)
            if percentage_thickness < 0:
                percentage_thickness = -(1 + percentage_thickness)
                percentage_thickness = min(0, percentage_thickness)
            else:
                percentage_thickness = 1 - percentage_thickness
                percentage_thickness = max(0, percentage_thickness)

            use_central_projection = not (bodypart[-1] in bodypart_ids_no_center)
            if not use_central_projection:
                percentage_thickness += 1
                percentage_thickness /= 2
                if percentage_thickness < 0:
                    percentage_thickness = 0

            x1_, y1_, x2_, y2_ = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            new_point, vis_res = self.get_segmentation_keypoint(keypoints, bodypart, resized_masks, percentage_thickness, percentage_projection, (x1_, y1_, x2_, y2_), use_central_projection=use_central_projection, crop_region=True)

            if new_point is None:
                continue

            new_annotations[count, :2] = new_point[:2]


            kp_vector[count, bodypart[0]] = 1 - percentage_projection
            kp_vector[count, bodypart[1]] = percentage_projection
            if percentage_thickness < 0:  # not use central procetion: percentage thickness can not be < 0
                percentage_thickness = np.abs(percentage_thickness)
                thickness_vector[count, 0] = percentage_thickness
                thickness_vector[count, 1] = 1 - percentage_thickness
            elif use_central_projection:
                thickness_vector[count, 1] = 1 - percentage_thickness
                thickness_vector[count, 2] = percentage_thickness
            else:
                thickness_vector[count, 0] = 1 - percentage_thickness
                thickness_vector[count, 2] = percentage_thickness

            count += 1

        if num > 0:
            keypoints = np.concatenate([keypoints, new_annotations[:count]])

        kp_vector = kp_vector[:count]
        original_kp_kp_vector = np.zeros((self.get_num_keypoints(), self.get_num_keypoints()), dtype=np.float32)
        original_kp_kp_vector[range(self.get_num_keypoints()), range(self.get_num_keypoints())] = 1
        kp_vector = np.concatenate([original_kp_kp_vector, kp_vector], dtype=np.float32)

        thickness_vector = thickness_vector[:count]
        original_kp_thickness_vector = np.zeros((self.get_num_keypoints(), 3), dtype=np.float32)
        original_kp_thickness_vector[:, 1] = 1
        thickness_vector = np.concatenate([original_kp_thickness_vector, thickness_vector], dtype=np.float32)

        representations = {"kp_vector": kp_vector, "thickness": thickness_vector}

        if num > 0:
            keypoints = keypoints[None, :, :]
        return keypoints, representations

    def create_bodypart_map(self):
        raise NotImplementedError

    @staticmethod
    def get_segmentation_keypoint(keypoints, bodypart, resized_masks, percentage_thickness, percentage_projection, bbox, use_central_projection=True, crop_region=False,
                                  projection_point_outside_mask=True):

        x1, y1, x2, y2 = bbox
        mask = np.zeros_like(resized_masks)
        ys, xs = np.where(resized_masks == bodypart[2])
        mask[ys, xs] = 1
        if crop_region:
            x1 = max(0, np.min(xs) - 2)
            x2 = min(np.max(xs) + 2, resized_masks.shape[1])
            y1 = max(0, np.min(ys) - 2)
            y2 = min(np.max(ys) + 2, resized_masks.shape[0])
            mask = mask[y1:y2, x1:x2]
            bbox = x1, y1, x2, y2
        projection_point = get_intermediate_point(keypoints[bodypart[0]], keypoints[bodypart[1]], percentage_projection)
        mask_projection = [int(round(projection_point[1])) - y1, int(round(projection_point[0])) - x1]
        if mask_projection[0] < 0 or mask_projection[0] >= mask.shape[0] or mask_projection[1] < 0 or mask_projection[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask area
            return None, None
        if (use_central_projection or not projection_point_outside_mask) and mask[mask_projection[0], mask_projection[1]] == 0:  # projection point should lie in mask
            return None, None
        # orthogonal line
        if (keypoints[bodypart[0]][1] - keypoints[bodypart[1]][1]) != 0:
            m = - (keypoints[bodypart[0]][0] - keypoints[bodypart[1]][0]) / (keypoints[bodypart[0]][1] - keypoints[bodypart[1]][1])
        else:
            m = float('inf')
        t = projection_point[1] - m * projection_point[0]

        x_inputs, y_outputs = generate_segmentation_line(m, t, bbox, projection_point)
        if len(x_inputs) == 0:
            return None, (mask, None, None, projection_point, None, None)

        x_mask = x_inputs - x1
        y_mask = y_outputs - y1

        mask_out = np.where(mask[y_mask, x_mask])

        if len(mask_out[0]) == 0:
            return None, (mask, x_inputs, y_outputs, projection_point, None, None)
        p1 = [x_mask[mask_out[0][0]] + x1, y_mask[mask_out[0][0]] + y1, 1]
        p2 = [x_mask[mask_out[0][-1]] + x1, y_mask[mask_out[0][-1]] + y1, 1]

        if percentage_thickness < 0 and use_central_projection:
            new_point = get_intermediate_point(projection_point, p1, np.abs(percentage_thickness))
        elif use_central_projection:
            new_point = get_intermediate_point(projection_point, p2, np.abs(percentage_thickness))
        else:
            new_point = get_intermediate_point(np.asarray(p1), p2, np.abs(percentage_thickness))

        mask_point = [int(round(new_point[1])) - y1, int(round(new_point[0])) - x1]
        if mask_point[0] < 0 or mask_point[0] >= mask.shape[0] or mask_point[1] < 0 or mask_point[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask
            return None, None
        if mask[mask_point[0], mask_point[1]] == 0:
            return None, None
        return new_point, (mask, x_inputs, y_outputs, projection_point, p1, p2)

def generate_segmentation_line(m, t, bbox, point):
    x1, y1, x2, y2 = bbox

    if np.abs(m) != float('inf') and m != 0:

        if np.abs(m) > 1:

            y_from_x1 = m * x1 + t
            y_from_x2 = m * (x2 - 1) + t
            y_low = y_from_x1 if y_from_x1 < y_from_x2 else y_from_x2
            y_high = y_from_x2 if y_low == y_from_x1 else y_from_x1

            y_min = max(y_low, y1)
            y_max = min(y_high, y2)
            y_min = np.ceil(y_min)
            y_max = np.floor(y_max)

            y_outputs = np.arange(y_min, y_max - 0.5)
            x_inputs = (y_outputs - t) / m
            if len(x_inputs) > 0 and x_inputs[0] > x_inputs[-1]:
                x_inputs = x_inputs[::-1]
                y_outputs = y_outputs[::-1]

        else:
            x_from_y1 = (y1 - t) / m
            x_from_y2 = ((y2 - 1) - t) / m
            x_low = x_from_y1 if x_from_y1 < x_from_y2 else x_from_y2
            x_high = x_from_y2 if x_low == x_from_y1 else x_from_y1

            x_min = max(x_low, x1)
            x_max = min(x_high, x2)
            x_min = np.ceil(x_min)
            x_max = np.floor(x_max)

            x_inputs = np.arange(x_min, x_max - 0.5)
            y_outputs = m * x_inputs + t
        y_outputs = np.asarray(np.round(y_outputs), dtype=int)
        x_inputs = np.asarray(np.round(x_inputs), dtype=int)

        # sometimes we have odd behaviour and the borders, just to be sure
        if x_inputs.shape[0] > 0 and x_inputs[-1] >= x2:
            x_inputs = x_inputs[:-1]
            y_outputs = y_outputs[:-1]
        if x_inputs.shape[0] > 0 and x_inputs[0] < x1:
            x_inputs = x_inputs[1:]
            y_outputs = y_outputs[1:]
        if x_inputs.shape[0] > 0 and (y_outputs[-1] >= y2 or y_outputs[-1] < y1):
            x_inputs = x_inputs[:-1]
            y_outputs = y_outputs[:-1]
        if x_inputs.shape[0] > 0 and (y_outputs[0] >= y2 or y_outputs[0] < y1):
            x_inputs = x_inputs[1:]
            y_outputs = y_outputs[1:]

    elif m == 0: # m is 0, meaning we have a line parallel to the x-axis
        if y1 <= t < y2:
            x_inputs = np.arange(x1, x2)
            y_outputs = np.full(x2 - x1, t, dtype=int)
        else:
            x_inputs = np.zeros(0)
            y_outputs = np.zeros(0)
    else:  # m is infinity, meaning we have a line parallel to the y-axis
        if x1 <= point[0] < x2:
            x_inputs = np.full(y2 - y1, point[0], dtype=int)
            y_outputs = np.arange(y1, y2, dtype=int)
        else:
            x_inputs = np.zeros(0)
            y_outputs = np.zeros(0)

    return x_inputs, y_outputs