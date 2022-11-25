
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transform_utils import get_affine_transform, affine_transform
from data.skijump.yt_skijump_data import YTSkijump
from utils.general_utils import get_dict


class TransformerDatasetWrapper(Dataset):
    def __init__(self, cfg, params=None):

        self.aspect_ratio = 192.0 / 256
        self.pixel_std = 200

        self.name = cfg.NAME
        self.cfg = cfg
        self.dataset = get_dict(params, 'dataset')

        self.num_joints = cfg.NUM_JOINTS
        self.image_size = cfg.INPUT_SIZE
        self.heatmap_size = cfg.OUTPUT_SIZE[0]

        self.transform = None if 'transforms' not in params else params['transforms']
        self.generate_additional_segmentation = cfg.GENERATE_KEYPOINTS_SEGMENTATION if hasattr(cfg, "GENERATE_KEYPOINTS_SEGMENTATION") else None
        if hasattr(cfg, "GENERATE_KEYPOINTS_SEGMENTATION_VAL"):
            self.generate_additional_segmentation = [cfg.GENERATE_KEYPOINTS_SEGMENTATION_VAL, cfg.GENERATE_KEYPOINTS_SEGMENTATION_VAL]
            print("---- generating exactly {} keypoints per image ----".format(cfg.GENERATE_KEYPOINTS_SEGMENTATION_VAL))

    def __len__(self):
        return len(self.dataset.image_ids)

    def __getitem__(self, idx):
        image_id = self.dataset.image_ids[idx]

        num_additional = None
        additional = self.generate_additional_segmentation
        if additional is not None and additional[1] > 0:
            num_additional = additional[0] if additional[0] == additional[1] else np.random.randint(additional[0], additional[1])
        elif additional is not None:
            num_additional = 0
        gen_additional_seg = num_additional if self.generate_additional_segmentation is not None else None
        joints = self.dataset.load_keypoints(image_id, generate_on_segmentations=gen_additional_seg)
        keypoint_vectors, thickness_vectors = None, None
        if gen_additional_seg is not None:
            joints, representations = joints
            keypoint_vectors = representations["kp_vector"]
            thickness_vectors = representations["thickness"]
            num_generated = keypoint_vectors.shape[0] - self.num_joints
            joints = np.pad(joints, ((0, 0), (0, self.generate_additional_segmentation[1] - num_generated), (0, 0)))
            keypoint_vectors = np.pad(keypoint_vectors, ((0, self.generate_additional_segmentation[1] - num_generated), (0, 0)))
            thickness_vectors = np.pad(thickness_vectors, ((0, self.generate_additional_segmentation[1] - num_generated), (0, 0)))
        joints = joints[0]

        img = self.dataset.load_image(image_id)
        bbox = self.dataset.load_bbox(image_id)

        height, width = img.shape[:2]

        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if not x2 >= x1 and y2 >= y1:
            raise RuntimeError
        clean_box = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        c, s = self._box2cs(clean_box[:4])
        r = 0

        joints_heatmap = joints.copy()
        orig_joints = joints.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(joints.shape[0]):
            if joints[i, 2] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

        if self.transform:
            input = self.transform(input)

        target, target_weight = self.generate_target(joints_heatmap)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {'image_id': image_id,
                'center': c,
                'scale': s,
                'rotation': r,
                'joints': joints,
                'non_transformed': orig_joints,
                'image_width': img.shape[1],
                'keypoint_vectors': keypoint_vectors,
                'thickness_vectors': thickness_vectors
                }

        return input, target, target_weight, meta

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w - 1) * 0.5
        center[1] = y + (h - 1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def generate_target(self, joints):
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, 2]

        target = np.zeros((num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.cfg.SIGMA * 3

        for joint_id in range(num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]

            x = np.arange(0, self.heatmap_size[0], 1, np.float32)
            y = np.arange(0, self.heatmap_size[1], 1, np.float32)
            y = y[:, np.newaxis]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.cfg.SIGMA ** 2))

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


def fliplr_keypoint_vectors(keypoint_vector, matched_parts):
    """
    flip percentages
    """

    for pair in matched_parts:
        keypoint_vector[:, pair[0]], keypoint_vector[:, pair[1]] = keypoint_vector[:, pair[1]], keypoint_vector[:, pair[0]].copy()

    return keypoint_vector


class SkijumpYTTransformerDataset(TransformerDatasetWrapper):
    def __init__(self, cfg, params):

        num_annotations, ignore_list = None, None
        skijump_parameters = {
            "dataset_dir": params['dataset_dir'],
            "subset": params['subset'],
            "num_annotations": num_annotations,
            "bbox": getattr(cfg, "USE_CROPPED", False),
            "verbose": get_dict(params, "verbose", True)
        }
        if "seg_representation_type" in params:
            skijump_parameters["seg_representation_type"] = params["seg_representation_type"]
        if "bodypart_segmentations" in params:
            skijump_parameters["bodypart_segmentations"] = params["bodypart_segmentations"]

        dataset = YTSkijump(skijump_parameters)

        params['dataset'] = dataset

        super().__init__(cfg, params)
