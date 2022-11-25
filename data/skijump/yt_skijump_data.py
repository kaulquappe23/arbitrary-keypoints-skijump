import os
import pickle

import numpy as np

from data.bodypart_segmentation_order import YTSkijumpBodypartOrder
from data.dataset_wrapper import DataWrapper
from data.skijump.skijump_joint_order import YTSkijumpJointOrder
from paths import YTSkijumpLoc
from utils.general_utils import get_dict


class YTSkijump(DataWrapper):

    def __init__(self, params):
        """
        @param params: Needs to contain dataset_dir, subset, might contain num_annotations, load_annotatins, ignore_list, bbox, restore_from_scale, bodypart_segmentations, seg_representation_type
        """
        super().__init__(params)
        self.name = "yt_skijump"
        self.image_ids = []
        self.image_paths = {}
        self.annotations = {}
        self.bboxes = {}

        dataset_dir = params["dataset_dir"]
        subset = params["subset"]
        bbox = get_dict(params, "bbox", False)
        verbose = get_dict(params, "verbose", True)

        self.subset = subset

        if subset.endswith(".csv"):
            subset = subset.replace(".csv", "")
        annotation_file = os.path.join(dataset_dir, subset + ".csv")
        bbox_file = annotation_file.replace(".csv", "_bbox.pkl") if isinstance(bbox, bool) and bbox else bbox

        bodypart_segmentations = get_dict(params, "bodypart_segmentations")
        seg_representation_type = get_dict(params, "seg_representation_type")

        self.used_bodyparts = YTSkijumpBodypartOrder.get_keypoint_bodypart_triples()
        self.bodypart_dict = YTSkijumpBodypartOrder.get_bodypart_to_keypoint_dict()
        self.bodypart_ids_no_center = [YTSkijumpBodypartOrder.l_ski, YTSkijumpBodypartOrder.r_ski]
        self.bodypart_indices = YTSkijumpJointOrder.line_bodypart_indices()

        if bodypart_segmentations is not None:
            self.full_image_mask = True
            self.bodypart_masks = {}
            self.norm_pose = (seg_representation_type == "norm_pose")

            seg_files = [f for f in os.listdir(bodypart_segmentations)]
            seg_ids = []
            for seg_file in seg_files:
                image_name = seg_file.replace(".png", ".jpg")
                # image_name = image_name[:image_name.rfind("_")] + "_(" + image_name[image_name.rfind("_") + 1:]
                seg_ids.append(image_name)
                self.bodypart_masks[image_name] = os.path.join(bodypart_segmentations, seg_file)

        with open(annotation_file, "r") as csv:
            if verbose:
                print("Loading: " + annotation_file)
            assertion_num = csv.readline()
            assertion_num = int(assertion_num[1:])
            keypoint_line = "event;frame_num;athlete;slowmotion;head_x;head_y;head_s;rsho_x;rsho_y;rsho_s;relb_x;relb_y;relb_s;rhan_x;rhan_y;rhan_s;lsho_x;lsho_y;lsho_s;lelb_x;lelb_y;lelb_s;lhan_x;lhan_y;lhan_s;rhip_x;rhip_y;" \
                            "rhip_s;rkne_x;rkne_y;rkne_s;rank_x;rank_y;rank_s;lhip_x;lhip_y;lhip_s;lkne_x;lkne_y;lkne_s;lank_x;lank_y;lank_s;rsti_x;rsti_y;rsti_s;rsta_x;rsta_y;rsta_s;lsti_x;lsti_y;lsti_s;lsta_x;lsta_y;lsta_s"
            read_kp_line = csv.readline().strip().replace(" ", "")
            assert read_kp_line.startswith(keypoint_line)
            cnt = 0
            cnt_seg = 0
            offset = 4
            num_points = self.get_num_keypoints()
            for line in csv:
                cnt += 1
                splits = line.split(sep=";")
                assert len(splits) == num_points * 3 + offset,  "error in file {} line {}".format(annotation_file, cnt)
                image_name = "{}_({:05d}).jpg".format(splits[0], int(splits[1]))

                if bodypart_segmentations is not None and image_name not in self.bodypart_masks:
                    continue
                self.image_ids.append(image_name)
                self.image_paths[image_name] = os.path.join(YTSkijumpLoc.frames_path, splits[0], image_name)

                annotations = np.zeros((1, num_points, 3))
                for i in range(num_points):
                    vis = float(splits[offset + i * 3 + 2])
                    if vis > 1:
                        vis = 1
                    annotations[0, i] = [float(splits[offset + i * 3]), float(splits[offset + i * 3 + 1]), vis]

                if bodypart_segmentations is not None:
                    annotations = annotations[0]
                    cnt_seg += 1
                self.annotations[image_name] = annotations
                if bbox:
                    if not os.path.exists(annotation_file.replace(".csv", "_bbox.pkl")):
                        x_coords = annotations[0, np.where(annotations[0, :, 2] > 0), 0]
                        y_coords = annotations[0, np.where(annotations[0, :, 2] > 0), 1]
                        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
                        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
                        w = max_x - min_x
                        h = max_y - min_y
                        offset_w = int(w * 0.2)
                        offset_h = int(h * 0.2)
                        min_x = max(0, min_x - offset_w)
                        min_y = max(0, min_y - offset_h)
                        image = self.load_image(image_name)
                        max_x = min(image.shape[1], max_x + offset_w)
                        w = max_x - min_x
                        max_y = min(image.shape[0], max_y + offset_h)
                        h = max_y - min_y
                        assert h >= 0 and w >= 0
                        self.bboxes[image_name] = [min_x, min_y, w, h]

            if bbox:
                if verbose:
                    print("USING ONLY CROPPED SKIJUMPERS")
                if os.path.exists(bbox_file):
                    with open(bbox_file, "rb") as f:
                        self.bboxes = pickle.load(f)
                else:
                    if verbose:
                        print("USING WHOLE IMAGES")
                    with open(annotation_file.replace(".csv", "_bbox.pkl"), "wb") as f:
                        pickle.dump(self.bboxes, f)
            else:
                raise RuntimeError("Only bounding boxes are supported")

            assert cnt == assertion_num, "number of expected entries is {}, but found {} entries".format(assertion_num, cnt)
            if verbose:
                print("Loaded {} annotations".format(cnt))
                if cnt_seg > 0:
                    print("Loaded {} segmentation masks".format(cnt_seg))

    def load_bodypart_mask(self, image_id, size=None, scale_mask=0.1):
        return super().load_bodypart_mask(image_id, size, scale_mask)


    def get_num_keypoints(self):
        return 17


