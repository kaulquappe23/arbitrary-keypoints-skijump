
import os

from data.skijump.skijump_joint_order import YTSkijumpJointOrder
from paths import YTSkijumpLoc
from experiments.transformer_config import TransformerConfig


class YTSkijumpTransformerConfig(TransformerConfig):

    def __init__(self):

        super().__init__()

        self.NAME = "yt_skijump"
        self.ROOT = YTSkijumpLoc.base_path
        self.ANNOTATIONS_DIR = YTSkijumpLoc.annotation_path

        self.SINGLE_GPU_BATCH_SIZE = 32

        self.USE_CROPPED = True

        self.NUM_JOINTS = YTSkijumpJointOrder.num_joints
        # keypoints to switch left and right if image is flipped horizontally
        self.FLIP_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 10], [8, 11], [9, 12], [13, 15], [14, 16]]
        # location of body part segmentation masks for all subsets
        self.BODYPART_SEGMENTATIONS = {"train": os.path.join(YTSkijumpLoc.base_path, YTSkijumpLoc.segmentation_path),
                                       "val": os.path.join(YTSkijumpLoc.base_path, YTSkijumpLoc.segmentation_path),
                                       "test": os.path.join(YTSkijumpLoc.base_path, YTSkijumpLoc.segmentation_path)}

        self.AFFINE_VARIANT = True


class YTSkijumpVectorEncodedTransformerConfig(YTSkijumpTransformerConfig):

    def __init__(self):

        super().__init__()

        # number of arbitrary keypoints that are created (min, max), to evaluate only on standard points, we set them to 0
        self.GENERATE_KEYPOINTS_SEGMENTATION = (0, 0)



class YTSkijumpSegmentationTransformerConfig(YTSkijumpTransformerConfig):

    def __init__(self):

        super().__init__()

        self.GENERATE_KEYPOINTS_SEGMENTATION = (5, 50)
        # during validation, we want the exact same arbitrary points for each validation run, we can set the number of points here
        self.GENERATE_KEYPOINTS_SEGMENTATION_VAL = 200

        self.EMBEDDING = ("vectorized_keypoints", ("concat", 0.5))



