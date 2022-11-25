
from data.base_config import BaseConfig


class TransformerConfig(BaseConfig):

    def __init__(self):
        super().__init__()
        self.OUTPUT_SIZE = [(48, 64)]
        self.INPUT_SIZE = (192, 256)
        self.PATCH_SIZE = (4, 3)

        self.SINGLE_GPU_BATCH_SIZE = 32

        self.GENERATORS = "cropped_heatmap"

        self.SCALE_TYPE = "short"

        self.EMBED_SIZE = 192

        self.PRETRAINED_HEAD = False
        self.PRETRAINED_TOKEN = False

        self.POS_ENCODING = "sine"
        self.CNN = "hrnet_stage3"

