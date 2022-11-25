
from experiments.yt_skijump_config import *

def get_transformer_config_by_name(name, vector_encoded=False):
    if name == "yt_skijump" and not vector_encoded:
        return YTSkijumpTransformerConfig()
    elif name == "yt_skijump-seg":
        return YTSkijumpSegmentationTransformerConfig()
    elif name == "yt_skijump" and vector_encoded:
        return YTSkijumpVectorEncodedTransformerConfig()
    else:
        raise RuntimeError("Unknown config: " + name)
