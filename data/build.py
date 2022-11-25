
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modidfied by Katja Ludwig
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch.utils.data
import torchvision

from data.transformer_dataset import SkijumpYTTransformerDataset
from experiments.yt_skijump_config import YTSkijumpTransformerConfig


def get_dataset_class(name, transformer=False):
    if transformer :
        if name == YTSkijumpTransformerConfig().NAME:
            return SkijumpYTTransformerDataset
        else:
            raise RuntimeError("Dataset unknown for transformer " + name)
    else:
        raise RuntimeError("Dataset unknown: " + name)


def get_params(cfg, subset, is_train=True, augment=True, transformer=False):
    params = {'subset': subset}
    name = cfg.NAME

    if "cropped" in cfg.GENERATORS:
        params["cropped"] = True
    if transformer:
        params["is_train"] = is_train
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        params["transforms"] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        params["augment"] = augment
        if hasattr(cfg, "BODYPART_SEGMENTATIONS"):
            if getattr(cfg, "GENERATE_KEYPOINTS_SEGMENTATION", (0, 0))[1] > 0:
                if is_train:
                    params["bodypart_segmentations"] = cfg.BODYPART_SEGMENTATIONS["train"]
                elif "val" in subset:
                    params["bodypart_segmentations"] = cfg.BODYPART_SEGMENTATIONS["val"]
                elif "test" in subset:
                    params["bodypart_segmentations"] = cfg.BODYPART_SEGMENTATIONS["test"]
                else:
                    params["bodypart_segmentations"] = cfg.BODYPART_SEGMENTATIONS[subset]
            params["seg_representation_type"] = "kp_thick_vectors"
    if name == YTSkijumpTransformerConfig().NAME:
        params['dataset_dir'] = os.path.join(cfg.ROOT, cfg.ANNOTATIONS_DIR)
    else:
        raise RuntimeError("Dataset unknown: " + name)
    return params


def make_test_dataloader(cfg, subset, batch_size=1, verbose=True):
    transformer = "Transformer" in type(cfg).__name__
    params = get_params(cfg, subset, is_train=False, augment=False, transformer=transformer)
    params["verbose"] = verbose
    dataset_class = get_dataset_class(cfg.NAME, transformer=transformer)
    dataset = dataset_class(cfg, params)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    return data_loader, dataset
