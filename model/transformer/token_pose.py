# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# Modified by Katja Ludwig
# ------------------------------------------------------------------------------

import os
from functools import partial

import einops as einops
import torch
from torch import nn

from model.hrnet.pose_higher_hrnet import HRNetW32Backbone
from model.transformer.vision_transformer import VisionTransformer


class TokenPoseNet(nn.Module):

    def __init__(self, keypoint_transform_config, in_feature_size=(256, 192, 3), patch_size=(4, 3), embed_dim=192, depth=12, num_heads=8, mlp_ratio=3, heatmap_size=(64, 48), cnn=None, correlate_everything=False):
        super().__init__()

        self.heatmap_size = heatmap_size if isinstance(heatmap_size, tuple) else (heatmap_size, heatmap_size)
        self.cnn = nn.Identity() if cnn is None else cnn
        embedding, mask_sequence_repetition = get_token_embedding(keypoint_transform_config, embed_dim)

        self.vision_transformer = VisionTransformer(
            in_feature_size=(in_feature_size[0] // 4, in_feature_size[1] // 4, 32), patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, scale_head=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_tokens=keypoint_transform_config['num_joints'], token_embedding=embedding, correlate_everything=correlate_everything)

        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if embed_dim <= hidden_heatmap_dim * 0.5 else nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, heatmap_dim)
        )

    def forward(self, x, keypoint_vector=None, thickness_vector=None):
        x = self.cnn(x)
        x_additional = []
        if keypoint_vector is not None:
            x_additional.append(keypoint_vector)
            if thickness_vector is not None:
                x_additional.append(thickness_vector)
        x_additional = x_additional if len(x_additional) > 0 else None
        x = self.vision_transformer(x, x_additional=x_additional)
        x = self.mlp_head(x)
        x = einops.rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x

    def init_weights(self, pretrained_transformer=None, verbose=True):
        if pretrained_transformer is not None and os.path.isfile(pretrained_transformer):
            pretrained_state_dict = torch.load(pretrained_transformer)
            res = self.load_state_dict(pretrained_state_dict)
            if verbose:
                print("LOADED WEIGHTS FROM {}, MISSING KEYS: {}".format(pretrained_transformer, len(res[0])))

        elif pretrained_transformer is not None and not os.path.isfile(pretrained_transformer):
            raise RuntimeError("Initialization failed. Check if weights file really exists!")

    def freeze_backbone(self):
        """
        Freeze backbone weights of HRNet and not of the last layers (the head)
        @return:
        """
        if isinstance(self.cnn, nn.Identity):
            raise RuntimeError("Backbone should be frozen, but TokenPoseNet has no CNN backbone")
        for name, param in self.cnn.named_parameters():
            param.requires_grad = False
        for name, param in self.cnn.named_buffers():
            param.requires_grad = False


class Concat(nn.Module):
    """
    torch.cat as nn.Module
    """
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.concat_dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.concat_dim)



def get_token_embedding(keypoint_transform_config, embed_dim):
    fraction_keypoint = 0.5
    return nn.ModuleList([  nn.Linear(keypoint_transform_config['num_original_joints'], int(round(embed_dim * fraction_keypoint))),
                            nn.Linear(3, embed_dim - int(round(embed_dim * fraction_keypoint))),
                            Concat(dim=2)]), 1


def get_pose_net(cfg, verbose=True):
    num_joints = cfg.NUM_JOINTS
    keypoint_transform_config = {
        'num_joints': num_joints + cfg.GENERATE_KEYPOINTS_SEGMENTATION[1],
        'num_original_joints': num_joints,
    }

    patch_size = getattr(cfg, "PATCH_SIZE", (4, 3))
    correlate_tokens = getattr(cfg, "TOKEN_ATTENTION", False)
    input_size = (cfg.INPUT_SIZE[1], cfg.INPUT_SIZE[0], 3)
    heatmap_size = (cfg.OUTPUT_SIZE[0][1], cfg.OUTPUT_SIZE[0][0])
    model = TokenPoseNet(in_feature_size=input_size, embed_dim=cfg.EMBED_SIZE, patch_size=patch_size, heatmap_size=heatmap_size,
                         keypoint_transform_config=keypoint_transform_config, cnn=HRNetW32Backbone(num_stages=3), correlate_everything=correlate_tokens)
    if verbose:
        print("Using HRNet stage 3 backbone")

    return model


def get_hrnet_key(key):
    if key.startswith("conv") or key.startswith("bn") or key.startswith("layer1"):
        return "1.stem." + key
    elif key.startswith("stage"):
        return "1." + key[:7] + "stage." + key[7:]
    elif key.startswith("transition"):
        return "1." + key
    elif key.startswith("final_layers.0"):
        return "1.final_layer_non_deconv" + key[14:]
    elif key.startswith("final_layers.1"):
        return "1.final_layer_after_deconv" + key[14:]
    elif key.startswith("deconv_layers.0"):
        return "1.deconv_layer.deconv_layer" + key[15:]
    raise RuntimeError("key not known {}".format(key))

