# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# Modified by Katja Ludwig
# containing parts from timm library https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# ------------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, scale_head=True, attn_drop=0., proj_drop=0., kv_token_length=None):
        """
        kv_token_length: if None, the attention is calculated between all tokens. If a number is set, only these tokens are used for keys and values and all tokens for the query
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5 if not scale_head else (dim // num_heads) ** -0.5
        self.kv_token_length = kv_token_length

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # batch size, number of tokens (visual and keypoints) and number of channels
        kv_tokens = self.kv_token_length if self.kv_token_length is not None else N
        qkv = self.qkv(x)  # linear transformation of current tokens to queries, keys and values (channel dimension tripled)
        qkv = qkv.reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor'))  # channels are split into 3 dims (qkv), number of heads and remaining channels
        qkv = qkv.permute(2, 0, 3, 1, 4)  # now we have the order 3, B, heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v have the dimensions B, heads, N, C
        k, v = k[:, :, -kv_tokens:], v[:, :, -kv_tokens:]  # k, v have dimensions B, heads, N_visual, C

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        k_transposed = k.transpose(-2, -1)  # k has dimensions B, heads, C, N
        attn = q @ k_transposed  # matrix multiplication in the last two dimensions, attn has dimensions B, heads, N, N_visual
        attn = attn * self.scale  # vector normalization
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # matrix multiplication of attention and values, dimensions are B, heads, N, C (all N, not only visual)
        x = x.transpose(1, 2)  # dimensions are B, N, heads, C
        x = x.reshape(B, N, C)  # removing heads dimension
        x = self.proj(x)  # linear projection from channels to channels
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., scale_head=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=None, num_tokens=None,
                 kv_token_length=None):
        super().__init__()
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, scale_head=scale_head, attn_drop=attn_drop, proj_drop=drop, kv_token_length=kv_token_length)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.non_token_indices = None

    def forward(self, x, pos_encoding=None, return_attention=False):
        """
        @param x: The input to the transformer block with visual tokens and keypoint (and thickness) tokens
        @param pos_encoding: if available, it is added before the block
        @param return_attention: return the attention result without additional mlp and norm execution
        @return:
        """
        if pos_encoding is not None:
            x[:, -pos_encoding.size(1):] += pos_encoding
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, feature_size=(224, 224, 3), patch_size=(16, 16), embed_dim=768, embed_type="conv"):
        super().__init__()
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.emb_type = embed_type
        if embed_type == "conv":
            self.proj = nn.Conv2d(feature_size[-1], embed_dim, kernel_size=patch_size, stride=patch_size)
        elif embed_type == "linear":
            self.proj = nn.Linear(feature_size[-1] * patch_size[0] * patch_size[1], embed_dim)
        else:
            raise RuntimeError("Patch embedding type {} not implemented".format(embed_type))

    def forward(self, x):
        if self.emb_type == "conv":
            x = self.proj(x).flatten(2).transpose(1, 2)
        elif self.emb_type == "linear":
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1])
            x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, in_feature_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., scale_head=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, num_tokens=1,
                 embed_type="linear", token_embedding=None, correlate_everything=True):
        """

        @param num_tokens:
        @param pos_encoding: learnable or sine
        @param token_embedding: embedding of tokens as torch layers (one or more as a list) or torch parameter. If it is a module list, the elements of the list are applied
        to the additional input during forward pass in the same order. After the transformations, the last entry of the module list is applied to the results.
        @param mask_sequence_repetition: if there is a sequence of tokens that correspond to each other, set the length of the sequence (e.g. for thickness tokens, the length is 2)
        """

        super().__init__()
        self.num_tokens = num_tokens
        self.in_feature_size = (in_feature_size, in_feature_size, 3) if isinstance(in_feature_size, int) else in_feature_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert self.in_feature_size[0] % self.patch_size[0] == 0 and self.in_feature_size[1] % self.patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_embed = PatchEmbed(
            feature_size=in_feature_size, patch_size=patch_size, embed_dim=embed_dim, embed_type=embed_type)
        num_patches = self.patch_embed.num_patches

        self.token = token_embedding

        self.pos_encoding = nn.Parameter(self._make_sine_position_encoding(embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        kv_token_length = None if correlate_everything else num_patches
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, scale_head=scale_head,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_patches=self.patch_embed.num_patches, num_tokens=self.num_tokens,
                kv_token_length=kv_token_length)
            for i in range(depth)])

        if isinstance(self.token, nn.Parameter):
            trunc_normal_(self.token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_sine_position_encoding(self, d_model, temperature=10000,
                                     scale=2 * math.pi):

        h, w = self.in_feature_size[0] // (self.patch_size[0]), self.in_feature_size[1] // (self.patch_size[1])  # number of patches per height and width
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)  # 1, 2, 3, 4, 5, ... in x-direction
        x_embed = area.cumsum(2, dtype=torch.float32)  # 1, 2, 3, 4, 5, ... in y-direction

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale  # equally spaced entries between 0 and scale in y-direction
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale  # equally spaced entries between 0 and scale in x-direction

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / one_direction_feats)  # temperature ** (half of embed size equally spaced double entries (always to identical entries)

        pos_x = x_embed[:, :, :, None] / dim_t  # adds embedding_size // 2 dimension in the end. smooth transition from largest value scale (pos_x[:, :, -1, 0]) to 0 (pos_x[:, :, 0, -1]) with exponential drop
        pos_y = y_embed[:, :, :, None] / dim_t  # same, but dimension 1 and 2 swapped
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # sine and cos wave from [:, :, 0, -1] to [:, :, -1, 0] becoming "slower" at [:, :, 0, -1]. Only every second entry taken but there are always two identical (for sine and cos). After stack and flatten, alternating sine and cosine values
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # combination of sine and cosine waves. in the embedding direction, the first halt (up to 96) is y-direction, then x-direction. the embedding direction is shifted to dimension 1
        pos = pos.flatten(2).permute(0, 2, 1)  # put the patches together
        return pos

    def prepare_tokens(self, x, x_additional=None):
        """
        Prepare the visual tokens and other tokens, e.g. keypoint or thickness tokens
        @param x:
        @param x_additional:
        @return:
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        if isinstance(self.token, nn.Parameter):
            # learnable tokens
            tokens = self.token.expand(B, -1, -1)
        else:
            # tokens generated through embeddings
            tokens = []
            for i, token_input in enumerate(x_additional):
                tokens.append(self.token[i](token_input))
            if len(tokens) == 1:
                tokens = tokens[0]
            else:
                tokens = self.token[-1](tokens)

        x += self.pos_encoding
        x = torch.cat((tokens, x), dim=1)

        return self.pos_drop(x)

    def forward(self, x, x_additional=None):
        """
        Forward pass of the vision transformer
        @param x: image or extracted feature map that is going to be embedded into visual tokens
        @param x_additional: can contain additional information that needs to be embedded, e.g., keypoint vectors and thickness vectors
        @return:
        """
        x = self.prepare_tokens(x, x_additional)

        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x, pos_encoding=self.pos_encoding if blk_idx > 0 else None)
        cut_tokens = self.patch_embed.num_patches
        return x[:, 0: -cut_tokens]  # only return token
