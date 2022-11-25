from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import torch


def set_deterministic(all=False):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    if all:
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

