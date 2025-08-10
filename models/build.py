# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .simmim import build_simmim


def build_model(config, is_pretrain: bool = True):
    # Only pre-training is supported in this pared-down repo
    return build_simmim(config)
