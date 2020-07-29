"""
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "PreprocCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PreprocCfg = ED()
# PreprocCfg.fs = 500
PreprocCfg.rpeak_mask_bias = 50  # ms
PreprocCfg.rpeak_threshold = 8
PreprocCfg.beat_winL = 250
PreprocCfg.beat_winR = 250


TrainCfg = ED()
