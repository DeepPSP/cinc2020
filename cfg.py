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
TrainCfg = ED()
