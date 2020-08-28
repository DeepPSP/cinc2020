"""
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "TrainCfg",
]


TrainCfg = ED()
TrainCfg.fs = 500
TrainCfg.train_ratio = 0.8
TrainCfg.classes = [
    'p',  # pwave
    'N',  # qrs complex
    't',  # twave
    'i',  # isoelectric
]
TrainCfg.start_from = int(2 * TrainCfg.fs)
TrainCfg.end_at = int(2 * TrainCfg.fs)
TrainCfg.input_len = int(4 * TrainCfg.fs)
