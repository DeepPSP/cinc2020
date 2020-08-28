"""
References:
-----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
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

# as for `start_from` and `end_at`, see ref. [1] section 3.1
TrainCfg.start_from = int(2 * TrainCfg.fs)
TrainCfg.end_at = int(2 * TrainCfg.fs)
TrainCfg.input_len = int(4 * TrainCfg.fs)

TrainCfg.over_sampling = 2
