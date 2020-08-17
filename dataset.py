"""
"""
import os, sys
from copy import deepcopy
from typing import Union, Optional, Tuple, Dict, Sequence, NoReturn

import numpy as np
from easydict import EasyDict as ED
import torch
from torch.utils.data.dataset import Dataset

from cfg import TrainCfg
from data_reader import CINC2020_Reader as CR


__all__ = [
    "CINC2020",
]


class CINC2020(Dataset):
    """
    """
    def __init__(self, config:ED, tranches:Optional[Sequence[str]]=None, train:bool=True) -> NoReturn:
        """
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        self.tranches = tranches or self.reader.db_tranches
        self.train = train
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return super().__len__()
