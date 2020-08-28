"""
"""
import os, sys
import json
from random import shuffle, randint
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

torch.set_default_tensor_type(torch.DoubleTensor)

from .data_reader import LUDBReader as LR


class LUDB(Dataset):
    """
    """
    def __init__(self, config:ED, leads:Optional[Union[Sequence[str], str]], training:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
            can be one of "A", "B", "AB", "E", "F", or None (or '', defaults to "ABEF")
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = LR(db_dir=config.db_dir)
        self.training = training
        self.classes = self.config.classes
        self.n_classes = len(self.classes)
        self.siglen = self.config.input_len
        if leads is None:
            self.leads = self.reader.all_leads
        elif isinstance(leads, str):
            self.leads = [leads]
        else:
            self.leads = list(leads)
        self.leads = [
            self.reader.all_leads[idx] \
                for idx,l in enumerate(self.leads) if l.lower() in self.reader.all_leads_lower
        ]

        self.records = self._train_test_split(config.train_ratio, force_recompute=False)

        self.__data_aug = self.training

    def __getitem__(self, index):
        """ finished, checked,
        """
        rec = self.records[index]
        values = self.reader.load_data(
            rec, data_format='channel_first', units='mV',
        )
        if self.config.normalize_data:
            values = (values - np.mean(values)) / np.std(values)
        labels = self.reader.load_ann(
            rec, leads=self.leads, metadata=False
        )['waves']

        if self.__data_aug:
            # data augmentation for input
            raise NotImplementedError

        return values, labels


    def __len__(self):
        """
        """
        return len(self.records)


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    
    def _train_test_split(self, train_ratio:float=0.8, force_recompute:bool=False) -> List[str]:
        """ finished, checked,

        Parameters:
        -----------
        train_ratio: float, default 0.8,
            ratio of the train set in the whole dataset (or the whole tranche(s))
        force_recompute: bool, default False,
            if True, force redo the train-test split,
            regardless of the existing ones stored in json files

        Returns:
        --------
        records: list of str,
            list of the records split for training or validation
        """
        raise NotImplementedError


    def persistence(self) -> NoReturn:
        """ finished, checked,

        make the dataset persistent w.r.t. the tranches and the ratios in `self.config`
        """
        prev_state = self.__data_aug
        self.disable_data_augmentation()
        if self.training:
            ratio = int(self.config.train_ratio*100)
        else:
            ratio = 100 - int(self.config.train_ratio*100)
        raise NotImplementedError

        self.__data_aug = prev_state
