"""
"""
import os, sys
import json
from random import shuffle
from copy import deepcopy
from typing import Union, Optional, Tuple, Dict, Sequence, Set, NoReturn

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
    def __init__(self, config:ED, tranches:Optional[str]=None, train:bool=True) -> NoReturn:
        """
        """
        self._TRANCHES = TrainCfg.tranche_classes.keys()  # ["A", "B", "AB", "E", "F"]
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        self.tranches = tranches
        self.train = train

        assert not self.tranches or self.tranches in self._TRANCHES

        self.records = self._train_test_split(config.train_ratio, force_recompute=False)

    def __getitem__(self, index):
        """
        """
        rec = self.records[index]
        values = self.reader.load_data(
            rec,
            data_format='channel_first', units='mV', backend='wfdb'
        )
        labels = self.reader.get_labels(
            rec,
            scored_only=True, abbr=True, normalize=True
        )
        return values, labels

    def _get_val_item(self, index):
        """
        """
        raise NotImplementedError

    def __len__(self):
        return super().__len__()

    
    def _train_test_split(self, train_ratio:float=0.8, force_recompute:bool=False) -> List[str]:
        """ NOT finished,

        Parameters:
        -----------
        train_ratio: float, default 0.8,
        force_recompute: bool, default False

        Returns:
        --------
        records: list of str,
            list of the records split for training or validation
        """
        _train_ratio = int(train_ratio*100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0
        train_file = os.path.join(self.reader.db_dir_base, f"train_{_train_ratio}.json")
        test_file = os.path.join(self.reader.db_dir_base, f"test_{_test_ratio}.json")
        if force_recompute or not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            tranche_records = {t: [] for t in _TRANCHES}
            train = {t: [] for t in _TRANCHES}
            test = {t: [] for t in _TRANCHES}
            for t in _TRANCHES:
                for _t in list(t):
                    tranche_records[t] += self.reader.all_records[_t]
            for t in _TRANCHES:
                is_valid = False
                while not is_valid:
                    shuffle(tranche_records[t])
                    split_idx = int(len(tranche_records[t])*train_ratio)
                    train[t] = tranche_records[t][:split_idx]
                    test[t] = tranche_records[t][split_idx:]
                    is_valid = _check_train_test_split_validity(train[t], test[t], set(TrainCfg.tranche_classes[t]))
            with open(train_file, "w") as f:
                json.dump(train, train_file, ensure_ascii=False)
            with open(test_file, "w") as f:
                json.dump(test, test_file, ensure_ascii=False)
        else:
            with open(train_file, "r") as f:
                train = json.load(train_file)
            with open(test_file, "r") as f:
                test = json.load(test_file)

        add = lambda a,b:a+b
        if not self.tranches:
            if self.train:
                records = list(map(add, [v for v in train.values()]))
            else:
                records = list(map(add, [v for v in test.values()]))
        else:
            if self.train:
                records = train[self.tranches]
            else:
                records = test[self.tranches]
        return records


    def _check_train_test_split_validity(self, train:List[str], test:List[str], all_classes:Set[str]) -> bool:
        """

        Parameters:
        -----------
        train: list of str,
            list of the records in the train set
        test: list of str,
            list of the records in the test set

        Returns:
        --------
        is_valid: bool,
            the split is valid or not
        """
        add = lambda a,b:a+b
        train_classes = set(list(map(add, [self.reader.get_labels(rec) for rec in train])))
        test_classes = set(list(map(add, [self.reader.get_labels(rec) for rec in test])))
        is_valid = (all_classes-train_classes==set()) and (train_classes==test_classes)
        return is_valid

