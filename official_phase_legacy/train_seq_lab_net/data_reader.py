"""
data reader for CPSC2019
"""
import os
import json
from collections import namedtuple
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
from easydict import EasyDict as ED

import utils
from utils.misc import (
    get_record_list_recursive,
    get_record_list_recursive2,
    get_record_list_recursive3,
    dict_to_str,
    ms2samples,
    ECGWaveForm, masks_to_waveforms,
)


__all__ = [
    "CPSC2019Reader",
]


class CPSC2019Reader(object):
    """
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        raise NotImplementedError