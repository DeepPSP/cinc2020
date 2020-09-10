"""
"""
import os, sys
import json
from random import shuffle, randint
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

# torch.set_default_tensor_type(torch.DoubleTensor)

from .data_reader import CPSC2019Reader as CR