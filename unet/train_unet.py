"""
References:
-----------
[1] 
"""
import os
import time
import logging
import argparse
from copy import deepcopy
from collections import deque
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

torch.set_default_tensor_type(torch.DoubleTensor)

from models.ecg_unet import ECG_UNET
# from models.utils.torch_utils import BCEWithLogitsWithClassWeightLoss
from model_configs import ECG_UNET_CONFIG
from utils.misc import init_logger, get_date_str, dict_to_str, str2bool


def train(model:nn.Module, device:torch.device, config:dict, log_step:int=20, logger:Optional[logging.Logger]=None, debug:bool=False):
    """
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
