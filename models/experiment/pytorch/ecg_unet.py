"""
UNet structure models,
mainly for ECG wave delineation
"""
import sys
from copy import deepcopy
from collections import OrderedDict
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from models.utils.torch_utils import (
    DoubleConv, DownDoubleConv,
)
