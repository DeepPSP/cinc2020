"""
(CRNN) models training

Training strategy:
------------------
1. the following pairs of classes will be treated the same:
    (RBBB, CRBBB), (PAC, SVPB), (PVC, VPB)

2. the following classes will be determined by the special detectors:
    PR, LAD, RAD, LQRSV, Brady,

3. models will be trained for each tranche separatly
    tranche A and B are from the same source, hence will be treated one during training,
    the distribution of the classes for each tranche are as follows:
    A+B: {'IAVB': 828, 'AF': 1374, 'AFL': 54, 'Brady': 271, 'CRBBB': 113, 'IRBBB': 86, 'LBBB': 274, 'NSIVCB': 4, 'PR': 3, 'PAC': 689, 'PVC': 188, 'LQT': 4, 'QAb': 1, 'RAD': 1, 'RBBB': 1858, 'SA': 11, 'SB': 45, 'NSR': 922, 'STach': 303, 'SVPB': 53, 'TAb': 22, 'TInv': 5, 'VPB': 8}
    C: {'AF': 2, 'Brady': 11, 'NSIVCB': 1, 'PAC': 3, 'RBBB': 2, 'SA': 2, 'STach': 11, 'SVPB': 4, 'TInv': 1}
    D: {'AF': 15, 'AFL': 1, 'NSR': 80, 'STach': 1}
    E: {'IAVB': 797, 'AF': 1514, 'AFL': 73, 'CRBBB': 542, 'IRBBB': 1118, 'LAnFB': 1626, 'LAD': 5146, 'LBBB': 536, 'LQRSV': 182, 'NSIVCB': 789, 'PR': 296, 'PAC': 398, 'LPR': 340, 'LQT': 118, 'QAb': 548, 'RAD': 343, 'SA': 772, 'SB': 637, 'NSR': 18092, 'STach': 826, 'SVPB': 157, 'TAb': 2345, 'TInv': 294}
    F: {'IAVB': 769, 'AF': 570, 'AFL': 186, 'Brady': 6, 'CRBBB': 28, 'IRBBB': 407, 'LAnFB': 180, 'LAD': 940, 'LBBB': 231, 'LQRSV': 374, 'NSIVCB': 203, 'PAC': 639, 'LQT': 1391, 'QAb': 464, 'RAD': 83, 'RBBB': 542, 'SA': 455, 'SB': 1677, 'NSR': 1752, 'STach': 1261, 'SVPB': 1, 'TAb': 2306, 'TInv': 812, 'VPB': 357}
    hence in this manner,
    training classes for each tranche are as follows:
    A+B: [IAVB', 'AF', 'AFL',  'IRBBB', 'LBBB', 'PAC', 'PVC', 'RBBB', 'SB', 'NSR', 'STach', 'TAb']
    E: ['IAVB', 'AF', 'AFL', 'IRBBB', 'LAnFB', 'LBBB', 'NSIVCB', 'PAC', 'LPR', 'LQT', 'QAb', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv']
    F: ['IAVB', 'AF', 'AFL', 'IRBBB', 'LAnFB', 'LBBB', 'NSIVCB', 'PAC', 'LQT', 'QAb', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv', 'VPB']


4. one model will be trained using the whole dataset

"""
import os
import argparse
from copy import deepcopy
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real, Number

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from models.ecg_crnn import ATI_CNN
from model_configs.ati_cnn import ATI_CNN_CONFIG


__all__ = []


DAS = True
