"""
"""
import sys
from typing import Union, NoReturn

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class Mish(torch.nn.Module):
    """ The Mish activation """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        """
        """
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Swish(torch.nn.Module):
    """ The Swish activation """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        """
        """
        x = x * F.sigmoid(x)
        return x


class Conv_Bn_Activation(nn.Module):
    """
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, activation:Union[str,callable], kernel_initializer:Optional[Union[str,callable]]=None, bn:bool=True, bias:bool=False) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()

        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv.weight)
            elif kernel_initializer.lower() in ['he_normal', 'kaiming_normal']:
                nn.init.kaiming_normal_(conv.weight)
            elif kernel_initializer.lower() in []:
                pass
            else:  # TODO: add more activations
                pass
            
        self.layers.append(conv)

        if bn:
            self.layers.append(nn.BatchNorm1d(out_channels))

        if isinstance(activation, str):
            activation = activation.lower()
        if activation == "mish":
            self.layers.append(Mish())
        elif activation == "swish":
            self.layers.append(Swish())
        elif activation == "relu":
            self.layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        elif callable(activation):
            self.layers.append(activation)
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")

    def forward(self, x):
        """
        """
        for l in self.layers:
            x = l(x)
        return x


class ATI_CNN(nn.Module):
    """
    """
    def __init__(self, classes:list, input_len:int, bidirectional:bool=True):
        """
        """
        super().__init__()
        self.classes = classes
        self.nb_classes = len(classes)
        self.nb_leads = 12
        self.input_len = input_len
        self.bidirectional = bidirectional

    def forward(self, input):
        """
        """
        raise NotImplementedError
