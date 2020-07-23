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
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, activation:Union[str,callable], kernel_initializer:Optional[Union[str,callable]]=None, bn:bool=True, bias:bool=True) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        padding = (kernel_size - 1) // 2  # 'same' padding

        self.layers = nn.ModuleList()

        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv_layer.weight)
            elif kernel_initializer.lower() in ['he_normal', 'kaiming_normal']:
                nn.init.kaiming_normal_(conv_layer.weight)
            elif kernel_initializer.lower() in ['he_uniform', 'kaiming_uniform']:
                nn.init.kaiming_uniform_(conv_layer.weight)
            elif kernel_initializer.lower() in ['xavier_normal', 'glorot_normal']:
                nn.init.xavier_normal_(conv_layer.weight)
            elif kernel_initializer.lower() in ['xavier_uniform', 'glorot_uniform']:
                nn.init.xavier_uniform_(conv_layer.weight)
            elif kernel_initializer.lower() == 'normal':
                nn.init.normal_(conv_layer.weight)
            elif kernel_initializer.lower() == 'uniform':
                nn.init.uniform_(conv_layer.weight)
            elif kernel_initializer.lower() == 'orthogonal':
                nn.init.orthogonal_(conv_layer.weight)
            else:  # TODO: add more activations
                raise ValueError(f"initializer {kernel_initializer} not supported")
            
        self.layers.append(conv_layer)

        if bn:
            self.layers.append(nn.BatchNorm1d(out_channels))

        if isinstance(activation, str):
            activation = activation.lower()

        if callable(activation):
            self.layers.append(activation)
        elif activation.lower() == "mish":
            self.layers.append(Mish())
        elif activation.lower() == "swish":
            self.layers.append(Swish())
        elif activation.lower() == "relu":
            self.layers.append(nn.ReLU(inplace=True))
        elif activation.lower() in ["leaky", "leaky_relu"]:
            self.layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation.lower() == "linear":
            pass
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")

    def forward(self, x):
        """
        """
        for l in self.layers:
            x = l(x)
        return x


class VGG_BLOCK(nn.Module):
    """
    """
    def __init__(self, num_convs:int, in_channels:int, out_channels:int):
        """
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=3,
                stride=1,
                activation="mish",
                kernel_initializer="he_normal",
                bn=True
            )
        )
        for _ in range(num_convs-1):
            self.layers.append(
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=3,
                    stride=1,
                    activation="mish",
                    kernel_initializer="he_normal",
                    bn=True
                )
            )
        self.layers.append(
            nn.MaxPool1d(3,3)
        )


class ATI_CNN(nn.Module):
    """
    """
    def __init__(self, classes:list, input_len:int, cnn:str='vgg', bidirectional:bool=True):
        """
        """
        super().__init__()
        self.classes = classes
        self.nb_classes = len(classes)
        self.nb_leads = 12
        self.input_len = input_len
        self.bidirectional = bidirectional

        self.vgg_blk_1 = VGG_BLOCK(
            num_convs=2,
            in_channels=self.nb_leads,
            out_channels=64,
        )
        self.vgg_blk_2 = VGG_BLOCK(
            num_convs=2,
            in_channels=64,
            out_channels=128,
        )
        self.vgg_blk_3 = VGG_BLOCK(
            num_convs=3,
            in_channels=128,
            out_channels=256,
        )
        self.vgg_blk_4 = VGG_BLOCK(
            num_convs=3,
            in_channels=256,
            out_channels=512,
        )
        self.vgg_blk_5 = VGG_BLOCK(
            num_convs=3,
            in_channels=512,
            out_channels=512,
        )


    def forward(self, input):
        """
        """
        x = self.vgg_blk_1(input)
        x = self.vgg_blk_2(x)
        x = self.vgg_blk_3(x)
        x = self.vgg_blk_4(x)
        x = self.vgg_blk_5(x)

        # NOT finished yet
