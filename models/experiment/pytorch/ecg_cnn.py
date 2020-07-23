"""
"""
import sys
from typing import Union, NoReturn, Optional

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


class Conv_Bn_Activation(nn.Sequential):
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
        self.add_module("conv1d", conv_layer)

        if bn:
            self.add_module("batch_norm", nn.BatchNorm1d(out_channels))

        if isinstance(activation, str):
            activation = activation.lower()

        if callable(activation):
            act_layer = activation
            act_name = "activation"
        elif activation.lower() == "mish":
            act_layer = Mish()
            act_name = "mish"
        elif activation.lower() == "swish":
            act_layer = Swish()
            act_name = "swish"
        elif activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
            act_name = "relu"
        elif activation.lower() in ["leaky", "leaky_relu"]:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
            act_name = "leaky_relu"
        elif activation.lower() == "linear":
            act_layer = None
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")
            act_layer = None

        if act_layer:
            self.add_module(act_name, act_layer)


class VGG_BLOCK(nn.Sequential):
    """
    """
    def __init__(self, num_convs:int, in_channels:int, out_channels:int) -> NoReturn:
        """
        """
        super().__init__()

        self.add_module(
            "block_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=3,
                stride=1,
                activation="mish",
                kernel_initializer="he_normal",
                bn=True
            )
        )
        for idx in range(num_convs-1):
            self.add_module(
                f"block_{idx+2}",
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=3,
                    stride=1,
                    activation="mish",
                    kernel_initializer="he_normal",
                    bn=True
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(3,3)
        )


class VGG6(nn.Sequential):
    """
    """
    def __init__(self, in_channels:int):
        """
        """
        super().__init__()
        self.add_module(
            "vgg_block_1",
            VGG_BLOCK(
                num_convs=2,
                in_channels=in_channels,
                out_channels=64,
            )
        )
        self.add_module(
            "vgg_block_2",
            VGG_BLOCK(
                num_convs=2,
                in_channels=64,
                out_channels=128,
            )
        )
        self.add_module(
            "vgg_block_3",
            VGG_BLOCK(
                num_convs=3,
                in_channels=128,
                out_channels=256,
            )
        )
        self.add_module(
            "vgg_block_4",
            VGG_BLOCK(
                num_convs=3,
                in_channels=256,
                out_channels=512,
            )
        )
        self.add_module(
            "vgg_block_5",
            VGG_BLOCK(
                num_convs=3,
                in_channels=512,
                out_channels=512,
            )
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
        cnn_choice = cnn.lower()
        self.bidirectional = bidirectional

        if cnn_choice == 'vgg':
            self.cnn = VGG6(self.nb_leads)


    def forward(self, input):
        """
        """
        x = self.vgg_blk_1(input)

        # NOT finished yet
