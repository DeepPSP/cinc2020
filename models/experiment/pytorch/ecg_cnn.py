"""
"""
import sys
from typing import Union, NoReturn, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from models.utils.torch_utils import (
    Mish, Swish,
    Conv_Bn_Activation,
    Attention, GatedAttention,
)


class VGGBlock(nn.Sequential):
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
            VGGBlock(
                num_convs=2,
                in_channels=in_channels,
                out_channels=64,
            )
        )
        self.add_module(
            "vgg_block_2",
            VGGBlock(
                num_convs=2,
                in_channels=64,
                out_channels=128,
            )
        )
        self.add_module(
            "vgg_block_3",
            VGGBlock(
                num_convs=3,
                in_channels=128,
                out_channels=256,
            )
        )
        self.add_module(
            "vgg_block_4",
            VGGBlock(
                num_convs=3,
                in_channels=256,
                out_channels=512,
            )
        )
        self.add_module(
            "vgg_block_5",
            VGGBlock(
                num_convs=3,
                in_channels=512,
                out_channels=512,
            )
        )


class ResBlock(nn.Module):
    """
    """
    def __init__(self,):
        """
        """
        super().__init__()
        raise NotImplementedError


class ResNet(nn.Module):
    """
    """
    def __init__(self, in_channels:int):
        """
        """
        super().__init__()
        raise NotImplementedError


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
        elif cnn_choice == 'resnet':
            raise NotImplementedError


    def forward(self, input):
        """
        """
        x = self.cnn(input)

        # NOT finished yet
