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
from easydict import EasyDict as ED

from models.utils.torch_utils import (
    Conv_Bn_Activation,
    DownSample,
)


__all__ = [
    "ECG_UNET",
]


class DoubleConv(nn.Sequential):
    """

    building blocks of UNet
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    __DEBUG__ = True
    __name__ = "DoubleConv"

    def __init__(self, in_channels:int, out_channels:int, filter_length:int, activation:Union[str,nn.Module]='relu', mid_channels:Optional[int]=None) -> NoReturn:
        """ finished, NOT checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the last convolutional layer
        filter_length: int,
            length of the filters (kernel size)
        activation: str or Module, default 'relu',
            activation of the convolutional layers
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.__kernel_size = filter_length

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.__mid_channels,
                kernel_size=self.__kernel_size,
                stride=1,
                bn=True,
                activation=activation,
            ),
        )
        self.add_module(
            "cba_2",
            Conv_Bn_Activation(
                in_channels=self.__mid_channels,
                out_channels=self.__out_channels,
                kernel_size=self.__kernel_size,
                stride=1,
                bn=True,
                activation=activation,
            )
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `DoubleConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class DownDoubleConv(nn.Sequential):
    """
    Downscaling with maxpool then double conv
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    __DEBUG__ = True
    __name__ = "DownDoubleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(self, down_scale:int, in_channels:int, out_channels:int, filter_length:int, activation:Union[str, nn.Module]='relu', mid_channels:Optional[int]=None, mode:str='max') -> NoReturn:
        """ finished, NOT checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the last convolutional layer
        filter_length: int,
            length of the filters (kernel size)
        activation: str or Module, default 'relu',
            activation of the convolutional layers
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        mode: str, default 'max',
            mode for down sampling,
            can be one of 'max', 'avg', 'conv', 'nearest', 'linear', 'bilinear'
        """
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.__kernel_size = filter_length

        self.add_module(
            "down_sample",
            DownSample(
                down_scale=self.__down_scale,
                in_channels=self.__in_channels,
                bn=False,
                mode=mode,
            )
        )
        self.add_module(
            "double_conv",
            DoubleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_length=self.__kernel_size,
                activation=activation,
                mid_channels=self.__mid_channels,
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(seq_len=_seq_len)
            _, _, _seq_len = output_shape
        return output_shape


class UpDoubleConv(nn.Module):
    """
    Upscaling then double conv (up sampling --> double convolution)
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    __DEBUG__ = True
    __name__ = "UpDoubleConv"
    __MODES__ = ['nearest', 'linear', 'bilinear', 'conv',]

    def __init__(self, up_scale:int, in_channels:int, out_channels:int, filter_length:int, activation:Union[str,nn.Module]='relu', mode:str='bilinear', mid_channels:Optional[int]=None) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters:
        -----------
        up_scale: int,
            scale of up sampling
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        filter_length: int,
            length of the filters (kernel size)
        activation: str or Module, default 'relu',
            activation of the convolutional layers
        mode: str, default 'bilinear', case insensitive,
            mode of up sampling
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        """
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else in_channels // 2
        self.__out_channels = out_channels
        self.__kernel_size = filter_length
        self.__mode == mode.lower()
        assert self.__mode in self.__MODES__

        # the following has to be checked
        # if bilinear, use the normal convolutions to reduce the number of channels
        if self.__mode == 'conv':
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(
                scale_factor=self.__up_scale,
                mode=mode,
            )
            self.conv = DoubleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_length=self.__kernel_size,
                activation=activation,
                mid_channels=self.__mid_channels,
            )

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        """

        Parameters:
        -----------
        to write
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ NOT finished,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        raise NotImplementedError


class ECG_UNET(nn.Module):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_UNET"
    
    def __init__(self, classes:Sequence[str], n_leads:int, config:dict) -> NoReturn:
        """
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)  # final out_channels
        self.__in_channels = n_leads
        self.config = ED(deepcopy(config))
        raise NotImplementedError

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ NOT finished,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        raise NotImplementedError
