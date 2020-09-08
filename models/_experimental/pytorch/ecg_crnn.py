"""
CRNN structure models,
for classifying ECG arrhythmias
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

torch.set_default_tensor_type(torch.DoubleTensor)

from cfg import ModelCfg
from model_configs.ati_cnn import ATI_CNN_CONFIG
from model_configs.cpsc import CPSC_CONFIG
from models.utils.torch_utils import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext, MultiHeadAttention,
)
from utils.utils_nn import compute_conv_output_shape
from utils.misc import dict_to_str


if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_CRNN",
    "VGGBlock", "VGG16",
    "CPSCBlock", "CPSCCNN",
    "ResNetBasicBlock", "ResNetBottleNeck", "ResNet",
    "MultiScopicBasicBlock", "MultiScopicBranch", "MultiScopicCNN",
]


class VGGBlock(nn.Sequential):
    """
    building blocks of the CNN feature extractor `VGG16`
    """
    __DEBUG__ = True
    __name__ = "VGGBlock"

    def __init__(self, num_convs:int, in_channels:int, out_channels:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        num_convs: int,
            number of convolutional layers of this block
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other parameters, including
            filter length (kernel size), activation choices,
            weight initializer, batch normalization choices, etc. for the convolutional layers;
            and pool size for the pooling layer
        """
        super().__init__()
        self.__num_convs = num_convs
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__groups = groups
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=self.config.filter_length,
                stride=self.config.subsample_length,
                groups=self.__groups,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
                batch_norm=self.config.batch_norm,
            )
        )
        for idx in range(num_convs-1):
            self.add_module(
                f"cba_{idx+2}",
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=self.config.filter_length,
                    stride=self.config.subsample_length,
                    groups=self.__groups,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    batch_norm=self.config.batch_norm,
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.config.pool_size, self.config.pool_stride)
        )

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        num_layers = 0
        for module in self:
            if num_layers < self.__num_convs:
                output_shape = module.compute_output_shape(seq_len, batch_size)
                _, _, seq_len = output_shape
            else:
                output_shape = compute_conv_output_shape(
                    input_shape=[batch_size, self.__out_channels, seq_len],
                    num_filters=self.__out_channels,
                    kernel_size=self.config.pool_size,
                    stride=self.config.pool_size,
                    channel_last=False,
                )
            num_layers += 1
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class VGG16(nn.Sequential):
    """
    CNN feature extractor of the CRNN models proposed in refs of `ECG_CRNN`
    """
    __DEBUG__ = True
    __name__ = "VGG16"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, including
            number of convolutional layers, number of filters for each layer,
            and more for `VGGBlock`
        """
        super().__init__()
        self.__in_channels = in_channels
        # self.config = deepcopy(ECG_CRNN_CONFIG.cnn.vgg16)
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        module_in_channels = in_channels
        for idx, (nc, nf) in enumerate(zip(self.config.num_convs, self.config.num_filters)):
            module_name = f"vgg_block_{idx+1}"
            self.add_module(
                name=module_name,
                module=VGGBlock(
                    num_convs=nc,
                    in_channels=module_in_channels,
                    out_channels=nf,
                    groups=self.config.groups,
                    **(self.config.block),
                )
            )
            module_in_channels = nf

    def forward(self, input):
        """
        keep up with `nn.Sequential.forward`
        """
        for module in self:
            input = module(input)
        return input

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        for module in self:
            output_shape = module.compute_output_shape(seq_len, batch_size)
            _, _, seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ResNetBasicBlock(nn.Module):
    """

    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`
    """
    __DEBUG__ = True
    __name__ = "ResNetBasicBlock"
    expansion = 1

    def __init__(self, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, groups:int=1, dilation:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        num_filters: int,
            number of filters for the convolutional layers
        filter_length: int,
            length (size) of the filter kernels
        subsample_lengths: int,
            subsample length,
            including pool size for short cut, and stride for the top convolutional layer
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        dilation: int, default 1,
            not used
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer,
            and short cut patterns, etc.
        """
        super().__init__()
        if dilation > 1:
            raise NotImplementedError(f"Dilation > 1 not supported in {self.__name__}")
        self.__num_convs = 2
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__kernel_size = filter_length
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.__groups = groups
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        if self.config.increase_channels_method.lower() == 'zero_padding' and self.__groups != 1:
            raise ValueError("zero padding for increasing channels can not be used with groups != 1")
        
        self.__increase_channels = (self.__out_channels > self.__in_channels)
        self.short_cut = self._make_short_cut_layer()

        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            conv_activation = (self.config.activation if i < self.__num_convs-1 else None)
            self.main_stream.add_module(
                f"cba_{i}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.__kernel_size,
                    stride=(self.__stride if i == 0 else 1),
                    groups=self.__groups,
                    batch_norm=True,
                    activation=conv_activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = self.__out_channels

        if isinstance(self.config.activation, str):
            self.out_activation = \
                Activations[self.config.activation.lower()](**self.config.kw_activation)
        else:
            self.out_activation = \
                self.config.activation(**self.config.kw_activation)
    
    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == 'conv':
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    groups=self.__groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            if self.config.increase_channels_method.lower() == 'zero_padding':
                batch_norm = False if self.config.subsample_mode.lower() != 'conv' else True
                short_cut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        batch_norm=batch_norm,
                        mode=self.config.subsample_mode,
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels),
                )
        else:
            short_cut = None
        return short_cut

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        identity = input

        out = self.main_stream(input)

        if self.short_cut is not None:
            identity = self.short_cut(input)

        out += identity
        out = self.out_activation(out)

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self.main_stream:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ResNetBottleNeck(nn.Module):
    """
    to write
    """
    __DEBUG__ = True
    __name__ = "ResNetBottleNeck"
    expansion = 4

    def __init__(self, in_channels:int, num_filters:int, subsample_length:int, groups:int=1, dilation:int=1, **config) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        num_filters: int,
            number of filters for the convolutional layers
        subsample_length: int,
            subsample length,
            including pool size for short cut, and stride for the top convolutional layer
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer,
            and short cut patterns, etc.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        raise NotImplementedError

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ResNet(nn.Sequential):
    """

    References:
    -----------
    [1] https://github.com/awni/ecg
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    TODO:
    -----
    1. check performances of activations other than "nn.ReLU", especially mish and swish
    2. to add
    """
    __DEBUG__ = True
    __name__ = "ResNet"
    building_block = ResNetBasicBlock

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        # self.__building_block = \
        #     ResNetBasicBlock if self.config.name == 'resnet' else ResNetBottleNeck
        
        self.add_module(
            "init_cba",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_length,
                stride=self.config.init_conv_stride,
                groups=self.config.groups,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
                bias=self.config.bias,
            )
        )
        
        if self.config.init_pool_size > 0:
            self.add_module(
                "init_pool",
                nn.MaxPool1d(
                    kernel_size=self.config.init_pool_size,
                    stride=self.config.init_pool_stride,
                    padding=(self.config.init_pool_size-1)//2,
                )
            )

        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = \
                list(repeat(self.config.filter_lengths, len(self.config.num_blocks)))
        else:
            self.__filter_lengths = self.config.filter_lengths
            assert len(self.__filter_lengths) == len(self.config.num_blocks), \
                f"`config.filter_lengths` indicates {len(self.__filter_lengths)} macro blocks, while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = \
                list(repeat(self.config.subsample_lengths, len(self.config.num_blocks)))
        else:
            self.__subsample_lengths = self.config.subsample_lengths
            assert len(self.__subsample_lengths) == len(self.config.num_blocks), \
                f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)} macro blocks, while `config.num_blocks` indicates {len(self.config.num_blocks)}"

        # grouped resnet (basic) blocks,
        # number of channels are doubled at the first block of each macro-block
        for macro_idx, nb in enumerate(self.config.num_blocks):
            macro_in_channels = (2**macro_idx) * self.config.init_num_filters
            macro_filter_lengths = self.__filter_lengths[macro_idx]
            macro_subsample_lengths = self.__subsample_lengths[macro_idx]
            block_in_channels = macro_in_channels
            block_num_filters = 2 * block_in_channels
            if isinstance(macro_filter_lengths, int):
                block_filter_lengths = list(repeat(macro_filter_lengths, nb))
            else:
                block_filter_lengths = macro_filter_lengths
            assert len(block_filter_lengths) == nb, \
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            if isinstance(macro_subsample_lengths, int):
                block_subsample_lengths = list(repeat(1, nb))
                block_subsample_lengths[-1] = macro_subsample_lengths
            else:
                block_subsample_lengths = macro_subsample_lengths
            assert len(block_subsample_lengths) == nb, \
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            for block_idx in range(nb):
                self.add_module(
                    f"block_{macro_idx}_{block_idx}",
                    self.building_block(
                        in_channels=block_in_channels,
                        num_filters=block_num_filters,
                        filter_length=block_filter_lengths[block_idx],
                        subsample_length=block_subsample_lengths[block_idx],
                        groups=self.config.groups,
                        dilation=1,
                        **(self.config.block)
                    )
                )
                block_in_channels = block_num_filters

    def forward(self, input):
        """
        """
        output = super().forward(input)
        return output

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if type(module).__name__ == "MaxPool1d":
                output_shape = compute_conv_output_shape(
                    input_shape=(batch_size, self.config.init_filter_length, _seq_len),
                    num_filters=self.config.init_filter_length,
                    kernel_size=self.config.init_pool_size,
                    stride=self.config.init_pool_stride,
                    padding=(self.config.init_pool_size-1)//2,
                )
            else:
                output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class CPSCBlock(nn.Sequential):
    """
    building block of the SOTA model of CPSC2018 challenge
    """
    __DEBUG__ = True
    __name__ = "CPSCBlock"

    def __init__(self, in_channels:int, num_filters:int, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropout:Optional[float]=None, **config) -> NoReturn:
        """

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        num_filters:int,
            number of filters for the convolutional layers
        filter_lengths: sequence of int,
            filter length (kernel size) of each convolutional layer
        subsample_lengths: sequence of int,
            subsample length (stride) of each convolutional layer
        dropout: float, optional,
            if positive, a `Dropout` layer will be introduced with this dropout probability
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer, etc.
        """
        super().__init__()
        self.__num_convs = len(filter_lengths)
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__dropout = dropout or 0.0
        self.config = deepcopy(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        for idx, (kernel_size, stride) in enumerate(zip(filter_lengths[:-1], subsample_lengths[:-1])):
            self.add_module(
                f"baby_{idx+1}",
                Conv_Bn_Activation(
                    self.__in_channels, self.__out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    batch_norm=self.config.batch_norm,
                )
            )
        self.add_module(
            "giant",
            Conv_Bn_Activation(
                self.__out_channels, self.__out_channels,
                kernel_size=filter_lengths[-1],
                stride=subsample_lengths[-1],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                batch_norm=self.config.batch_norm,
            )
        )
        if self.__dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.__dropout),
            )

    def forward(self, input:Tensor) -> Tensor:
        """
        keep up with `nn.Sequential.forward`
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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        n_layers = 0
        _seq_len = seq_len
        for module in self:
            if n_layers >= self.__num_convs:
                break
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
            n_layers += 1
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class CPSCCNN(nn.Sequential):
    """
    CNN part of the SOTA model of the CPSC2018 challenge
    """
    __DEBUG__ = True
    __name__ = "CPSCCNN"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        num_filters = self.config.num_filters
        filter_lengths = self.config.filter_lengths
        subsample_lengths = self.config.subsample_lengths
        dropouts = self.config.dropouts
        blk_in = self.__in_channels
        for blk_idx, (blk_nf, blk_fl, blk_sl, blk_dp) \
            in enumerate(zip(num_filters, filter_lengths, subsample_lengths, dropouts)):
            self.add_module(
                f"cpsc_block_{blk_idx+1}",
                CPSCBlock(
                    in_channels=blk_in,
                    num_filters=blk_nf,
                    filter_lengths=blk_fl,
                    subsample_lengths=blk_sl,
                    dropout=blk_dp,
                )
            )
            blk_in = blk_nf[-1]

    def forward(self, input:Tensor) -> Tensor:
        """
        keep up with `nn.Sequential.forward`
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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiScopicBasicBlock(nn.Sequential):
    """
    basic building block of the CNN part of the SOTA model from CPSC2019 challenge (entry 0416)

    (conv -> activation) * N --> bn --> down_sample
    """
    __DEBUG__ = True
    __name__ = "MultiScopicBasicBlock"

    def __init__(self, in_channels:int, scopes:Sequence[int], num_filters:Union[int,Sequence[int]], filter_lengths:Union[int,Sequence[int]], subsample_length:int, groups:int=1, **config) -> NoReturn:
        """ finished, not checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        scopes: sequence of int,
            scopes of the convolutional layers, via `dilation`
        num_filters: int or sequence of int,
        filter_lengths: int or sequence of int,
        subsample_length: int,
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_convs = len(self.__scopes)
        if isinstance(num_filters, int):
            self.__out_channels = list(repeat(num_filters, self.__num_convs))
        else:
            self.__out_channels = num_filters
            assert len(self.__out_channels) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `num_filters` indicates {len(self.__out_channels)}"
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = filter_lengths
            assert len(self.__filter_lengths) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        self.__subsample_length = subsample_length
        self.__groups = groups
        self.config = ED(deepcopy(config))

        conv_in_channels = self.__in_channels
        for idx in range(self.__num_convs):
            self.add_module(
                f"ca_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels[idx],
                    kernel_size=self.__filter_lengths[idx],
                    stride=1,
                    dilation=self.__scopes[idx],
                    groups=self.__groups,
                    batch_norm=False,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = self.__out_channels[idx]
        self.add_module(
            "bn",
            nn.BatchNorm1d(self.__out_channels[-1])
        )
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels[-1],
                groups=self.__groups,
                # padding=
                batch_norm=False,
                mode=self.config.subsample_mode,
            )
        )
        if self.config.dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.config.dropout, inplace=False)
            )

    def forward(self, input:Tensor) -> Tensor:
        """
        input: of shape (batch_size, channels, seq_len)
        """
        output = super().forward(input)
        return output

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for idx, module in enumerate(self):
            if idx == self.__num_convs:  # bn layer
                continue
            elif self.config.dropout > 0 and idx == len(self)-1:  # dropout layer
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiScopicBranch(nn.Sequential):
    """
    branch path of the CNN part of the SOTA model from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = True
    __name__ = "MultiScopicBranch"

    def __init__(self, in_channels:int, scopes:Sequence[Sequence[int]], num_filters:Union[Sequence[int],Sequence[Sequence[int]]], filter_lengths:Union[Sequence[int],Sequence[Sequence[int]]], subsample_lengths:Union[int,Sequence[int]], groups:int=1, **config):
        """

        Parameters:
        -----------
        in_channels
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_blocks = len(self.__scopes)
        self.__num_filters = num_filters
        assert len(self.__num_filters) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `num_filters` indicates {len(self.__num_filters)}"
        self.__filter_lengths = filter_lengths
        assert len(self.__filter_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `filter_lengths` indicates {llen(self.__filter_lengths)}"
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(repeat(subsample_lengths, self.__num_blocks))
        else:
            self.__subsample_lengths = filter_lengths
            assert len(self.__subsample_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `subsample_lengths` indicates {llen(self.__subsample_lengths)}"
        self.__groups = groups
        self.config = ED(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx in range(self.__num_blocks):
            self.add_module(
                f"block_{idx}",
                MultiScopicBasicBlock(
                    in_channels=block_in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.__num_filters[idx],
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.__groups,
                    dropout=self.config.dropouts[idx],
                    **(self.config.block)
                )
            )
            block_in_channels = self.__num_filters[idx]

    def forward(self, input:Tensor) -> Tensor:
        """
        input: of shape (batch_size, channels, seq_len)
        """
        output = super().forward(input)
        return output

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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for idx, module in enumerate(self):
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiScopicCNN(nn.Module):
    """
    CNN part of the SOTA model from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = True
    __name__ = "MultiScopicCNN"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        self.__scopes = self.config.scopes
        self.__num_branches = len(self.__scopes)

        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.branches = nn.ModuleDict()
        for idx in range(self.__num_branches):
            self.branches[f"branch_{idx}"] = \
                MultiScopicBranch(
                    in_channels=self.__in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.config.num_filters[idx],
                    filter_lengths=self.config.filter_lengths[idx],
                    subsample_lengths=self.config.subsample_lengths[idx],
                    dropouts=self.config.dropouts[idx],
                    block=self.config.block,  # a dict
                )

    def forward(self, input:Tensor) -> Tensor:
        """
        input: of shape (batch_size, channels, seq_len)
        """
        branch_out = OrderedDict()
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            branch_out[key] = self.branches[key].forward(input)
        output = torch.cat(
            [branch_out[f"branch_{idx}"] for idx in range(self.__num_branches)],
            dim=1,  # along channels
        )
        return output
    
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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        out_channels = 0
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            _, _branch_oc, _seq_len = \
                self.branches[key].compute_output_shape(seq_len, batch_size)
            out_channels += _branch_oc
        return (batch_size, out_channels, _seq_len)

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ECG_CRNN(nn.Module):
    """

    C(R)NN models modified from the following refs.

    References:
    -----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    [3] Hannun, Awni Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65.
    [4] https://stanfordmlgroup.github.io/projects/ecg2/
    [5] https://github.com/awni/ecg
    [6] CPSC2018 entry 0236
    [7] CPSC2019 entry 0416
    """
    __DEBUG__ = True
    __name__ = 'ECG_CRNN'

    def __init__(self, classes:Sequence[str], input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        input_len: int, optional,
            sequence length (last dim.) of the input,
            defaults to `TrainCfg.input_len`,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len or TrainCfg.input_len
        self.config = deepcopy(ECG_CRNN_CONFIG)
        self.config.update(config or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        
        cnn_choice = self.config.cnn.name.lower()
        if "vgg16" in cnn_choice:
            self.cnn = VGG16(self.n_leads, **(self.config.cnn[cnn_choice]))
            rnn_input_size = self.config.cnn.vgg16.num_filters[-1]
        elif "resnet" in cnn_choice:
            self.cnn = ResNet(self.n_leads, **(self.config.cnn[cnn_choice]))
            rnn_input_size = \
                2**len(self.config.cnn.resnet.num_blocks) * self.config.cnn.resnet.init_num_filters
        elif "multi_scopic" in cnn_choice:
            self.cnn = MultiScopicCNN(self.n_leads, **(self.config.cnn[cnn_choice]))
            rnn_input_size = self.cnn.compute_output_shape(None, None)[1]
        else:
            raise NotImplementedError
        # self.cnn_output_len = cnn_output_shape[2]
        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(self.input_len, batch_size=None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}")

        if self.config.rnn.name.lower() == 'none':
            self.rnn = None
            _, clf_input_size, _ = self.cnn.compute_output_shape(self.input_len, batch_size=None)
            self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.rnn.name.lower() == 'lstm':
            hidden_sizes = self.config.rnn.lstm.hidden_sizes + [self.n_classes]
            if self.__DEBUG__:
                print(f"lstm hidden sizes {self.config.rnn.lstm.hidden_sizes} ---> {hidden_sizes}")
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropout=self.config.rnn.lstm.dropout,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=self.config.rnn.lstm.retseq,
            )
            if self.config.rnn.lstm.retseq:
                self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
            else:
                self.max_pool = None
            clf_input_size = self.rnn.compute_output_shape(None,None)[-1]
        elif self.config.rnn.name.lower() == 'attention':
            hidden_sizes = self.config.rnn.attention.hidden_sizes
            attn_in_channels = hidden_sizes[-1]
            if self.config.rnn.attention.bidirectional:
                attn_in_channels *= 2
            self.rnn = nn.Sequential(
                StackedLSTM(
                    input_size=rnn_input_size,
                    hidden_sizes=hidden_sizes,
                    bias=self.config.rnn.attention.bias,
                    dropout=self.config.rnn.attention.dropout,
                    bidirectional=self.config.rnn.attention.bidirectional,
                    return_sequences=True,
                ),
                SelfAttention(
                    in_features=attn_in_channels,
                    head_num=self.config.rnn.attention.head_num,
                    dropout=self.config.rnn.attention.dropout,
                    bias=self.config.rnn.attention.bias,
                )
            )
            self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
            clf_input_size = self.rnn[-1].compute_output_shape(None,None)[-1]
        else:
            raise NotImplementedError

        if self.__DEBUG__:
            print(f"clf_input_size = {clf_input_size}")

        # input of `self.clf` has shape: batch_size, channels
        self.clf = nn.Linear(clf_input_size, self.n_classes)
        self.sigmoid = nn.Sigmoid()  # for making inference

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        x = self.cnn(input)  # batch_size, channels, seq_len
        # print(f"cnn out shape = {x.shape}")
        if self.rnn:
            # (batch_size, channels, seq_len) -> (seq_len, batch_size, input_size)
            x = x.permute(2,0,1)
            x = self.rnn(x)
            if self.max_pool:
                # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
                x = x.permute(1,2,0)
                x = self.max_pool(x)  # (batch_size, channels, 1)
                # x = torch.flatten(x, start_dim=1)  # (batch_size, channels)
                x = x.squeeze(dim=-1)
            else:
                # x of shape (batch_size, channels)
                pass
            # print(f"rnn out shape = {x.shape}")
        else:
            # (batch_size, channels, seq_len) --> (batch_size, channels)
            x = self.max_pool(x)
            # print(f"max_pool out shape = {x.shape}")
            # x = torch.flatten(x, start_dim=1)
            x = x.squeeze(dim=-1)
        # print(f"clf in shape = {x.shape}")
        pred = self.clf(x)  # batch_size, n_classes
        return pred

    @torch.no_grad()
    def inference(self, input:Tensor, class_names:bool=False, bin_pred_thr:float=0.5) -> Union[np.ndarray, pd.DataFrame]:
        """
        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        pred = self.forward(input)
        pred = self.sigmoid(pred)
        bin_pred = (pred>=bin_pred_thr).int()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()
        for row_idx, row in enumerate(bin_pred):
            row_max_prob = pred[row_idx,...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                bin_pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                bin_pred[row_idx,...] = \
                    (((pred[row_idx,...]+ModelCfg.bin_pred_look_again_tol) >= row_max_prob) & (pred[row_idx,...] >= ModelCfg.bin_pred_nsr_thr)).astype(int)
        if class_names:
            pred = pd.DataFrame(pred)
            pred.columns = self.classes
            # pred['bin_pred'] = pred.apply(
            #     lambda row: np.array(self.classes)[np.where(row.values>=bin_pred_thr)[0]],
            #     axis=1
            # )
            pred['bin_pred'] = ''
            for row_idx in range(len(pred)):
                pred.at[row_idx, 'bin_pred'] = \
                    np.array(self.classes)[np.where(bin_pred==1)[0]].tolist()
            return pred
        return pred, bin_pred

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params
