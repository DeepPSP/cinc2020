"""
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

from cfg import ModelCfg
from model_configs.ati_cnn import ATI_CNN_CONFIG
from model_configs.cpsc import CPSC_CONFIG
from models.utils.torch_utils import (
    Mish, Swish, Activations,
    Conv_Bn_Activation,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext,
    compute_conv_output_shape,
)
from utils.misc import dict_to_str


__all__ = [
    "ATI_CNN",
]


class VGGBlock(nn.Sequential):
    """
    """
    def __init__(self, num_convs:int, in_channels:int, out_channels:int, **kwargs) -> NoReturn:
        """
        """
        super().__init__()
        self.__num_convs = num_convs
        self.__in_channels = in_channels
        self.__out_channels = out_channels

        self.config = deepcopy(ATI_CNN_CONFIG.cnn.vgg_block)

        self.add_module(
            "block_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=self.config.filter_length,
                stride=self.config.subsample_length,
                activation=self.config.activation,
                kernel_initializer=self.config.kernel_initializer,
                bn=self.config.batch_norm,
            )
        )
        for idx in range(num_convs-1):
            self.add_module(
                f"block_{idx+2}",
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=self.config.filter_length,
                    stride=self.config.subsample_length,
                    activation=self.config.activation,
                    kernel_initializer=self.config.kernel_initializer,
                    bn=self.config.batch_norm,
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.config.pool_kernel, self.config.pool_stride)
        )

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """
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
                    kernel_size=self.config.pool_kernel,
                    stride=self.config.pool_stride,
                    channel_last=False,
                )
            num_layers += 1
        return output_shape


class VGG6(nn.Sequential):
    """
    """
    def __init__(self, in_channels:int):
        """
        """
        super().__init__()
        self.__in_channels = in_channels
        
        self.config = deepcopy(ATI_CNN_CONFIG.cnn.vgg6)
        for idx, (nc, nf) in enumerate(zip(self.config.num_convs, self.config.num_filters)):
            module_name = f"vgg_block_{idx+1}"
            if idx == 0:
                module_in_channels = in_channels
            else:
                module_in_channels = self.config.num_filters[idx-1]
            module_out_channels = nf
            self.add_module(
                name=module_name,
                module=VGGBlock(
                    num_convs=nc,
                    in_channels=module_in_channels,
                    out_channels=module_out_channels,
                )
            )

    def forward(self, input):
        """
        keep up with `nn.Sequential.forward`
        """
        for module in self:
            input = module(input)
        return input

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """
        """
        for module in self:
            output_shape = module.compute_output_shape(seq_len, batch_size)
            _, _, seq_len = output_shape
        return output_shape


class ResNetBasicBlock(nn.Module):
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
    __name__ = "ResNetBasicBlock"
    expansion = 1

    def __init__(self, in_channels:int, out_channels:int, stride:int=1, downsample:Optional[nn.Module]=None, groups:int=1, base_width:int=64, dilation:Real=1, norm_layer:Optional[nn.Module]=None, **kwargs) -> NoReturn:
        """
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # if groups != 1 or base_width != 64:  # from torchvision
        #     raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(f"Dilation > 1 not supported in {self.__name__}")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        # NOTE that in ref. [2], the conv3x3 and the conv1x1 layers both have `bias = False`
        self.conv_bn_activation_1 = Conv_Bn_Activation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            bn=norm_layer,
            activation="relu",
            kernel_initializer='he_normal',
            bias=False,
        )
        self.conv_bn_activation_2 = Conv_Bn_Activation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            bn=norm_layer,
            activation=None,
            kernel_initializer="he_normal",
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        """
        """
        identity = input

        out = self.conv_bn_activation_1(input)
        out = self.conv_bn_activation_2(out)

        if self.downsample is not None:
            identity = self.downsample(input)

        out += identity
        out = self.activation(out)

        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """
        """
        raise NotImplementedError


class ResNetBottleneck(nn.Module):
    """
    """
    __name__ = "ResNetBottleneck"
    expansion = 4

    def __init__(self, in_channels:int, out_channels:int, stride:int=1, downsample:Optional[nn.Module]=None, groups:int=1, base_width:int=64, dilation:Real=1, norm_layer:Optional[nn.Module]=None, **kwargs) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, input):
        """
        """
        raise NotImplementedError


class ResNet(nn.Module):
    """
    """
    def __init__(self, in_channels:int):
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, input):
        """
        """
        raise NotImplementedError


class ATI_CNN(nn.Module):
    """
    """
    def __init__(self, classes:list, input_len:int, **config):
        """
        """
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len
        self.config = deepcopy(ATI_CNN_CONFIG)
        self.config.update(config)
        nl = "\n"
        print(f"configuration of ATI_CNN is as follows{nl}{dict_to_str(self.config)}")
        
        cnn_choice = self.config.cnn.name.lower()
        if cnn_choice == "vgg6":
            self.cnn = VGG6(self.n_leads)
            rnn_input_size = self.config.cnn.vgg6.num_filters[-1]
        elif cnn_choice == "resnet":
            raise NotImplementedError
        cnn_output_shape = self.cnn.compute_output_shape(input_len, batch_size=None)
        self.cnn_output_len = cnn_output_shape[2]
        print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}")

        rnn_choice = self.config.rnn.name.lower()
        if rnn_choice == 'lstm':
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.hidden_sizes,
                bias=self.config.rnn.bias,
                dropout=self.config.rnn.dropout,
                bidirectional=self.config.rnn.bidirectional,
                return_sequences=self.config.rnn.retseq,
            )
            if self.config.rnn.bidirectional:
                clf_input_size = 2*self.config.rnn.hidden_sizes[-1]
            else:
                clf_input_size = self.config.rnn.hidden_sizes[-1]
            if self.config.rnn.retseq:
                clf_input_size *= self.cnn_output_len
        elif rnn_choice == 'attention':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.clf = nn.Linear(clf_input_size, self.n_classes)


    def forward(self, input:Tensor) -> Tensor:
        """
        """
        x = self.cnn(input)  # batch_size, channels, seq_len
        # input shape of lstm: (seq_len, batch, input_size)
        x = x.permute(2,0,1)  # seq_len, batch_size, channels
        x = self.rnn(x)
        # the directions can be separated using 
        # output.view(seq_len, batch, num_directions, hidden_size), 
        # with forward and backward being direction 0 and 1 respectively
        if self.config.rnn.retseq:
            x, _ = x
            x = x.permute(1,0,2)
            batch_size, seq_len, hidden_size = x.size()
            x = x.view(batch_size, seq_len*hidden_size)
        else:
            x = x[-1, ...]  # `return_sequences=False`, of shape (batch_size, channels)
        pred = self.clf(x)
        return pred


class CPSCBlock(nn.Sequential):
    """
    the best model of CPSC2018
    """
    def __init__(self, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropout:Optional[float]=None, **kwargs) -> NoReturn:
        """
        """
        super().__init__()
        self.__num_convs = len(filter_lengths)
        self.__in_channels = 12
        self.__out_channels = 12
        self.__dropout = dropout or 0.0

        self.config = deepcopy(CPSC_CONFIG.cnn.cpsc_block)
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
                    bn=self.config.batch_norm,
                )
            )
        self.add_module(
            "giant",
            Conv_Bn_Activation(
                self.__in_channels, self.__out_channels,
                kernel_size=filter_lengths[-1],
                stride=subsample_lengths[-1],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                bn=self.config.batch_norm,
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
        """
        """
        n_layers = 0
        for module in self:
            if n_layers >= self.__num_convs:
                break
            output_shape = module.compute_output_shape(seq_len, batch_size)
            _, _, seq_len = output_shape
        return output_shape


class CPSC(nn.Sequential):
    """
    SOTA model of the CPSC2018 challenge
    """
    def __init__(self, classes:list, input_len:int, **config):
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len
        self.config = deepcopy(CPSC_CONFIG)
        self.config.update(config)
        print(f"configuration of CPSC is as follows\n{dict_to_str(self.config)}")

        cnn_choice = self.config.cnn.name.lower()
        if cnn_choice == 'cpsc_2018':
            self.cnn = nn.Sequential()
            cnn_config = self.config.cnn.cpsc
            for blk_idx, (blk_fl, blk_s, blk_d) in enumerate(zip(cnn_config.filter_lengths, cnn_config.strides, cnn_config.dropouts)):
                self.cnn.add_module(
                    f"cpsc_block_{blk_idx+1}",
                    CPSCBlock(
                        filter_lengths=blk_fl,
                        subsample_lengths=blk_s,
                        dropout=blk_d,
                    )
                )
        else:
            raise NotImplementedError

        cnn_output_shape = self.cnn.compute_output_shape()

        self.rnn = nn.Sequential()
        self.rnn.add_module(
            "bidirectional_gru",
            nn.GRU(input_size=12, hidden_size=12, bidirectional=True),
        )
        self.rnn.add_module(
            "leaky",
            Activations["leaky"](negative_slope=0.2),
        )
        self.rnn.add_module(
            "dropout",
            nn.Dropout(0.2),
        )
        self.rnn.add_module(
            "attention",
            AttentionWithContext(12, 12),
        )
        self.rnn.add_module(
            "batch_normalization",
            nn.BatchNorm1d(12),
        )
        self.rnn.add_module(
            "leaky",
            Activations["leaky"](negative_slope=0.2),
        )
        self.rnn.add_module(
            "dropout",
            nn.Dropout(0.2),
        )

        # self.clf = nn.Linear()  # TODO: set correct the in-and-out-features
        

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError
