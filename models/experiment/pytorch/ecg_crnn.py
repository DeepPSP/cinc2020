"""
CRNN structure models,
for classifying ECG arrhythmias
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
    Bn_Activation, Conv_Bn_Activation,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext,
    compute_conv_output_shape,
)
from utils.misc import dict_to_str


__all__ = [
    "ATI_CNN",
    "CPSC",
]


class VGGBlock(nn.Sequential):
    """
    """
    def __init__(self, num_convs:int, in_channels:int, out_channels:int, **config) -> NoReturn:
        """
        """
        super().__init__()
        self.__num_convs = num_convs
        self.__in_channels = in_channels
        self.__out_channels = out_channels

        # self.config = deepcopy(ATI_CNN_CONFIG.cnn.vgg_block)
        # self.config.update(config)
        self.config = ED(config)

        self.add_module(
            "cba_1",
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
                f"cba_{idx+2}",
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
            nn.MaxPool1d(self.config.pool_size)
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
    def __init__(self, in_channels:int, **config):
        """
        """
        super().__init__()
        self.__in_channels = in_channels
        
        # self.config = deepcopy(ATI_CNN_CONFIG.cnn.vgg6)
        self.config = ED(config)

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
                    **(config.block),
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


class ResNetStanfordBlock(nn.Module):
    """
    """
    def __init__(self, in_channels:int, num_filters:int, pool_size:int, stride:int=1, dilation:Real=1, block_index:int, **config) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__pool_size = pool_size
        self.__stride = stride
        self.__dilation = dilation
        self.__block_index = block_index
        self.config = ED(config)

        self.short_cut = nn.MaxPool1d(self.__pool_size)
        self.__zero_pad = (block_index % self.config.increase_channels_at) == 0 \
            and block_index > 0
        
        self.main_stream = nn.Sequential()
        num_cba_layer = 1
        cba_in_channels = self.__in_channels
        for i in range(self.config.num_skip):
            if not (block_index == 0 and i == 0):
                self.main_stream.add_module(
                    "ba",
                    Bn_Activation(
                        num_features=self.__in_channels,
                        activation=self.config.activation,
                        dropout=self.config.dropout,
                    ),
                )
            self.main_stream.add_module(
                f"cba_{num_cba_layer}",
                Conv_Bn_Activation(
                    in_channels=cba_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.config.filter_length,
                    stride = (self.__stride if i == 0 else 1),
                    activation=self.config.activation,
                    kernel_initializer=self.config.init,
                )
            )
            num_cba_layer += 1
            cba_in_channels = self.__out_channels

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        sc = self.short_cut.forward(input)
        if self.__zero_pad:
            sc = self.zero_pad(sc)
        output = self.main_stream.forward(input) + sc
        return output

    def zero_pad(self, x:Tensor) -> Tensor:
        """
        """
        out = torch.zeros_like(x)
        out = torch.cat((x, out), dim=1)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """
        """
        _seq_len = seq_len
        for module in self.main_stream:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
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

        Parameters:
        -----------
        to write
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
        self.cba_1 = Conv_Bn_Activation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            bn=norm_layer,
            activation="relu",
            kernel_initializer='he_normal',
            bias=False,
        )
        self.cba_2 = Conv_Bn_Activation(
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

        out = self.cba_1(input)
        out = self.cba_2(out)

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
            self.cnn = VGG6(self.n_leads, **(self.config.cnn.vgg6))
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


class CPSCMiniBlock(nn.Sequential):
    """
    building block of the SOTA model of CPSC2018
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
            n_layers += 1
        return output_shape


class CPSCBlock(nn.Sequential):
    """
    CNN part of the SOTA model of the CPSC2018 challenge
    """
    def __init__(self, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropouts:Optional[float]=None, **kwargs) -> NoReturn:
        """
        """
        super().__init__()
        for blk_idx, (blk_fl, blk_s, blk_d) in enumerate(zip(filter_lengths, subsample_lengths, dropouts)):
            self.add_module(
                f"cpsc_mini_block_{blk_idx+1}",
                CPSCMiniBlock(
                    filter_lengths=blk_fl,
                    subsample_lengths=blk_s,
                    dropout=blk_d,
                )
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
        for module in self:
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
            cnn_config = self.config.cnn.cpsc
            self.cnn = CPSCBlock(
                filter_lengths=cnn_config.filter_lengths,
                subsample_lengths=cnn_config.strides,
                dropouts=cnn_config.dropouts,
            )
        else:
            raise NotImplementedError

        cnn_output_shape = self.cnn.compute_output_shape(self.input_len)

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
