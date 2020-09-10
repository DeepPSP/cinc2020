"""
Sequence labeling nets, for wave delineation,

the labeling granularity is the frequency of the input signal,
divided by the length (counted by the number of basic blocks) of each branch
"""

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)

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

from cfg import ModelCfg
# from model_configs import ECG_CRNN_CONFIG
from models.utils.torch_utils import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext,
    SelfAttention, MultiHeadAttention,
    AttentivePooling,
    SeqLin,
)
from utils.utils_nn import compute_conv_output_shape
from utils.misc import dict_to_str


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

    def __init__(self, in_channels:int, scopes:Sequence[Sequence[int]], num_filters:Union[Sequence[int],Sequence[Sequence[int]]], filter_lengths:Union[Sequence[int],Sequence[Sequence[int]]], subsample_lengths:Union[int,Sequence[int]], groups:int=1, **config) -> NoReturn:
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


class ECG_SEQ_LAB_NET(nn.module):
    """
    """
    def __init__(self, classes:Sequence[str], config:dict) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for sequence labeling
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.n_classes = len(classes)
        self.n_leads = 12
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
            __debug_seq_len = 4000
        
        # currently, the CNN part only uses `MultiScopicCNN`
        self.cnn = MultiScopicCNN(self.n_leads, **(self.config.cnn[cnn_choice]))
        rnn_input_size = self.cnn.compute_output_shape(self.input_len, batch_size=None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(self.input_len, batch_size=None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}")

        # if self.config.rnn.name.lower() == 'linear':
        #     self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        #     self.rnn = SeqLin(
        #         in_channels=rnn_input_size,
        #         out_channels=self.config.rnn.linear.out_channels,
        #         activation=self.config.rnn.linear.activation,
        #         bias=self.config.rnn.linear.bias,
        #         dropouts=self.config.rnn.linear.dropouts,
        #     )
        #     clf_input_size = self.rnn.compute_output_shape(None,None)[-1]
        # elif self.config.rnn.name.lower() == 'lstm':
        #     hidden_sizes = self.config.rnn.lstm.hidden_sizes + [self.n_classes]
        #     if self.__DEBUG__:
        #         print(f"lstm hidden sizes {self.config.rnn.lstm.hidden_sizes} ---> {hidden_sizes}")
        #     self.rnn = StackedLSTM(
        #         input_size=rnn_input_size,
        #         hidden_sizes=hidden_sizes,
        #         bias=self.config.rnn.lstm.bias,
        #         dropout=self.config.rnn.lstm.dropout,
        #         bidirectional=self.config.rnn.lstm.bidirectional,
        #         return_sequences=self.config.rnn.lstm.retseq,
        #     )
        #     if self.config.rnn.lstm.retseq:
        #         self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        #     else:
        #         self.max_pool = None
        #     clf_input_size = self.rnn.compute_output_shape(None,None)[-1]
        # elif self.config.rnn.name.lower() == 'attention':
        #     hidden_sizes = self.config.rnn.attention.hidden_sizes
        #     attn_in_channels = hidden_sizes[-1]
        #     if self.config.rnn.attention.bidirectional:
        #         attn_in_channels *= 2
        #     self.rnn = nn.Sequential(
        #         StackedLSTM(
        #             input_size=rnn_input_size,
        #             hidden_sizes=hidden_sizes,
        #             bias=self.config.rnn.attention.bias,
        #             dropout=self.config.rnn.attention.dropout,
        #             bidirectional=self.config.rnn.attention.bidirectional,
        #             return_sequences=True,
        #         ),
        #         SelfAttention(
        #             in_features=attn_in_channels,
        #             head_num=self.config.rnn.attention.head_num,
        #             dropout=self.config.rnn.attention.dropout,
        #             bias=self.config.rnn.attention.bias,
        #         )
        #     )
        #     self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        #     clf_input_size = self.rnn[-1].compute_output_shape(None,None)[-1]
        # else:
        #     raise NotImplementedError

        # if self.__DEBUG__:
        #     print(f"clf_input_size = {clf_input_size}")

        # # input of `self.clf` has shape: batch_size, channels
        # self.clf = nn.Linear(clf_input_size, self.n_classes)

        # # sigmoid for inference
        # self.sigmoid = nn.Sigmoid()  # for making inference

    def forward(self, input:Tensor) -> Tensor:
        """ NOT finished,
        """
        raise NotImplementedError

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params
