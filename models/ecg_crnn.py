"""
validated CRNN structure models,
for classifying ECG arrhythmias
"""
from copy import deepcopy
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

# from cfg import ModelCfg
from model_configs import ECG_CRNN_CONFIG
from cfg import TrainCfg
# from model_configs.cpsc import CPSC_CONFIG
from models.utils.torch_utils import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM, BidirectionalLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext,
    compute_conv_output_shape,
)
from utils.misc import dict_to_str


__all__ = [
    "ECG_CRNN",
]


class VGGBlock(nn.Sequential):
    """
    building blocks of the CNN feature extractor `VGG6`
    """
    __DEBUG__ = False
    __name__ = "VGGBlock"

    def __init__(self, num_convs:int, in_channels:int, out_channels:int, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        num_convs: int,
            number of convolutional layers of this block
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
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

        # self.config = deepcopy(ECG_CRNN_CONFIG.cnn.vgg_block)
        # self.config.update(config)
        self.config = ED(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=self.config.filter_length,
                stride=self.config.subsample_length,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
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
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bn=self.config.batch_norm,
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.config.pool_size)
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


class VGG6(nn.Sequential):
    """
    CNN feature extractor of the CRNN models proposed in refs of `ATI_CNN`
    """
    __DEBUG__ = True
    __name__ = "VGG6"

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
        # self.config = deepcopy(ECG_CRNN_CONFIG.cnn.vgg6)
        self.config = ED(config)
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


class ResNetBasicBlock(nn.Module):
    """

    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`
    """
    __DEBUG__ = True
    __name__ = "ResNetBasicBlock"
    expansion = 1

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
        if dilation > 1:
            raise NotImplementedError(f"Dilation > 1 not supported in {self.__name__}")
        self.__num_convs = 2
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.config = ED(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        
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
                    kernel_size=self.config.filter_length,
                    stride=(self.__stride if i == 0 else 1),
                    bn=True,
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
                    bn=True,
                    method=self.config.subsample_method,
                )
            if self.config.increase_channels_method.lower() == 'zero_padding':
                bn = False if self.config.subsample_method.lower() != 'conv' else True
                short_cut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        bn=bn,
                        method=self.config.subsample_method,
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
    __DEBUG__ = False
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
        self.config = ED(config)
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

        # grouped resnet (basic) blocks,
        # number of channels are doubled at the first block of each group
        for group_idx, nb in enumerate(self.config.num_blocks):
            group_in_channels = (2**group_idx) * self.config.init_num_filters
            block_in_channels = group_in_channels
            block_num_filters = 2 * block_in_channels
            for block_idx in range(nb):
                block_subsample_length = self.config.subsample_length if block_idx == 0 else 1
                self.add_module(
                    f"block_{group_idx}_{block_idx}",
                    self.building_block(
                        in_channels=block_in_channels,
                        num_filters=block_num_filters,
                        subsample_length=block_subsample_length,
                        groups=1,
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
                    pad=(self.config.init_pool_size-1)//2,
                )
            else:
                output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class ECG_CRNN(nn.Module):
    """

    CRNN models proposed in the following refs.

    References:
    -----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    """
    __DEBUG__ = True
    __name__ = 'ECG_CRNN'

    def __init__(self, classes:list, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
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
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len or TrainCfg.input_len
        self.config = deepcopy(ECG_CRNN_CONFIG)
        self.config.update(config or {})
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
        
        cnn_choice = self.config.cnn.name.lower()
        if cnn_choice == "vgg6":
            self.cnn = VGG6(self.n_leads, **(self.config.cnn[cnn_choice]))
            rnn_input_size = self.config.cnn.vgg6.num_filters[-1]
        elif cnn_choice == "resnet":
            self.cnn = ResNet(self.n_leads, **(self.config.cnn[cnn_choice]))
            rnn_input_size = \
                2**len(self.config.cnn.resnet.num_blocks) * self.config.cnn.resnet.init_num_filters
        else:
            raise NotImplementedError
        # self.cnn_output_len = cnn_output_shape[2]
        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(self.input_len, batch_size=None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}")

        rnn_choice = self.config.rnn.name.lower()
        if rnn_choice == 'lstm':
            # self.rnn = StackedLSTM(
            #     input_size=rnn_input_size,
            #     hidden_sizes=self.config.rnn.hidden_sizes,
            #     bias=self.config.rnn.bias,
            #     dropout=self.config.rnn.dropout,
            #     bidirectional=self.config.rnn.bidirectional,
            #     return_sequences=self.config.rnn.retseq,
            # )
            self.rnn = BidirectionalLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.hidden_sizes[-1],
                num_layers=len(self.config.rnn.hidden_sizes)
                bias=self.config.rnn.bias,
                dropout=self.config.rnn.dropout,
            )
            clf_input_size = self.rnn.compute_output_shape(None,None)[-1]
            # if self.config.rnn.bidirectional:
            #     clf_input_size = 2*self.config.rnn.hidden_sizes[-1]
            # else:
            #     clf_input_size = self.config.rnn.hidden_sizes[-1]
            # if self.config.rnn.retseq:
            #     clf_input_size *= self.cnn_output_len
        elif rnn_choice == 'attention':
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.__DEBUG__:
            print(f"clf_input_size = {clf_input_size}")

        if self.config.rnn.retseq:
            self.max_pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        self.clf = nn.Linear(clf_input_size, self.n_classes)
        self.sigmoid = nn.Sigmoid()  # for making inference

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
            # (seq_len, batch, channels) -> (batch, channels, seq_len)
            x = x.permute(1,2,0)
            x = self.max_pool(x)  # (batch, channels, 1)
            x = torch.flatten(x, 1)  # (batch, channels)
        pred = self.clf(x)
        if not self.training:
            pred = self.sigmoid(pred)
        return pred

    def inference(self, input:Tensor, class_names:bool=False, bin_pred_thr:float=0.5) -> Union[Tensor, pd.DataFrame]:
        """
        """
        pred = self.forward(input)
        if self.training:
            pred = self.sigmoid(pred)
        if class_names:
            pred = pred.cpu().detach().numpy()
            pred = pd.DataFrame(pred)
            pred.columns = self.classes
            pred['bin_pred'] = pred.apply(
                lambda row: np.array(self.classes)[np.where(row.values>=bin_pred_thr)[0]],
                axis=1
            )
        return pred
