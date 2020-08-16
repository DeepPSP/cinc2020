"""
CRNN structure models,
for classifying ECG arrhythmias
"""
from copy import deepcopy
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real, Number

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

# from cfg import ModelCfg
from model_configs.ati_cnn import ATI_CNN_CONFIG
from model_configs.cpsc import CPSC_CONFIG
from models.utils.torch_utils import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext,
    compute_conv_output_shape,
)
from utils.misc import dict_to_str


__all__ = [
    # CRNN structure 1
    "ATI_CNN",
    "VGGBlock", "VGG6",
    "ResNetStanfordBlock", "ResNetStanford",
    "ResNetBasicBlock", "ResNetBottleNeck", "ResNet",
    # CRNN structure 2
    "CPSC",
    "CPSCMiniBlock", "CPSCBlock",
]


class VGGBlock(nn.Sequential):
    """
    building blocks of the CNN feature extractor `VGG6`
    """
    __DEBUG__ = False
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
                    kernel_size=self.config.pool_kernel,
                    stride=self.config.pool_stride,
                    channel_last=False,
                )
            num_layers += 1
        return output_shape


class VGG6(nn.Sequential):
    """
    CNN feature extractor of the CRNN models proposed in refs of `ATI_CNN`
    """
    __DEBUG__ = True
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
        # self.config = deepcopy(ATI_CNN_CONFIG.cnn.vgg6)
        self.config = ED(config)

        module_in_channels = in_channels
        for idx, (nc, nf) in enumerate(zip(self.config.num_convs, self.config.num_filters)):
            module_name = f"vgg_block_{idx+1}"
            self.add_module(
                name=module_name,
                module=VGGBlock(
                    num_convs=nc,
                    in_channels=module_in_channels,
                    out_channels=nf,
                    **(config.block),
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


class ResNetStanfordBlock(nn.Module):
    """
    building blocks of the CNN feature extractor `ResNetStanford`
    """
    __DEBUG__ = True
    def __init__(self, block_index:int, in_channels:int, num_filters:int, subsample_length:int, dilation:int=1, **config) -> NoReturn:
        """ finished, checked,

        the main stream uses `subsample_length` as stride to perform down-sampling,
        the short cut uses `subsample_length` as pool size to perform down-sampling,

        Parameters:
        -----------
        block_index: int,
            index of the block in the whole sequence of `ResNetStanford`
        in_channels: int,
            number of features (channels) of the input
        num_filters: int,
            number of filters for the convolutional layers
        subsample_length: int,
            subsample length,
            including pool size for short cut, and stride for the top convolutional layer
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer, dropout,
            and short cut patterns, etc.

        Issues:
        -------
        1. even kernel size would case mismatch of shapes of main stream and short cut
        """
        super().__init__()
        self.__block_index = block_index
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.config = ED(config)
        self.__num_convs = self.config.num_skip
        
        self.__increase_channels = (self.__out_channels > self.__in_channels)
        self.short_cut = self._make_short_cut_layer()
        
        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            if not (block_index == 0 and i == 0):
                self.main_stream.add_module(
                    f"ba_{self.__block_index}_{i}",
                    Bn_Activation(
                        num_features=self.__in_channels,
                        activation=self.config.activation,
                        kw_activation=self.config.kw_activation,
                        dropout=self.config.dropout if i > 0 else 0,
                    ),
                )
            self.main_stream.add_module(
                f"conv_{self.__block_index}_{i}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.config.filter_length,
                    stride = (self.__stride if i == 0 else 1),
                    bn=False,
                    activation=None,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                )
            )
            conv_in_channels = self.__out_channels

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        if self.__DEBUG__:
            print(f"forwarding in the {self.__block_index}-th `ResNetStanfordBlock`...")
            args = {k.split("__")[1]:v for k,v in self.__dict__.items() if isinstance(v, Number) and '__' in k}
            print(f"input arguments:\n{args}")
            print(f"input shape = {input.shape}")
        if self.short_cut:
            sc = self.short_cut(input)
        else:
            sc = input
        output = self.main_stream(input)
        if self.__DEBUG__:
            print(f"shape of short_cut output = {sc.shape}, shape of main stream output = {output.shape}")
        output = output +sc
        return output

    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """
        """
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == 'conv':
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    bn=False,
                    method=self.config.subsample_method,
                )
            if self.config.increase_channels_method.lower() == 'zero_padding':
                short_cut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        bn=False,
                        method=self.config.subsample_method,
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels),
                )
        else:
            short_cut = None
        return short_cut

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


class ResNetStanford(nn.Sequential):
    """
    the model proposed in ref. [1] and implemented in ref. [2]

    References:
    -----------
    [1] Hannun, Awni Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65.
    [2] https://github.com/awni/ecg
    """
    __DEBUG__ = True
    def __init__(self, in_channels:int, **config):
        """
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(config)

        if self.__DEBUG__:
            print(f"configuration of ResNetStanford is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.num_filters_start,
                kernel_size=self.config.filter_length,
                stride=1,
                bn=True,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
            )
        )

        module_in_channels = self.config.num_filters_start
        for idx, subsample_length in enumerate(self.config.subsample_lengths):
            num_filters = self.get_num_filters_at_index(idx, self.config.num_filters_start)
            self.add_module(
                f"resnet_block_{idx}",
                ResNetStanfordBlock(
                    block_index=idx,
                    in_channels=module_in_channels,
                    num_filters=num_filters,
                    subsample_length=subsample_length,
                    **self.config,
                )
            )
            module_in_channels = num_filters
            # if idx % self.config.increase_channels_at == 0 and idx > 0:
            #     module_in_channels *= 2

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        output = super().forward(input)
        return output

    def get_num_filters_at_index(self, index:int, num_start_filters:int) -> int:
        """

        Parameters:
        -----------
        index: int,
            index of a `ResNetStanfordBlock` in the sequence of such blocks in the whole network
        num_start_filters: int,
            number of filters of the first convolutional layer of the whole network

        Returns:
        --------
        num_filters: int,
            number of filters at the {index}-th `ResNetStanfordBlock`
        """
        num_filters = 2**int(index / self.config.increase_channels_at) * num_start_filters
        return num_filters

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
            the output shape of this Module, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
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
            print(f"__down_scale = {self.__down_scale}, __increase_channels = {self.__increase_channels}")
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
            print(f"configuration of ResNet is as follows\n{dict_to_str(self.config)}")
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


class ATI_CNN(nn.Module):
    """

    CRNN models proposed in the following refs.

    References:
    -----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    """
    __DEBUG__ = True
    def __init__(self, classes:list, input_len:int, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        input_len: int,
            sequence length (last dim.) of the input
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len
        self.config = deepcopy(ATI_CNN_CONFIG)
        self.config.update(config)
        if self.__DEBUG__:
            print(f"configuration of ATI_CNN is as follows\n{dict_to_str(self.config)}")
        
        cnn_choice = self.config.cnn.name.lower()
        if cnn_choice == "vgg6":
            self.cnn = VGG6(self.n_leads, **(self.config.cnn.vgg6))
            rnn_input_size = self.config.cnn.vgg6.num_filters[-1]
        elif cnn_choice == "resnet":
            self.cnn = ResNet(self.n_leads, **(self.config.cnn.resnet))
            rnn_input_size = 2**len(self.config.cnn.num_blocks) * self.config.cnn.init_num_filters
        cnn_output_shape = self.cnn.compute_output_shape(input_len, batch_size=None)
        self.cnn_output_len = cnn_output_shape[2]
        if self.__DEBUG__:
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
    building block of the SOTA model of CPSC2018 challenge
    """
    __DEBUG__ = True
    def __init__(self, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropout:Optional[float]=None, **kwargs) -> NoReturn:
        """

        Parameters:
        -----------
        filter_lengths: sequence of int,
            filter length (kernel size) of each convolutional layer
        subsample_lengths: sequence of int,
            subsample length (stride) of each convolutional layer
        dropout: float, optional,
            if positive, a `Dropout` layer will be introduced with this dropout probability
        kwargs: dict,
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
    __DEBUG__ = True
    def __init__(self, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropouts:Optional[float]=None, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        filter_lengths: sequence of int,
            filter length (kernel size) of each convolutional layer in each `CPSCMiniBlock`
        subsample_lengths: sequence of int,
            subsample length (stride) of each convolutional layer in each `CPSCMiniBlock`
        dropout: sequence of float, optional,
            dropout for each `CPSCMiniBlock`
        kwargs: dict,
        """
        super().__init__()
        for blk_idx, (blk_fl, blk_sl, blk_dp) in enumerate(zip(filter_lengths, subsample_lengths, dropouts)):
            self.add_module(
                f"cpsc_mini_block_{blk_idx+1}",
                CPSCMiniBlock(
                    filter_lengths=blk_fl,
                    subsample_lengths=blk_sl,
                    dropout=blk_dp,
                )
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
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class CPSC(nn.Sequential):
    """
    SOTA model of the CPSC2018 challenge
    """
    __DEBUG__ = True
    def __init__(self, classes:list, input_len:int, **config) -> NoReturn:
        """

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        input_len: int,
            sequence length (last dim.) of the input
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len
        self.config = deepcopy(CPSC_CONFIG)
        self.config.update(config)
        if self.__DEBUG__:
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
        output = self.cnn(input)
        output = self.rnn(output)
        return output
