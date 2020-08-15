"""
validated CRNN structure models,
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
]


class VGGBlock(nn.Sequential):
    """
    building blocks of the CNN feature extractor `VGG6`
    """
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
                    kernel_initializer=self.config.kernel_initializer,
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


class ATI_CNN(nn.Module):
    """

    CRNN models proposed in the following refs.

    References:
    -----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    """
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
        print(f"configuration of ATI_CNN is as follows\n{dict_to_str(self.config)}")
        
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
