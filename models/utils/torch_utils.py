"""
basic building blocks, for 1d signal (time series)
"""
import sys
from math import floor
from itertools import repeat
from typing import Union, Sequence, Tuple, Optional, NoReturn

from packaging import version
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from easydict import EasyDict as ED

torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "Mish", "Swish",
    "Initializers", "Activations",
    "Bn_Activation", "Conv_Bn_Activation",
    "DownSample",
    "BidirectionalLSTM", "StackedLSTM",
    "AML_Attention", "AML_GatedAttention",
    "AttentionWithContext",
    "ZeroPadding",
    "WeightedBCELoss", "BCEWithLogitsWithClassWeightLoss",
    "compute_output_shape",
    "compute_conv_output_shape",
    "compute_maxpool_output_shape", "compute_avgpool_output_shape",
    "default_collate_fn",
]


if version.parse(torch.__version__) >= version.parse('1.5.0'):
    def _true_divide(dividend, divisor):
        return torch.true_divide(dividend, divisor)
else:
    def _true_divide(dividend, divisor):
        return dividend / divisor


# ---------------------------------------------
# activations
class Mish(torch.nn.Module):
    __name__ = "Mish"
    """ The Mish activation """
    def __init__(self, inplace:bool=False):
        """
        """
        super().__init__()
        self.inplace = inplace

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        if self.inplace:
            input = input * (torch.tanh(F.softplus(input)))
            output = input
        else:
            output = input * (torch.tanh(F.softplus(input)))
        return output


class Swish(torch.nn.Module):
    __name__ = "Swish"
    """ The Swish activation """
    def __init__(self, inplace:bool=False):
        """
        """
        super().__init__()
        self.inplace = inplace

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        if self.inplace:
            input = input * F.sigmoid(input)
            output = input
        else:
            output = input * F.sigmoid(input)
        return output


# ---------------------------------------------
# initializers
Initializers = ED()
Initializers.he_normal = nn.init.kaiming_normal_
Initializers.kaiming_normal = nn.init.kaiming_normal_
Initializers.he_uniform = nn.init.kaiming_uniform_
Initializers.kaiming_uniform = nn.init.kaiming_uniform_
Initializers.xavier_normal = nn.init.xavier_normal_
Initializers.glorot_normal = nn.init.xavier_normal_
Initializers.xavier_uniform = nn.init.xavier_uniform_
Initializers.glorot_uniform = nn.init.xavier_uniform_
Initializers.normal = nn.init.normal_
Initializers.uniform = nn.init.uniform_
Initializers.orthogonal = nn.init.orthogonal_
Initializers.zeros = nn.init.zeros_
Initializers.ones = nn.init.ones_
Initializers.constant = nn.init.constant_


# ---------------------------------------------
# activations
Activations = ED()
Activations.mish = Mish
Activations.swish = Swish
Activations.relu = nn.ReLU
Activations.leaky = nn.LeakyReLU
Activations.leaky_relu = Activations.leaky
# Activations.linear = None


# ---------------------------------------------
# basic building blocks of CNN
class Bn_Activation(nn.Sequential):
    """ finished, checked,

    batch normalization --> activation
    """
    __name__ = "Bn_Activation"

    def __init__(self, num_features:int, activation:Union[str,nn.Module], kw_activation:Optional[dict]=None, dropout:float=0.0) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        num_features: int,
            number of features (channels) of the input (and output)
        activation: str or Module,
            name of the activation or an activation `Module`
        kw_activation: dict, optional,
            key word arguments for `activation`
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer at the end of the block
        """
        super().__init__()
        self.__num_features = num_features
        self.__kw_activation = kw_activation or {}
        self.__dropout = dropout
        if callable(activation):
            act_layer = activation
            act_name = f"activation_{type(act_layer).__name__}"
        elif isinstance(activation, str) and activation.lower() in Activations.keys():
            act_layer = Activations[activation.lower()](**self.__kw_activation)
            act_name = f"activation_{activation.lower()}"

        self.add_module(
            "batch_norm",
            nn.BatchNorm1d(num_features),
        )
        self.add_module(
            act_name,
            act_layer,
        )
        if self.__dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.__dropout),
            )
    
    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`
        
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
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
            the output shape of this `Bn_Activation` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__num_features, seq_len)
        return output_shape


class Conv_Bn_Activation(nn.Sequential):
    """ finished, checked,

    1d convolution --> batch normalization (optional) -- > activation (optional),
    with "same" padding
    """
    __name__ = "Conv_Bn_Activation"

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:Optional[int]=None, dilation:int=1, groups:int=1, batch_norm:Union[bool,nn.Module]=True, activation:Optional[Union[str,nn.Module]]=None, kernel_initializer:Optional[Union[str,callable]]=None, bias:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size (length) of the convolving kernel
        stride: int,
            stride (subsample length) of the convolution
        padding: int, optional,
            zero-padding added to both sides of the input
        dilation: int, default 1,
            spacing between the kernel points
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or Module, default True,
            batch normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, optional,
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear",
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        """
        super().__init__()
        if padding is None:
            self.__padding = (kernel_size - 1) // 2  # 'same' padding when stride = 1
        elif isinstance(padding, int):
            self.__padding = padding
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation
        self.__groups = groups
        self.__bias = bias
        self.__kw_activation = kwargs.get("kw_activation", {})
        self.__kw_initializer = kwargs.get("kw_initializer", {})

        conv_layer = nn.Conv1d(
            self.__in_channels, self.__out_channels,
            self.__kernel_size, self.__stride, self.__padding, self.__dilation, self.__groups,
            bias=self.__bias,
        )

        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv_layer.weight)
            elif isinstance(kernel_initializer, str) and kernel_initializer.lower() in Initializers.keys():
                Initializers[kernel_initializer.lower()](conv_layer.weight, **self.__kw_initializer)
            else:  # TODO: add more activations
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        self.add_module("conv1d", conv_layer)

        if batch_norm:
            bn_layer = nn.BatchNorm1d(out_channels) if isinstance(batch_norm, bool) else batch_norm(out_channels)
            self.add_module("batch_norm", bn_layer)

        if isinstance(activation, str):
            activation = activation.lower()

        if not activation:
            act_layer = None
            act_name = None
        elif callable(activation):
            act_layer = activation
            act_name = f"activation_{type(act_layer).__name__}"
        elif isinstance(activation, str) and activation.lower() in Activations.keys():
            act_layer = Activations[activation.lower()](**self.__kw_activation)
            act_name = f"activation_{activation.lower()}"
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")
            act_layer = None
            act_name = None

        if act_layer:
            self.add_module(act_name, act_layer)

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
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
            the output shape of this `Conv_Bn_Activation` layer, given `seq_len` and `batch_size`
        """
        input_shape = [batch_size, self.__in_channels, seq_len]
        output_shape = compute_conv_output_shape(
            input_shape=input_shape,
            num_filters=self.__out_channels,
            kernel_size=self.__kernel_size,
            stride=self.__stride,
            dilation=self.__dilation,
            padding=self.__padding,
            channel_last=False,
        )
        return output_shape


class DownSample(nn.Sequential):
    """

    NOTE: this down sampling module allows changement of number of channels,
    via additional convolution, with some abuse of terminology

    the 'conv' mode is not simply down 'sampling' if `group` != `in_channels`
    """
    __name__ = "DownSample"
    __MODES__ = ['max', 'avg', 'conv', 'nearest', 'area', 'linear',]

    def __init__(self, down_scale:int, in_channels:int, out_channels:Optional[int]=None, groups:Optional[int]=None, padding:int=0, batch_norm:Union[bool,nn.Module]=False, mode:str='max') -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        down_scale: int,
            scale of down sampling
        in_channels: int,
            number of channels of the input
        out_channels: int, optional,
            number of channels of the output
        groups: int, optional,
            connection pattern (of channels) of the inputs and outputs
        padding: int, default 0,
            zero-padding added to both sides of the input
        batch_norm: bool or Module, default False,
            batch normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        mode: str, default 'max',
        """
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels or in_channels
        self.__groups = groups or self.__in_channels
        self.__padding = padding

        if self.__mode == 'max':
            if self.__in_channels == self.__out_channels:
                down_layer = nn.MaxPool1d(kernel_size=self.__down_scale, padding=self.__padding)
            else:
                down_layer = nn.Sequential((
                    nn.MaxPool1d(kernel_size=self.__down_scale, padding=self.__padding),
                    nn.Conv1d(
                        self.__in_channels, self.__out_channels, 
                        kernel_size=1, groups=self.__groups, bias=False
                    ),
                ))
        elif self.__mode == 'avg':
            if self.__in_channels == self.__out_channels:
                down_layer = nn.AvgPool1d(kernel_size=self.__down_scale, padding=self.__padding)
            else:
                down_layer = nn.Sequential(
                    (
                        nn.AvgPool1d(kernel_size=self.__down_scale, padding=self.__padding),
                        nn.Conv1d(
                            self.__in_channels,self.__out_channels,
                            kernel_size=1,groups=self.__groups, bias=False,
                        ),
                    )
                )
        elif self.__mode == 'conv':
            down_layer = nn.Conv1d(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                kernel_size=1,
                groups=self.__groups,
                bias=False,
                stride=self.__down_scale,
            )
        else:
            down_layer = None
        if down_layer:
            self.add_module(
                "down_sample",
                down_layer,
            )

        if batch_norm:
            bn_layer = nn.BatchNorm1d(self.__out_channels) if isinstance(batch_norm, bool) \
                else batch_norm(self.__out_channels)
            self.add_module(
                "batch_normalization",
                bn_layer,
            )

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        if self.__mode in ['max', 'avg', 'conv',]:
            output = super().forward(input)
        else:
            # align_corners = False if mode in ['nearest', 'area'] else True
            output = F.interpolate(
                input=input,
                scale_factor=1/self.__down_scale,
                mode=mode,
                # align_corners=align_corners,
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
            the output shape of this `Bn_Activation` layer, given `seq_len` and `batch_size`
        """
        if self.__mode == 'conv':
            out_seq_len = compute_conv_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == 'max':
            out_seq_len = compute_maxpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__down_scale, stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode in ['avg', 'nearest', 'area', 'linear',]:
            out_seq_len = compute_avgpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__down_scale, stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        output_shape = (batch_size, self.__out_channels, out_seq_len)
        return output_shape


class BidirectionalLSTM(nn.Module):
    """
    from crnn_torch of references.ati_cnn
    """
    __name__ = "BidirectionalLSTM"

    def __init__(self, input_size:int, hidden_size:int, num_layers:int=1, bias:bool=True, dropout:float=0.0, return_sequences:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        input_size: int,
            the number of features in the input
        hidden_size: int,
            the number of features in the hidden state
        num_layers: int, default 1,
            number of lstm layers
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
        return_sequences: bool, default True,
            if True, returns the last output in the output sequence,
            otherwise the full sequence.
        """
        super().__init__()
        self.__output_size = 2 * hidden_size
        self.return_sequence = return_sequences

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bias=bias,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        """
        output, _ = self.lstm(input)  #  seq_len, batch_size, 2 * hidden_size
        if not self.return_sequence:
            output = output[-1,...]
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
            the output shape of this `BidirectionalLSTM` layer, given `seq_len` and `batch_size`
        """
        output_shape = (seq_len, batch_size, self.__output_size)
        return output_shape


class StackedLSTM(nn.Sequential):
    """ finished, checked (no bugs, but correctness is left further to check),

    stacked LSTM, which allows different hidden sizes for each LSTM layer

    NOTE:
    -----
    1. `batch_first` is fixed `False`
    2. currently, how to correctly pass the argument `hx` between LSTM layers is not known to me, hence should be careful (and not recommended, use `nn.LSTM` and set `num_layers` instead) to use
    """
    __DEBUG__ = False
    __name__ = "StackedLSTM"

    def __init__(self, input_size:int, hidden_sizes:Sequence[int], bias:Union[Sequence[bool], bool]=True, dropout:float=0.0, bidirectional:bool=True, return_sequences:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        input_size: int,
            the number of features in the input
        hidden_sizes: sequence of int,
            the number of features in the hidden state of each LSTM layer
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
        bidirectional: bool, default True,
        return_sequences: bool, default True,
        """
        super().__init__()
        self.__hidden_sizes = hidden_sizes
        self.num_lstm_layers = len(hidden_sizes)
        l_bias = bias if isinstance(bias, Sequence) else list(repeat(bias, self.num_lstm_layers))
        self.__dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = False
        self.return_sequences = return_sequences

        layer_name_prefix = "bidirectional_lstm" if bidirectional else "lstm"
        for idx, (hs, b) in enumerate(zip(hidden_sizes, l_bias)):
            if idx == 0:
                _input_size = input_size
            else:
                _input_size = hidden_sizes[idx-1]
                if self.bidirectional:
                    _input_size = 2*_input_size
            self.add_module(
                name=f"{layer_name_prefix}_{idx+1}",
                module=nn.LSTM(
                    input_size=_input_size,
                    hidden_size=hs,
                    num_layers=1,
                    bias=b,
                    batch_first=self.batch_first,
                    bidirectional=self.bidirectional,
                )
            )
            if self.__dropout > 0 and idx < self.num_lstm_layers-1:
                self.add_module(
                    name=f"dropout_{idx+1}",
                    module=nn.Dropout(self.__dropout),
                )
    
    def forward(self, input:Union[Tensor, PackedSequence], hx:Optional[Tuple[Tensor, Tensor]]=None) -> Union[Tensor, Tuple[Union[Tensor, PackedSequence], Tuple[Tensor, Tensor]]]:
        """
        keep up with `nn.LSTM.forward`, parameters ref. `nn.LSTM.forward`

        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        hx: 2-tuple of Tensor, optional,
        """
        n_layers = 0
        output, _hx = input, hx
        div = 2 if self.__dropout > 0 else 1
        for module in self:
            n_lstm, res = divmod(n_layers, div)
            if res == 1:
                output = module(output)
                # print(f"module = {type(module).__name__}")
            else:
                # print(f"n_layers = {n_layers}, input shape = {output.shape}")
                if n_lstm > 0:
                    _hx = None
                output, _hx = module(output, _hx)
                # print(f"module = {type(module).__name__}")
                # print(f"n_layers = {n_layers}, input shape = {output.shape}")
            n_layers += 1
        if self.return_sequences:
            final_output = output  # seq_len, batch_size, n_direction*hidden_size
        else:
            final_output = output[-1,...]  # batch_size, n_direction*hidden_size
        return final_output

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
            the output shape of this `StackedLSTM` layer, given `seq_len` and `batch_size`
        """
        output_size = self.__hidden_sizes[-1]
        if self.bidirectional:
            output_size *= 2
        if self.return_sequences:
            output_shape = (seq_len, batch_size, output_size)
        else:
            output_shape = (batch_size, output_size)
        return output_shape


# ---------------------------------------------
# attention mechanisms, from various sources
class AML_Attention(nn.Module):
    """ NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    References:
    -----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L6
    """
    __name__ = "AML_Attention"

    def __init__(self, L:int, D:int, K:int):
        """ NOT checked,
        """
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, input):
        """
        """
        A = self.attention(input)  # NxK
        return A

class AML_GatedAttention(nn.Module):
    """ NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    TODO: compare with `nn.MultiheadAttention`

    References:
    -----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L72
    """
    __name__ = "AML_GatedAttention"

    def __init__(self, L:int, D:int, K:int):
        """ NOT checked,
        """
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, input):
        """
        """
        A_V = self.attention_V(input)  # NxD
        A_U = self.attention_U(input)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        return A


class AttentionWithContext(nn.Module):
    """ finished, checked (might have bugs),

    from 0236 of CPSC2018 challenge
    """
    __DEBUG__ = False
    __name__ = "AttentionWithContext"

    def __init__(self, in_channels:int, bias:bool=True, initializer:str='glorot_uniform'):
        """ finished, checked (might have bugs),

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input signal
        bias: bool, default True,
            if True, adds a learnable bias to the output
        initializer: str, default 'glorot_uniform',
            weight initializer
        """
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias

        self.W = Parameter(torch.Tensor(in_channels, in_channels))
        if self.__DEBUG__:
            print(f"AttentionWithContext W.shape = {self.W.shape}")
        self.init(self.W)

        if self.bias:
            self.b = Parameter(torch.Tensor(in_channels))
            if self.__DEBUG__:
                print(f"AttentionWithContext b.shape = {self.b.shape}")
            # Initializers['zeros'](self.b)
            self.u = Parameter(torch.Tensor(in_channels))
            if self.__DEBUG__:
                print(f"AttentionWithContext u.shape = {self.u.shape}")
            # self.init(self.u)
        else:
            self.register_parameter('b', None)
            self.register_parameter('u', None)

    def compute_mask(self, input:Tensor, input_mask:Optional[Tensor]=None):
        """

        Parameters:
        -----------
        to write
        """
        return None

    def forward(self, input:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        mask: Tensor, optional,
        """
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: input.shape = {input.shape}, W.shape = {self.W.shape}")
        # (seq_len, batch_size, n_channels) -> (batch_size, seq_len, n_channels)
        _input = input.permute(1,0,2)

        # linear + activation
        # (batch_size, seq_len, n_channels) x (n_channels, n_channels)
        # -> (batch_size, seq_len, n_channels)
        uit = torch.tensordot(_input, self.W, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: uit.shape = {uit.shape}")
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)

        # scores (weights)
        # (batch_size, seq_len, n_channels) x (n_channels,)
        # -> (batch_size, seq_len)
        ait = torch.tensordot(uit, self.u, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: ait.shape = {ait.shape}")
        
        # softmax along seq_len
        # (batch_size, seq_len)
        a = torch.exp(ait)
        if mask is not None:
            a = a * mask
        a = _true_divide(a, torch.sum(a, dim=-1, keepdim=True) + torch.finfo(torch.float).eps)
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: a.shape = {a.shape}")

        # weighted -> sum
        # (batch_size, seq_len, n_channels) x (batch_size, seq_len, 1)
        # -> (batch_size, seq_len, n_channels)
        weighted_input = _input * a[..., np.newaxis]
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: weighted_input.shape = {weighted_input.shape}")
        output = torch.sum(weighted_input, dim=-1)
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: output.shape = {output.shape}")
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
            the output shape of this `ZeroPadding` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape


class Attention(nn.Module):
    """
    simplified version of `AttentionWithContext`
    """
    __DEBUG__ = True
    __name__ = "Attention"

    def __init__(self, in_channels:int, bias:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
        bias: bool, default True,
        """
        super().__init__()
        self.__in_channels = in_channels
        self.linear = nn.Sequential(
            nn.Linear(self.__in_channels, self.__in_channels, bias=bias),
            nn.Tanh(),
        )
        self.att_v = Parameter(torch.Tensor(self.__in_channels))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        """
        # input: (seq_len, batch_size, n_channels) -> (batch_size, seq_len, n_channels)
        _input = input.permute(1,0,2)
        proj = self.linear(_input)
        # (batch_size, seq_len)
        score = torch.tensordot(proj, self.att_v, dims=1)  # equiv. to torch.matmul
        score = self.softmax(score)
        # (batch_size, seq_len, channels) -> (batch_size, channels)
        output = (_input * score[..., np.newaxis]).sum(dim=1)
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
            the output shape of this `ZeroPadding` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__in_channels)
        return output_shape


class ZeroPadding(nn.Module):
    """
    zero padding for increasing channels,
    degenerates to `identity` if in and out channels are equal
    """
    __name__ = "ZeroPadding"
    __LOC__ = ["head", "tail",]

    def __init__(self, in_channels:int, out_channels:int, loc:str="head") -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels for the output
        loc: str, default "top", case insensitive,
            padding to the head or the tail channel
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__increase_channels = out_channels - in_channels
        assert self.__increase_channels >= 0
        self.__loc = loc.lower()
        assert self.__loc in self.__LOC__

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        batch_size, _, seq_len = input.shape
        if self.__increase_channels > 0:
            output = torch.zeros((batch_size, self.__increase_channels, seq_len))
            if self.__loc == "head":
                output = torch.cat((output, input), dim=1)
            elif self.__loc == "tail":
                output = torch.cat((input, output), dim=1)
        else:
            output = input
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
            the output shape of this `ZeroPadding` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape



# utils for computing output shape
def compute_output_shape(layer_type:str, input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, output_padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, checked,

    compute the output shape of a (transpose) convolution/maxpool/avgpool layer
    
    Parameters:
    -----------
    layer_type: str,
        type (conv, maxpool, avgpool, etc.) of the layer
    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    out_padding: int, or sequence of int, default 0,
        additional size added to one side of the output shape,
        used only for transpose convolution
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor

    References:
    -----------
    [1] https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    __TYPES__ = [
        'conv', 'convolution',
        'deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',
        'maxpool', 'maxpooling',
        'avgpool', 'avgpooling', 'averagepool', 'averagepooling',
    ]
    lt = "".join(layer_type.lower().split("_"))
    assert lt in __TYPES__
    if lt in ['conv', 'convolution',]:
        minus_term = lambda d, k: d * (k - 1) + 1
        out_channels = num_filters
    elif lt in ['maxpool', 'maxpooling',]:
        minus_term = lambda d, k: d * (k - 1) + 1
        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in ['avgpool', 'avgpooling', 'averagepool', 'averagepooling',]:
        minus_term = lambda d, k: k
        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in ['deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',]:
        out_channels = num_filters
    dim = len(input_shape) - 2
    assert dim > 0, "input_shape should be a sequence of length at least 3, to be a valid (with batch and channel) shape of a non-degenerate Tensor"

    none_dim_msg = "only batch and channel dimension can be `None`"
    if channel_last:
        assert all([n is not None for n in input_shape[1:-1]]), none_dim_msg
    else:
        assert all([n is not None for n in input_shape[2:]]), none_dim_msg

    if isinstance(kernel_size, int):
        _kernel_size = list(repeat(kernel_size, dim))
    elif len(kernel_size) == dim:
        _kernel_size = kernel_size
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(kernel_size)} dimensions, both not including the channel dimension")
    
    if isinstance(stride, int):
        _stride = list(repeat(stride, dim))
    elif len(stride) == dim:
        _stride = stride
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(stride)} dimensions, both not including the channel dimension")

    if isinstance(padding, int):
        _padding = list(repeat(padding, dim))
    elif len(padding) == dim:
        _padding = padding
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(padding)} dimensions, both not including the channel dimension")

    if isinstance(output_padding, int):
        _output_padding = list(repeat(output_padding, dim))
    elif len(output_padding) == dim:
        _output_padding = output_padding
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(output_padding)} dimensions, both not including the channel dimension")

    if isinstance(dilation, int):
        _dilation = list(repeat(dilation, dim))
    elif len(dilation) == dim:
        _dilation = dilation
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(dilation)} dimensions, both not including the channel dimension")
    
    if channel_last:
        _input_shape = input_shape[1:-1]
    else:
        _input_shape = input_shape[2:]
    
    if lt in ['deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',]:
        output_shape = [
            (i-1) * s - 2 * p + d * (k-1) + o + 1 \
                for i, p, o, d, k, s in \
                    zip(_input_shape, _padding, _output_padding, _dilation, _kernel_size, _stride)
        ]    
    else:
        output_shape = [
            floor( ( ( i + 2*p - minus_term(d, k) ) / s ) + 1 ) \
                for i, p, d, k, s in \
                    zip(_input_shape, _padding, _dilation, _kernel_size, _stride)
        ]
    if channel_last:
        output_shape = tuple([input_shape[0]] + output_shape + [out_channels])
    else:
        output_shape = tuple([input_shape[0], out_channels] + output_shape)

    return output_shape


def compute_conv_output_shape(input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

    compute the output shape of a convolution/maxpool/avgpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'conv',
        input_shape, num_filters, kernel_size, stride, padding, 0, dilation,
        channel_last,
    )
    return output_shape


def compute_maxpool_output_shape(input_shape:Sequence[Union[int, type(None)]], kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

    compute the output shape of a maxpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'maxpool',
        input_shape, 1, kernel_size, stride, padding, 0, dilation,
        channel_last,
    )
    return output_shape


def compute_avgpool_output_shape(input_shape:Sequence[Union[int, type(None)]], kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

    compute the output shape of a avgpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'avgpool',
        input_shape, 1, kernel_size, stride, padding, 0, 1,
        channel_last,
    )
    return output_shape


def compute_deconv_output_shape(input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, output_padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, checked,

    compute the output shape of a transpose convolution layer
    
    Parameters:
    -----------
    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    out_padding: int, or sequence of int, default 0,
        additional size added to one side of the output shape,
        used only for transpose convolution
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'deconv',
        input_shape, num_filters, kernel_size, stride, padding, output_padding, dilation,
        channel_last,
    )
    return output_shape
    

# custom losses
def weighted_binary_cross_entropy(sigmoid_x:Tensor, targets:Tensor, pos_weight:Tensor, weight:Optional[Tensor]=None, size_average:bool=True, reduce:bool=True) -> Tensor:
    """ NOT checked,

    Parameters:
    -----------
    sigmoid_x: Tensor,
        predicted probability of size [N,C], N sample and C Class.
        Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
    targets: Tensor,
        true value, one-hot-like vector of size [N,C]
    pos_weight: Tensor,
        Weight for postive sample
    weight: Tensor, optional,
    size_average: bool, default True,
    reduce: bool, default True,

    Reference (original source):
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(nn.Module):
    """ NOT checked,

    Reference (original source):
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    __name__ = "WeightedBCELoss"

    def __init__(self, pos_weight:Tensor=1, weight:Optional[Tensor]=None, PosWeightIsDynamic:bool=False, WeightIsDynamic:bool=False, size_average:bool=True, reduce:bool=True) -> NoReturn:
        """ Not checked,

        Parameters:
        -----------
        pos_weight: Tensor, default 1,
            Weight for postive samples. Size [1,C]
        weight: Tensor, optional,
            Weight for Each class. Size [1,C]
        PosWeightIsDynamic: bool, default False,
            If True, the pos_weight is computed on each batch.
            If `pos_weight` is None, then it remains None.
        WeightIsDynamic: bool, default False,
            If True, the weight is computed on each batch.
            If `weight` is None, then it remains None.
        size_average: bool, default True,
        reduce: bool, default True,
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """
        """
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts) / (positive_counts + 1e-5)

        return weighted_binary_cross_entropy(input, target,
                                             pos_weight=self.pos_weight,
                                             weight=self.weight,
                                             size_average=self.size_average,
                                             reduce=self.reduce)


class BCEWithLogitsWithClassWeightLoss(nn.BCEWithLogitsLoss):
    """
    """
    __name__ = "BCEWithLogitsWithClassWeightsLoss"

    def __init__(self, class_weight:Tensor) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        class_weight: Tensor,
            class weight, of shape (1, n_classes)
        """
        super().__init__(reduction='none')
        self.class_weight = class_weight

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """
        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * self.class_weight)
        return loss




def default_collate_fn(batch:Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Tensor, Tensor]:
    """ finished, checked,

    collate functions for model training

    the data generator (`Dataset`) should generate (`__getitem__`) 2-tuples `signals, labels`

    Parameters:
    -----------
    batch: sequence,
        sequence of 2-tuples,
        in which the first element is the signal, the second is the label
    
    Returns:
    --------
    values: Tensor,
        the concatenated values as input for training
    labels: Tensor,
        the concatenated labels as ground truth for training
    """
    values = [[item[0]] for item in batch]
    labels = [[item[1]] for item in batch]
    values = np.concatenate(values, axis=0).astype(np.float64)
    values = torch.from_numpy(values)
    labels = np.concatenate(labels, axis=0).astype(np.float64)
    labels = torch.from_numpy(labels)
    return values, labels
