"""
"""
import sys
from math import floor
from typing import Union, Sequence, Tuple, Optional, NoReturn

from packaging import version
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from easydict import EasyDict as ED


__all__ = [
    "Mish", "Swish",
    "Initializers", "Activations",
    "Conv_Bn_Activation",
    "BidirectionalLSTM", "StackedLSTM",
    "AML_Attention", "AML_GatedAttention",
    "AttentionWithContext",
    "MultiheadAttention",
    "compute_conv_output_shape",
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
    """ The Mish activation """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        """
        """
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Swish(torch.nn.Module):
    """ The Swish activation """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        """
        """
        x = x * F.sigmoid(x)
        return x


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
class Conv_Bn_Activation(nn.Sequential):
    """ finished, checked

    1d convolution --> batch normalization (optional) -- > activation (optional)
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, bn:Union[bool,nn.Module]=True, activation:Optional[Union[str,nn.Module]]=None, kernel_initializer:Optional[Union[str,callable]]=None, bias:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size of the convolving kernel
        stride: int,
            stride of the convolution
        bn: bool or Module,
            batch normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, optional,
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear",
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of Initializers
        bias: bool, default True,
            If True, adds a learnable bias to the output
        """
        super().__init__()
        padding = (kernel_size - 1) // 2  # 'same' padding when stride = 1
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__bias = bias
        self.__kw_activation = kwargs.get("kw_activation", {})
        self.__kw_initializer = kwargs.get("kw_initializer", {})

        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv_layer.weight)
            elif isinstance(kernel_initializer, str) and kernel_initializer.lower() in Initializers.keys():
                Initializers[kernel_initializer.lower()](conv_layer.weight, **self.__kw_initializer)
            else:  # TODO: add more activations
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        self.add_module("conv1d", conv_layer)

        if bn:
            bn_layer = nn.BatchNorm1d(out_channels) if isinstance(bn, bool) else bn(out_channels)
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
        just use the forward function of `nn.Sequential`
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
            pad=self.__padding,
            channel_last=False,
        )
        return output_shape


class BidirectionalLSTM(nn.Module):
    """
    from crnn_torch of references.ati_cnn
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int) -> NoReturn:
        """

        Parameters:
        -----------
        input_size: int,
            the number of features in the input
        hidden_size: int,
            the number of features in the hidden state
        output_size: int,
            the number of features in the ouput
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        recurrent, _ = self.lstm(input)
        T, b, h = recurrent.size()  # seq_len, batch_size, hidden_size
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class StackedLSTM(nn.Sequential):
    """ finished, checked (no bugs, but correctness is left further to check),

    stacked LSTM, which allows different hidden sizes for each LSTM layer

    NOTE that `batch_first` is fixed `False`
    NOTE: currently, how to correctly pass the argument `hx` between LSTM layers is not known to me, hence should be careful (and not recommended, use `nn.LSTM` and set `num_layers` instead) to use
    """
    def __init__(self, input_size:int, hidden_sizes:Sequence[int], bias:Union[Sequence[bool], bool]=True, dropout:float=0.0, bidirectional:bool=True, return_sequences:bool=True) -> NoReturn:
        """

        Parameters: (to write)
        -----------
        input_size: int,
            the number of features in the input
        hidden_sizes: sequence of int,
            the number of features in the hidden state of each LSTM layer
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to this value
        bidirectional: bool, default True,
        return_sequences: bool, default True
        """
        super().__init__()
        
        self.num_lstm_layers = len(hidden_sizes)
        l_bias = bias if isinstance(bias, Sequence) else [bias for _ in range(self.num_lstm_layers)]
        self._dropout = dropout
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
            if self._dropout > 0 and idx < self.num_lstm_layers-1:
                self.add_module(
                    name=f"dropout_{idx+1}",
                    module=nn.Dropout(self._dropout),
                )
    
    def forward(self, input:Union[Tensor, PackedSequence], hx:Optional[Tuple[Tensor, Tensor]]=None) -> Union[Tensor, Tuple[Union[Tensor, PackedSequence], Tuple[Tensor, Tensor]]]:
        """
        keep up with `nn.LSTM.forward`
        """
        n_layers = 0
        _input, _hx = input, hx
        div = 2 if self._dropout > 0 else 1
        for module in self:
            n_lstm, res = divmod(n_layers, div)
            if res == 1:
                _input = module(_input)
                print(f"module = {type(module).__name__}")
            else:
                # print(f"n_layers = {n_layers}, input shape = {_input.shape}")
                if n_lstm > 0:
                    _hx = None
                _input, _hx = module(_input, _hx)
                print(f"module = {type(module).__name__}")
                # print(f"n_layers = {n_layers}, input shape = {_input.shape}")
            n_layers += 1
        if self.return_sequences:
            final_output = _input, _hx
        else:
            final_output = _hx[0]
        return final_output


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
    def __init__(self, L:int, D:int, K:int):
        """
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
    def __init__(self, L:int, D:int, K:int):
        """
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
    """ finished, checked,

    from CPSC0236
    """
    def __init__(self, in_channels:int, out_channels:int, bias:bool=True, initializer:str='glorot_uniform'):
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias

        self.W = Parameter(torch.Tensor(out_channels, out_channels))
        print(f"AttentionWithContext W.shape = {self.W.shape}")
        self.init(self.W)

        if self.bias:
            self.b = Parameter(torch.Tensor(out_channels))
            print(f"AttentionWithContext b.shape = {self.b.shape}")
            # Initializers['zeros'](self.b)
            self.u = Parameter(torch.Tensor(out_channels))
            print(f"AttentionWithContext u.shape = {self.u.shape}")
            # self.init(self.u)
        else:
            self.register_parameter('b', None)
            self.register_parameter('u', None)

    def compute_mask(self, input:Tensor, input_mask:Optional[Tensor]=None):
        """
        """
        return None

    def forward(self, input:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        """
        print(f"AttentionWithContext forward: input.shape = {input.shape}, W.shape = {self.W.shape}")
        uit = torch.tensordot(input, self.W, dims=1)
        print(f"AttentionWithContext forward: uit.shape = {uit.shape}")
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)
        ait = torch.tensordot(uit, self.u, dims=1)
        print(f"AttentionWithContext forward: ait.shape = {ait.shape}")
        a = torch.exp(ait)
        if mask is not None:
            a = a * mask
        a = _true_divide(a, torch.sum(a, dim=1, keepdim=True) + torch.finfo(torch.float).eps)
        print(f"AttentionWithContext forward: a.shape = {a.shape}")
        weighted_input = input * a[...,np.newaxis]
        print(f"AttentionWithContext forward: weighted_input.shape = {weighted_input.shape}")
        output = torch.sum(weighted_input, dim=1)
        print(f"AttentionWithContext forward: output.shape = {output.shape}")
        return output


# nn.MultiheadAttention,
# just for comparison with other attention mechanism
class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim:int, num_heads:int, dropout:float=0., bias:bool=True, add_bias_kv:bool=False, add_zero_attn:bool=False, kdim:Optional[int]=None, vdim:Optional[int]=None):
        """
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self._qkv_same_embed_dim = (self.kdim == embed_dim and self.vdim == embed_dim)

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        """
        """
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class DoubleConv(nn.Sequential):
    """

    building blocks of UNet
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels:int, out_channels:int, filter_length:int, activation:Union[str,nn.Module]='relu', mid_channels:Optional[int]=None) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.__kernel_size = filter_length

        self.add_module(
            "conv_bn_activation_1",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.__mid_channels,
                kernel_size=self.__kernel_size,
                bn=True,
                activation=activation,
            ),
        )
        self.add_module(
            "conv_bn_activation_2",
            Conv_Bn_Activation(
                in_channels=self.__mid_channels,
                out_channels=self.__out_channels,
                kernel_size=self.__kernel_size,
                bn=True,
                activation=activation,
            )
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """
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
    def __init__(self, down_scale:int, in_channels:int, out_channels:int, filter_length:int, activation:Union[str,nn.Module]='relu', mid_channels:Optional[int]=None) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.__kernel_size = filter_length

        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.__down_scale),
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
        """
        """
        _seq_len = seq_len
        for idx, module in enumerate(self):
            if idx == 0:  # max pool
                output_shape = compute_conv_output_shape(
                    input_shape=[batch_size, self.__in_channels, _seq_len]
                )
            elif idx == 1:  # double conv
                output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class UpDoubleConv(nn.Module):
    """
    Upscaling then double conv
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    def __init__(self, up_scale:int, in_channels:int, out_channels:int, filter_length:int, activation:Union[str,nn.Module]='relu', mode:str='bilinear', mid_channels:Optional[int]=None) -> NoReturn:
        """
        Parameters:
        -----------
        to write
        """
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else in_channels // 2
        self.__out_channels = out_channels
        self.__kernel_size = filter_length

        raise NotImplementedError
        # the following has to be checked
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=self.__up_scale,
                mode=mode, align_corners=True,
            )
            self.conv = DoubleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_length=self.__kernel_size,
                activation=activation,
                mid_channels=self.__mid_channels,
            )
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError
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
        """
        """
        raise NotImplementedError


def compute_conv_output_shape(input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, pad:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, checked,

    compute the output shape of a convolution layer
    
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
    pad: int, or sequence of int, default 0,
        pad length(s) of the layer, should be compatible with `input_shape`
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
    dim = len(input_shape)-2
    assert dim > 0, "input_shape should be a sequence of length at least 3, to be a valid (with batch and channel) shape of a non-degenerate Tensor"

    none_dim_msg = "only batch and channel dimension can be `None`"
    if channel_last:
        assert all([n is not None for n in input_shape[1:-1]]), none_dim_msg
    else:
        assert all([n is not None for n in input_shape[2:]]), none_dim_msg

    if isinstance(kernel_size, int):
        _kernel_size = [kernel_size for _ in range(dim)]
    elif len(kernel_size) == dim:
        _kernel_size = kernel_size
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(kernel_size)} dimensions, both not including the channel dimension")
    
    if isinstance(stride, int):
        _stride = [stride for _ in range(dim)]
    elif len(stride) == dim:
        _stride = stride
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(stride)} dimensions, both not including the channel dimension")

    if isinstance(pad, int):
        _pad = [pad for _ in range(dim)]
    elif len(pad) == dim:
        _pad = pad
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(pad)} dimensions, both not including the channel dimension")

    if isinstance(dilation, int):
        _dilation = [dilation for _ in range(dim)]
    elif len(dilation) == dim:
        _dilation = dilation
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(dilation)} dimensions, both not including the channel dimension")
    
    if channel_last:
        _input_shape = input_shape[1:-1]
    else:
        _input_shape = input_shape[2:]
    output_shape = [
        floor( ( ( i + 2*p - d*(k-1) - 1 ) / s ) + 1 ) \
            for i, p, d, k, s in zip(_input_shape, _pad, _dilation, _kernel_size, _stride)
    ]
    if channel_last:
        output_shape = tuple([input_shape[0]] + output_shape + [num_filters])
    else:
        output_shape = tuple([input_shape[0], num_filters] + output_shape)

    return output_shape
