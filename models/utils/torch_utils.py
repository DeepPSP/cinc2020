"""
"""
import sys
from typing import Union, NoReturn, Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F


__all__ = [
    "Mish", "Swish",
    "Conv_Bn_Activation",
    "Attention", "GatedAttention",
]


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


# initializers
Initializers = {
    'he_normal': nn.init.kaiming_normal_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'he_uniform': nn.init.kaiming_uniform_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'glorot_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'glorot_uniform': nn.init.xavier_uniform_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'orthogonal': nn.init.orthogonal_,
}


# basic building blocks of CNN
class Conv_Bn_Activation(nn.Sequential):
    """
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, activation:Union[str,callable], kernel_initializer:Optional[Union[str,callable]]=None, bn:bool=True, bias:bool=True) -> NoReturn:
        """

        Parameters:
        -----------
        to write
        """
        super().__init__()
        padding = (kernel_size - 1) // 2  # 'same' padding

        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv_layer.weight)
            elif isinstance(kernel_initializer, str) and kernel_initializer.lower() in Initializers.keys():
                Initializers[kernel_initializer.lower()](conv_layer.weight)
            else:  # TODO: add more activations
                raise ValueError(f"initializer {kernel_initializer} not supported")
        self.add_module("conv1d", conv_layer)

        if bn:
            self.add_module("batch_norm", nn.BatchNorm1d(out_channels))

        if isinstance(activation, str):
            activation = activation.lower()

        if callable(activation):
            act_layer = activation
            act_name = "activation"
        elif activation.lower() == "mish":
            act_layer = Mish()
            act_name = "mish"
        elif activation.lower() == "swish":
            act_layer = Swish()
            act_name = "swish"
        elif activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
            act_name = "relu"
        elif activation.lower() in ["leaky", "leaky_relu"]:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
            act_name = "leaky_relu"
        elif activation.lower() == "linear":
            act_layer = None
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")
            act_layer = None

        if act_layer:
            self.add_module(act_name, act_layer)


# attention
class Attention(nn.Module):
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

class GatedAttention(nn.Module):
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
    """
    from CPSC0236
    """
    def __init__(self, bias:bool=True, initializer:str='glorot_uniform', **kwargs):
        """
        """
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias

    def compute_mask(self, input, input_mask=None):
        """
        """
        return None

    def forward(self, x, mask=None):
        """
        """
        # uit = self.dot_product(x, self.W)
        # if self.bias:
        #     uit += self.b
        # uit = K.tanh(uit)
        # ait = self.dot_product(uit, self.u)
        # a = K.exp(ait)
        # if mask is not None:
        #     a *= K.cast(mask, K.floatx())
        # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # a = K.expand_dims(a)
        # weighted_input = x * a
        # return K.sum(weighted_input, axis=1)



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
        super(MultiheadAttention, self).__init__()
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
