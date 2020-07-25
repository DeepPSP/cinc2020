"""
"""
import sys
from typing import Union, NoReturn, Optional

import torch
from torch import nn
from torch import Tensor
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
    def __init__(self):
        """
        """
        super().__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        """
        A = self.attention(input)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, input)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        """
        """
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        """
        """
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    """ NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    TODO: compare with `nn.MultiheadAttention`

    References:
    -----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L72
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        """
        A_V = self.attention_V(input)  # NxD
        A_U = self.attention_U(input)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        """
        """
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        """
        """
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class AttentionWithContext(nn.Module):
    """
    from CPSC0236
    """
    def __init__(self, bias:bool=True, initializer:str='glorot_uniform', **kwargs):
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias


    def compute_mask(self, input, input_mask=None):
        return None

    def forward(self, x, mask=None):
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
