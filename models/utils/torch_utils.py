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
            elif kernel_initializer.lower() in ['he_normal', 'kaiming_normal']:
                nn.init.kaiming_normal_(conv_layer.weight)
            elif kernel_initializer.lower() in ['he_uniform', 'kaiming_uniform']:
                nn.init.kaiming_uniform_(conv_layer.weight)
            elif kernel_initializer.lower() in ['xavier_normal', 'glorot_normal']:
                nn.init.xavier_normal_(conv_layer.weight)
            elif kernel_initializer.lower() in ['xavier_uniform', 'glorot_uniform']:
                nn.init.xavier_uniform_(conv_layer.weight)
            elif kernel_initializer.lower() == 'normal':
                nn.init.normal_(conv_layer.weight)
            elif kernel_initializer.lower() == 'uniform':
                nn.init.uniform_(conv_layer.weight)
            elif kernel_initializer.lower() == 'orthogonal':
                nn.init.orthogonal_(conv_layer.weight)
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
    """

    TODO: adjust for 1D

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

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        """
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
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
    """

    TODO: adjust for 1D, compare with `nn.MultiheadAttention`

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

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

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

    def forward(self, x):
        """
        """
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
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
