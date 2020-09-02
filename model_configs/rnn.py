"""
"""
from copy import deepcopy

from easydict import EasyDict as ED

from cfg import ModelCfg


__all__ = [
    "lstm",
    "attention",
]


lstm = ED()
lstm.bias = True
lstm.dropout = 0.2
lstm.bidirectional = True
lstm.retseq = False
lstm.hidden_sizes = [256, 64]


attention = ED()
# almost the same with lstm, but the last layer is an attention layer
attention.head_num = 1
attention.bias = True
attention.dropout = 0.2
attention.bidirectional = True
attention.hidden_sizes = [256, 64]
