"""
the model of (attention-based) time-incremental CNN

the cnn layers of this model has a constant kernel size 3,
but keep increasing the number of channels
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish,
    vgg6,
)


__all__ = [
    "ATI_CNN_CONFIG",
]


ATI_CNN_CONFIG = ED()

# cnn part
ATI_CNN_CONFIG.cnn = ED()
ATI_CNN_CONFIG.cnn.name = 'vgg6'


if ATI_CNN_CONFIG.cnn.name == 'vgg6':
    ATI_CNN_CONFIG.cnn.vgg_block = deepcopy(vgg_block_basic)
    ATI_CNN_CONFIG.cnn.vgg6 = deepcopy(vgg6)
elif ATI_CNN_CONFIG.cnn.name == 'vgg6_mish':
    ATI_CNN_CONFIG.cnn.vgg_block = deepcopy(vgg_block_mish)
    ATI_CNN_CONFIG.cnn.vgg6 = deepcopy(vgg6)
elif ATI_CNN_CONFIG.cnn.name == 'vgg6_swish':
    ATI_CNN_CONFIG.cnn.vgg_block = deepcopy(vgg_block_swish)
    ATI_CNN_CONFIG.cnn.vgg6 = deepcopy(vgg6)
elif ATI_CNN_CONFIG.cnn.name == 'vgg6_dilation':  # not finished
    ATI_CNN_CONFIG.cnn.vgg_block = deepcopy(vgg_block_basic)
    ATI_CNN_CONFIG.cnn.vgg6 = deepcopy(vgg6)
elif ATI_CNN_CONFIG.cnn.name == 'resnet':  # NOT finished
    ATI_CNN_CONFIG.cnn.resnet_block = ED()
    ATI_CNN_CONFIG.cnn.resnet_bottleneck = ED()
else:
    pass


# rnn part
ATI_CNN_CONFIG.rnn = ED()
ATI_CNN_CONFIG.rnn.name = 'lstm'

if ATI_CNN_CONFIG.rnn.name == 'lstm':
    ATI_CNN_CONFIG.rnn.bias = True
    ATI_CNN_CONFIG.rnn.dropout = 0.2
    ATI_CNN_CONFIG.rnn.bidirectional = True
    ATI_CNN_CONFIG.rnn.retseq = False
    ATI_CNN_CONFIG.rnn.hidden_sizes = [128,32]
elif ATI_CNN_CONFIG.rnn.name == 'attention':
    pass
else:
    pass
