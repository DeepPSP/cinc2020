"""
the model of CRNN structures
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish, vgg6,
    resnet_block_stanford, resnet_stanford,
    resnet_block_basic, resnet_bottle_neck, resnet,
)


__all__ = [
    "ECG_CRNN_CONFIG",
]


ECG_CRNN_CONFIG = ED()

# cnn part
ECG_CRNN_CONFIG.cnn = ED()
ECG_CRNN_CONFIG.cnn.name = 'vgg6'


ECG_CRNN_CONFIG.cnn.vgg6 = deepcopy(vgg6)
ECG_CRNN_CONFIG.cnn.vgg6.block = deepcopy(vgg_block_basic)
ECG_CRNN_CONFIG.cnn.vgg6_mish = deepcopy(vgg6)
ECG_CRNN_CONFIG.cnn.vgg6_mish.block = deepcopy(vgg_block_mish)
ECG_CRNN_CONFIG.cnn.vgg6_swish = deepcopy(vgg6)
ECG_CRNN_CONFIG.cnn.vgg6_swish.block = deepcopy(vgg_block_swish)
# ECG_CRNN_CONFIG.cnn.vgg6_dilation = deepcopy(vgg6)
# ECG_CRNN_CONFIG.cnn.vgg6_dilation.block = deepcopy(vgg_block_basic)
ECG_CRNN_CONFIG.cnn.resnet = deepcopy(resnet)
ECG_CRNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_basic)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck = deepcopy(resnet)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck.block = deepcopy(resnet_bottle_neck)
ECG_CRNN_CONFIG.cnn.resnet_stanford = deepcopy(resnet_stanford)
ECG_CRNN_CONFIG.cnn.resnet_stanford.block = deepcopy(resnet_block_stanford)


# rnn part
ECG_CRNN_CONFIG.rnn = ED()
ECG_CRNN_CONFIG.rnn.name = 'lstm'

if ECG_CRNN_CONFIG.rnn.name == 'lstm':
    ECG_CRNN_CONFIG.rnn.bias = True
    ECG_CRNN_CONFIG.rnn.dropout = 0.2
    ECG_CRNN_CONFIG.rnn.bidirectional = True
    ECG_CRNN_CONFIG.rnn.retseq = False
    ECG_CRNN_CONFIG.rnn.hidden_sizes = [256, 128]
elif ECG_CRNN_CONFIG.rnn.name == 'attention':
    pass
else:
    pass
