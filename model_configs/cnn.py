"""
configs for the basic cnn layers and blocks
"""
from copy import deepcopy

from easydict import EasyDict as ED

from cfg import ModelCfg


__all__ = [
    "vgg_block_basic", "vgg_block_mish", "vgg_block_swish",
    "vgg6",
    "resnet_block_basic",
    "resnet",
    "cpsc_block_basic", "cpsc_block_mish", "cpsc_block_swish",
    "cpsc_2018",
]


# VGG
vgg_block_basic = ED()
vgg_block_basic.filter_length = 3
vgg_block_basic.subsample_length = 1
vgg_block_basic.dilation = 1
vgg_block_basic.batch_norm = True
vgg_block_basic.pool_kernel = 3
vgg_block_basic.pool_stride = 3
vgg_block_basic.kernel_initializer = "he_normal"
vgg_block_basic.activation = "relu"

vgg_block_mish = deepcopy(vgg_block_basic)
vgg_block_mish.activation = "mish"

vgg_block_swish = deepcopy(vgg_block_basic)
vgg_block_mish.activation = "swish"

vgg6 = ED()
vgg6.num_convs = [2, 2, 3, 3, 3]
vgg6.num_filters = [64, 128, 256, 512, 512]


# ResNet
resnet_block_basic = ED()

resent = ED()


# CPSC
cpsc_block_basic = ED()
cpsc_block_basic.activation = "leaky"
cpsc_block_basic.kw_activation = ED(negative_slope=0.3)
cpsc_block_basic.batch_norm = False
cpsc_block_basic.kernel_initializer = "he_normal"

cpsc_block_mish = deepcopy(cpsc_block_basic)
cpsc_block_mish.activation = "mish"
del cpsc_block_mish.kw_activation

cpsc_block_swish = deepcopy(cpsc_block_basic)
cpsc_block_swish.activation = "swish"
del cpsc_block_swish.kw_activation

cpsc_2018 = ED()
cpsc_2018.filter_lengths = [
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 48],
]
cpsc_2018.strides = [
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
]
cpsc_2018.dropouts = [0.2, 0.2, 0.2, 0.2, 0.2]


# TODO: add more
