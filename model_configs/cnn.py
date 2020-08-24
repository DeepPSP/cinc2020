"""
configs for the basic cnn layers and blocks
"""
from copy import deepcopy

from easydict import EasyDict as ED

from cfg import ModelCfg


__all__ = [
    "vgg_block_basic", "vgg_block_mish", "vgg_block_swish",
    "vgg16", "vgg16_leadwise",
    "resnet_block_basic", "resnet_bottle_neck",
    "resnet", "resnet_leadwise",
    "resnet_block_stanford",
    "resnet_stanford",
    "cpsc_block_basic", "cpsc_block_mish", "cpsc_block_swish",
    "cpsc_2018", "cpsc_2018_leadwise",
]


# VGG
vgg_block_basic = ED()
vgg_block_basic.filter_length = 3
vgg_block_basic.subsample_length = 1
vgg_block_basic.dilation = 1
vgg_block_basic.batch_norm = True
vgg_block_basic.pool_size = 3
vgg_block_basic.kernel_initializer = "he_normal"
vgg_block_basic.kw_initializer = {}
vgg_block_basic.activation = "relu"
vgg_block_basic.kw_activation = {}

vgg_block_mish = deepcopy(vgg_block_basic)
vgg_block_mish.activation = "mish"

vgg_block_swish = deepcopy(vgg_block_basic)
vgg_block_swish.activation = "swish"

vgg16 = ED()
vgg16.num_convs = [2, 2, 3, 3, 3]
vgg16.num_filters = [64, 128, 256, 512, 512]
vgg16.groups = 1


vgg16_leadwise = deepcopy(vgg16)
vgg16_leadwise.groups = 12


# ResNet
resnet = ED()
resnet.num_blocks = [
    2, 2, 2, 2, 2,
]
resnet.groups = 1
resnet.init_num_filters = 32
resnet.init_filter_length = 15  # corr. to 30 ms
resnet.init_conv_stride = 2
resnet.init_pool_size = 3
resnet.init_pool_stride = 2
resnet.subsample_length = 2
resnet.kernel_initializer = "he_normal"
resnet.kw_initializer = {}
resnet.activation = "relu"  # "mish", "swish"
resnet.kw_activation = {}
resnet.bias = False

resnet_leadwise = deepcopy(resnet)
resnet_leadwise.groups = 12

resnet_block_basic = ED()
resnet_block_basic.increase_channels_method = 'conv'  # or 'zero_padding'
resnet_block_basic.subsample_method = 'conv'  # or 'max', 'avg'
resnet_block_basic.filter_length = 3
resnet_block_basic.kernel_initializer = resnet.kernel_initializer
resnet_block_basic.kw_initializer = deepcopy(resnet.kw_initializer)
resnet_block_basic.activation = resnet.activation
resnet_block_basic.kw_activation = deepcopy(resnet.kw_activation)
resnet_block_basic.bias = False

resnet_bottle_neck = ED()
resnet_bottle_neck.kernel_initializer = resnet.kernel_initializer
resnet_bottle_neck.kw_initializer = deepcopy(resnet.kw_initializer)
resnet_bottle_neck.activation = resnet.activation
resnet_bottle_neck.kw_activation = deepcopy(resnet.kw_activation)
resnet_bottle_neck.bias = False


# ResNet Stanford
resnet_stanford = ED()

resnet_block_stanford = ED()
resnet_block_stanford.increase_channels_at = 4
resnet_block_stanford.increase_channels_method = 'conv'  # or 'zero_padding'
resnet_block_stanford.num_skip = 2
resnet_block_stanford.subsample_lengths = [
    1, 2, 1, 2,
    1, 2, 1, 2,
    1, 2, 1, 2,
    1, 2, 1, 2,
]
resnet_block_stanford.subsample_method = 'conv'  # 'max', 'avg'
resnet_block_stanford.filter_length = 17
resnet_block_stanford.num_filters_start = 32
resnet_block_stanford.kernel_initializer = "he_normal"
resnet_block_stanford.kw_initializer = {}
resnet_block_stanford.activation = "relu"
resnet_block_stanford.kw_activation = {}
resnet_block_stanford.dropout = 0.2


# CPSC
cpsc_block_basic = ED()
cpsc_block_basic.activation = "leaky"
cpsc_block_basic.kw_activation = ED(negative_slope=0.3)
cpsc_block_basic.batch_norm = False
cpsc_block_basic.kernel_initializer = "he_normal"
cpsc_block_basic.kw_initializer = {}

cpsc_block_mish = deepcopy(cpsc_block_basic)
cpsc_block_mish.activation = "mish"
del cpsc_block_mish.kw_activation

cpsc_block_swish = deepcopy(cpsc_block_basic)
cpsc_block_swish.activation = "swish"
del cpsc_block_swish.kw_activation

cpsc_2018 = ED()
# cpsc_2018.num_filters = [  # original
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
# ]
cpsc_2018.num_filters = [  # original
    [32, 32, 32],
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
]
cpsc_2018.filter_lengths = [
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 48],
]
cpsc_2018.subsample_lengths = [
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
]
cpsc_2018.dropouts = [0.2, 0.2, 0.2, 0.2, 0.2]
cpsc_2018.groups = 1

cpsc_2018_leadwise = deepcopy(cpsc_2018)
cpsc_2018_leadwise.groups = 12


# TODO: add more
