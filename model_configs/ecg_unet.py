"""
the model of UNET structures
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "unet",
    "unet_down_block", "unet_up_block",
]


unet = ED()

unet.groups = 1

unet.init_num_filters = 4  # keep the same with n_classes
unet.init_filter_length = 9
unet.out_filter_length = 9
unet.batch_norm = True
unet.kernel_initializer = "he_normal"
unet.kw_initializer = {}
unet.activation = "relu"
unet.kw_activation = {}

unet.down_up_block_num = 4

unet.down_mode = 'max'
unet.down_scales = list(repeat(2, unet.down_up_block_num))
unet.down_num_filters = [
    unet.init_num_filters * (2**idx) for idx in range(1, unet.down_up_block_num+1)
]
unet.down_filter_lengths = list(repeat(unet.init_filter_length, unet.down_up_block_num))

unet.up_mode = 'nearest'
unet.up_scales = list(repeat(2, unet.down_up_block_num))
unet.up_num_filters = [
    unet.init_num_filters * (2**idx) for idx in range(unet.down_up_block_num-1,-1,-1)
]
unet.up_deconv_filter_lengths = list(repeat(9, unet.down_up_block_num))
unet.up_conv_filter_lengths = list(repeat(unet.init_filter_length, unet.down_up_block_num))


unet_down_block = ED()
unet_down_block.batch_norm = unet.batch_norm
unet_down_block.kernel_initializer = unet.kernel_initializer 
unet_down_block.kw_initializer = deepcopy(unet.kw_initializer)
unet_down_block.activation = unet.activation
unet_down_block.kw_activation = deepcopy(unet.kw_activation)


unet_up_block = ED()
unet_up_block.batch_norm = unet.batch_norm
unet_up_block.kernel_initializer = unet.kernel_initializer 
unet_up_block.kw_initializer = deepcopy(unet.kw_initializer)
unet_up_block.activation = unet.activation
unet_up_block.kw_activation = deepcopy(unet.kw_activation)
