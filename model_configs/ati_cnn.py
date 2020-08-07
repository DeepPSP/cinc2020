"""
"""
from easydict import EasyDict as ED


__all__ = [
    "ATI_CNN_CONFIG",
]


ATI_CNN_CONFIG = ED()

# cnn part
ATI_CNN_CONFIG.cnn = ED()
ATI_CNN_CONFIG.cnn.name = 'vgg6'

if ATI_CNN_CONFIG.cnn.name == 'vgg6':
    ATI_CNN_CONFIG.cnn.vgg_block = ED()
    ATI_CNN_CONFIG.cnn.vgg_block.filter_length = 3
    ATI_CNN_CONFIG.cnn.vgg_block.subsample_length = 1
    ATI_CNN_CONFIG.cnn.vgg_block.activation = "mish"
    ATI_CNN_CONFIG.cnn.vgg_block.kernel_initializer = "he_normal"
    ATI_CNN_CONFIG.cnn.vgg_block.batch_norm = True
    ATI_CNN_CONFIG.cnn.vgg_block.pool_kernel = 3
    ATI_CNN_CONFIG.cnn.vgg_block.pool_stride = 3
    ATI_CNN_CONFIG.cnn.vgg6 = ED()
    ATI_CNN_CONFIG.cnn.vgg6.num_convs = [2,2,3,3,3]
    ATI_CNN_CONFIG.cnn.vgg6.num_filters = [64,128,256,512,512]
elif ATI_CNN_CONFIG.cnn.name == 'resnet':  # NOT finished
    ATI_CNN_CONFIG.cnn.resnet_block = ED()
    ATI_CNN_CONFIG.cnn.resnet_bottleneck = ED()
else:
    pass


# rnn part
ATI_CNN_CONFIG.rnn = ED()
ATI_CNN_CONFIG.rnn.name = 'lstm'

if ATI_CNN_CONFIG.rnn.name == 'lstm':
    ATI_CNN_CONFIG.rnn.bidirectional = True
    ATI_CNN_CONFIG.rnn.retseq = False
    ATI_CNN_CONFIG.rnn.hidden_sizes = [128,32]
elif ATI_CNN_CONFIG.rnn.name == 'attention':
    pass
else:
    pass
