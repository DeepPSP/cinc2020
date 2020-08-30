"""
References:
-----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "TrainCfg",
]


TrainCfg = ED()
TrainCfg.fs = 500
TrainCfg.train_ratio = 0.8
TrainCfg.classes = [
    'p',  # pwave
    'N',  # qrs complex
    't',  # twave
    'i',  # isoelectric
]
TrainCfg.class_map = ED(p=1, N=2, t=3, i=0)

# as for `start_from` and `end_at`, see ref. [1] section 3.1
TrainCfg.start_from = int(2 * TrainCfg.fs)
TrainCfg.end_at = int(2 * TrainCfg.fs)
TrainCfg.input_len = int(4 * TrainCfg.fs)

TrainCfg.over_sampling = 2

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd", "rmsprop",

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

# configs of loss function
TrainCfg.loss = 'CrossEntropyLoss'
TrainCfg.eval_every = 20
