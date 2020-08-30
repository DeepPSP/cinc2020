"""

References:
-----------
[1] https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""
import os
import argparse

import numpy as np
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint

from models.legacy import get_model
from cfg import TrainCfg


def train(model, config, train_x, train_y, test_x, test_y):
    """
    """
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(config.checkpoints,'weights.{epoch:04d}-{val_loss:.3f}.hdf5'),
        verbose=2,
        monitor='val_acc',
        save_best_only=False,
    )

    model.fit(train_x, train_y, batch_size=config.batch_size, epochs=config.n_epochs, verbose=2, validation_data=(test_x, test_y), callbacks=[checkpointer])


def get_args(**kwargs):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description='Train the Model on CINC2020',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-l', '--learning-rate',
    #     metavar='LR', type=float, nargs='?', default=0.001,
    #     help='Learning rate',
    #     dest='learning_rate')
    # parser.add_argument(
    #     '-g', '--gpu',
    #     metavar='G', type=str, default='0',
    #     help='GPU',
    #     dest='gpu')
    parser.add_argument(
        '-t', '--tranches',
        type=str, default='',
        help='the tranches for training',
        dest='tranches_for_training')
    parser.add_argument(
        '-c', '--cnn-name',
        type=str, default='resnet',
        help='choice of cnn feature extractor',
        dest='cnn_name')
    parser.add_argument(
        '-r', '--rnn-name',
        type=str, default='lstm',
        help='choice of rnn structures',
        dest='rnn_name')
    parser.add_argument(
        '--keep-checkpoint-max', type=int, default=100,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    parser.add_argument(
        '--optimizer', type=str, default='adam',
        help='training optimizer',
        dest='train_optimizer')
    parser.add_argument(
        '--debug', type=str2bool, default=False,
        help='train with more debugging information',
        dest='debug')
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)


DAS = True  # JD DAS platform

if __name__ == "__main__":
    config = get_args(**TrainCfg)

    print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    print(f"GPU status: {K.tensorflow_backend._get_available_gpus()}")
    print(f"Using keras of version {keras.__version__}")
    print(f'with configuration {config}')

    model = get_model(config)

    train_ratio = int(config.train_ratio*100)
    test_ratio = 100 - train_ratio

    tranches = config.tranches_for_training or "ABEF"

    # not finished
    train_x, train_y, test_x, test_y = [], [], [], []
    for t in tranches:
        train_x.append(np.load(os.path.join(config.db_dir, f"train_X_tranches_{t}_ratio_{train_ratio}_siglen_{config.input_len}.npy")))
        train_y.append(np.load(os.path.join(config.db_dir, f"train_y_tranches_{t}_ratio_{train_ratio}_siglen_{config.input_len}.npy")))
        test_x.append(np.load(os.path.join(config.db_dir, f"test_X_tranches_{t}_ratio_{test_ratio}_siglen_{config.input_len}.npy")))
        test_y.append(np.load(os.path.join(config.db_dir, f"test_y_tranches_{t}_ratio_{test_ratio}_siglen_{config.input_len}.npy")))

    train_x = np.concatenate(train_x, axis=0).transpose(0,2,1)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0).transpose(0,2,1)
    test_y = np.concatenate(test_y, axis=0)

    train(model, config, train_x, train_y, test_x, test_y)
