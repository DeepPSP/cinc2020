from keras import layers
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Layer,
    LSTM, GRU,
    TimeDistributed, Bidirectional,
    ReLU, LeakyReLU,
    BatchNormalization,
    Dense, Dropout, Activation, Flatten, 
    Input, Reshape, GRU, CuDNNGRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D, AveragePooling1D,
    concatenate,
)
from keras.initializers import he_normal, he_uniform, Orthogonal
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import argparse

from .const import SEED, model_input_len, batch_size, all_labels, nb_leads


class TI_CNN(Sequential):
    """
    """
    def __init__(self, classes:list, input_len:int, bidirectional:bool=True):
        """
        """
        super(Sequential, self).__init__(name='TI_CNN')
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
    
        self._build_model()

    def _build_model(self):
        """
        """
        self.add(
        Conv1D(
            input_shape = (self.input_len, nb_leads),
            filters=64, kernel_size=3, strides=1, padding='same',
            kernel_initializer=he_normal(SEED),
            name='conv1_1',
            )
        )
        self.add(
            BatchNormalization(name='bn1_1',)
        )
        self.add(
            ReLU(name='relu1_1',)
        )
        self.add(
            Conv1D(
                filters=64, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv1_2',
                )
        )
        self.add(
            BatchNormalization(name='bn1_2',)
        )
        self.add(
            ReLU(name='relu1_2',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling1',
            )
        )
        self.add(
            Conv1D(
                filters=128, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv2_1',
                )
        )
        self.add(
            BatchNormalization(name='bn2_1',)
        )
        self.add(
            ReLU(name='relu2_1',)
        )
        self.add(
            Conv1D(
                filters=128, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv2_2',
                )
        )
        self.add(
            BatchNormalization(name='conv2_2',)
        )
        self.add(
            ReLU(name='relu2_2',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling2',
            )
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_1',
                )
        )
        self.add(
            BatchNormalization(name='bn3_1',)
        )
        self.add(
            ReLU(name='relu3_1',)
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_2',
                )
        )

        self.add(
            BatchNormalization(name='bn3_2',)
        )
        self.add(
            ReLU(name='relu3_2',)
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_3',
                )
        )
        self.add(
            BatchNormalization(name='bn3_3',)
        )
        self.add(
            ReLU(name='relu3_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling3',
            )
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_1',
                )
        )
        self.add(
            BatchNormalization(name='bn4_1',)
        )
        self.add(
            ReLU(name='relu4_1',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_2',
                )
        )

        self.add(
            BatchNormalization(name='bn4_2',)
        )
        self.add(
            ReLU(name='relu4_2',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_3',
                )
        )
        self.add(
            BatchNormalization(name='bn4_3',)
        )
        self.add(
            ReLU(name='relu4_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling4',
            )
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_1',
                )
        )
        self.add(
            BatchNormalization(name='bn5_1',)
        )
        self.add(
            ReLU(name='relu5_1',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_2',
                )
        )

        self.add(
            BatchNormalization(name='bn5_2',)
        )
        self.add(
            ReLU(name='relu5_2',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_3',
                )
        )
        self.add(
            BatchNormalization(name='bn5_3',)
    )
        self.add(
            ReLU(name='relu5_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling5',
            )
        )

        self.add(
            Bidirectional(LSTM(
                128, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=True,
                name='bd_lstm1',
            ))
        )

        self.add(
            Bidirectional(LSTM(
                32, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=True,
                name='bd_lstm2',
            ))
        )

        self.add(
            Bidirectional(LSTM(
                9, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=False,
                name='bd_lstm3',
            ))
        )

        self.add(
            Dense(
                self.nb_classes,activation='sigmoid',
                name='prediction',
            )
        )

        self.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

        return self


# def get_model():
#     """ not finished,
#     """
#     TI_CNN = Sequential(name='TI_CNN')

#     TI_CNN.add(
#         Conv1D(
#             input_shape = (freq*cell_len_t, 12),
#             filters=64, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv1_1',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn1_1',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu1_1',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=64, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv1_2',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn1_2',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu1_2',)
#     )
#     TI_CNN.add(
#         MaxPooling1D(
#             pool_size=3, strides=3,
#             name='maxpooling1',
#         )
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=128, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv2_1',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn2_1',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu2_1',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=128, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv2_2',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='conv2_2',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu2_2',)
#     )
#     TI_CNN.add(
#         MaxPooling1D(
#             pool_size=3, strides=3,
#             name='maxpooling2',
#         )
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=256, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv3_1',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn3_1',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu3_1',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=256, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv3_2',
#             )
#     )

#     TI_CNN.add(
#         BatchNormalization(name='bn3_2',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu3_2',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=256, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv3_3',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn3_3',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu3_3',)
#     )
#     TI_CNN.add(
#         MaxPooling1D(
#             pool_size=3, strides=3,
#             name='maxpooling3',
#         )
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv4_1',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn4_1',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu4_1',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv4_2',
#             )
#     )

#     TI_CNN.add(
#         BatchNormalization(name='bn4_2',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu4_2',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv4_3',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn4_3',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu4_3',)
#     )
#     TI_CNN.add(
#         MaxPooling1D(
#             pool_size=3, strides=3,
#             name='maxpooling4',
#         )
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv5_1',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn5_1',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu5_1',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv5_2',
#             )
#     )

#     TI_CNN.add(
#         BatchNormalization(name='bn5_2',)
#     )
#     TI_CNN.add(
#         ReLU(name='relu5_2',)
#     )
#     TI_CNN.add(
#         Conv1D(
#             filters=512, kernel_size=3, strides=1, padding='same',
#             kernel_initializer=he_normal(SEED),
#             name='conv5_3',
#             )
#     )
#     TI_CNN.add(
#         BatchNormalization(name='bn5_3',)
# )
#     TI_CNN.add(
#         ReLU(name='relu5_3',)
#     )
#     TI_CNN.add(
#         MaxPooling1D(
#             pool_size=3, strides=3,
#             name='maxpooling5',
#         )
#     )

#     TI_CNN.add(
#         Bidirectional(LSTM(
#             128, kernel_initializer=Orthogonal(seed=SEED),
#             return_sequences=True,
#             name='bd_lstm1',
#         ))
#     )

#     TI_CNN.add(
#         Bidirectional(LSTM(
#             32, kernel_initializer=Orthogonal(seed=SEED),
#             return_sequences=True,
#             name='bd_lstm2',
#         ))
#     )

#     TI_CNN.add(
#         Bidirectional(LSTM(
#             9, kernel_initializer=Orthogonal(seed=SEED),
#             return_sequences=False,
#             name='bd_lstm3',
#         ))
#     )

#     TI_CNN.add(
#         Dense(
#             nb_classes,activation='sigmoid',
#             name='prediction',
#         )
#     )

#     TI_CNN.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

#     return TI_CNN

if __name__ == '__main__':
    # model = get_model()
    checkpointer = ModelCheckpoint(filepath='./ckpt/weights.hdf5', verbose=1, monitor='val_acc', save_best_only=True)
    csv_logger = CSVLogger('./ckpt/logger.csv')

    # model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

    # TODO: add the following callbacks:
    # LearningRateScheduler, CSVLogger
