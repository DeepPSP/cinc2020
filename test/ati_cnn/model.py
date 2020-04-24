from keras import layers
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import (
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
from keras.callbacks import ModelCheckpoint


SEED = 42
freq = 500
cell_len_t = 6
nb_classes = 9
batch_size=128

all_labels = ['N', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']


TI_CNN = Sequential(name='TI_CNN')

TI_CNN.add(
    Conv1D(
        input_shape = (freq*cell_len_t, 12),
        filters=64, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=64, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    MaxPooling1D(
        pool_size=3, strides=3,
    )
)
TI_CNN.add(
    Conv1D(
        filters=128, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=128, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    MaxPooling1D(
        pool_size=3, strides=3,
    )
)
TI_CNN.add(
    Conv1D(
        filters=256, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=256, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)

TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=256, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    MaxPooling1D(
        pool_size=3, strides=3,
    )
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)

TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    MaxPooling1D(
        pool_size=3, strides=3,
    )
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)

TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    Conv1D(
        filters=512, kernel_size=3, strides=1, padding='same',
        kernel_initializer=he_normal(SEED),
        )
)
TI_CNN.add(
    BatchNormalization()
)
TI_CNN.add(
    ReLU()
)
TI_CNN.add(
    MaxPooling1D(
        pool_size=3, strides=3,
    )
)

TI_CNN.add(
    LSTM(
        128, kernel_initializer=Orthogonal(seed=SEED),
        return_sequences=True,
    )
)

TI_CNN.add(
    LSTM(
        32, kernel_initializer=Orthogonal(seed=SEED),
        return_sequences=True,
    )
)

TI_CNN.add(
    LSTM(
        9, kernel_initializer=Orthogonal(seed=SEED),
        return_sequences=False,
    )
)

TI_CNN.add(
    Dense(nb_classes,activation='sigmoid')
)

TI_CNN.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

checkpointer = ModelCheckpoint(filepath='./tmp/weights.hdf5', verbose=1, monitor='val_acc', save_best_only=True)

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
