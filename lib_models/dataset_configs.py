import os
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

from utils import get_hyperparams_descr


def get_config_for_winequality_dataset_tensorflow():
    import tensorflow as tf
    assets_subdir = 'wine-quality'
    filename = 'winequality-white.csv'
    sep = ';'
    root_dir = os.getcwd()
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/', assets_subdir, filename)
    header = 0
    col_names = None
    dtype1 = np.float32
    target_col_name = 'quality'
    mini_batch_size = 256
    has_header_row = True

    # dev_split_ratio=0.85, train_split_ratio=0.7

    NormalizationClass = tf.keras.layers.experimental.preprocessing.Normalization

    activation_fun = 'relu'
    l2_lambda = 0.001
    regularizer = tf.keras.regularizers.L2(l2_lambda)
    epochs_count = 300
    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = 0.001
    adaptive_lr = tf.optimizers.Adam(learning_rate=learning_rate)

    layer_sizes = (11, 120, 1)

    return file_abs_path, has_header_row, sep, target_col_name, mini_batch_size, layer_sizes, NormalizationClass, \
           activation_fun, l2_lambda, regularizer, epochs_count, error_fn, learning_rate, adaptive_lr


def get_layers_descr_as_list_tensorflow(layer_sizes, activation_fun, regularizer, features_normalizer=None):
    import tensorflow as tf
    layers = []

    print('layer_sizes: ', layer_sizes)

    if features_normalizer is not None:
        layers.append(features_normalizer)
    input_dim = layer_sizes[0]
    layers.append(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    last_layer_idx = len(layer_sizes) - 1
    print('last_layer_idx: ', last_layer_idx)
    for layer_idx in range(len(layer_sizes)):
        is_last_layer = layer_idx >= last_layer_idx
        if not is_last_layer:  # NB.: not adding an activation function after the last layer
            new_layer = tf.keras.layers.Dense(units=layer_sizes[layer_idx], activation=activation_fun,
                                              bias_regularizer=regularizer, activity_regularizer=regularizer)
        else:
            new_layer = tf.keras.layers.Dense(units=layer_sizes[layer_idx])
        layers.append(new_layer)
    return layers


def get_config_for_winequality_dataset_pytorch():
    from torch import nn
    # todo: support more than one target column
    # todo: include here the parameter for test/validation split ratio
    dtype1 = np.float32
    dtype2 = None
    col_names = None
    header = 0
    features_scaler = RobustScaler  # MaxAbsScaler  # MinMaxScaler  # Normalizer  # StandardScaler  #

    # hidden_layer_sizes = (3*32*32, 512, 512, 10)
    hidden_layer_sizes = (11, 120, 1)  # (5, 10, 10, 10, 1)
    activation_fun = nn.RReLU()  # other options: nn.ReLU(), nn.Tanh, ..
    mini_batch_size = 64
    learning_rate = 0.0012
    momentum = 0.1
    adaptive_learning_rate = 'constant'
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()  # BCELoss
    l2_lambda = 0.0014  # 0.003  # 0.005  # 0.01  # 0.001
    model_descr = 'NN MLP regressor (fully connected)'

    assets_subdir = 'wine-quality'
    filename = 'winequality-white.csv'
    sep = ';'
    return get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes,
                                  activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                                  loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep)


def get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes,
                           activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                           loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep):
    root_dir = os.getcwd()
    print('root dir:', root_dir)
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/', assets_subdir, filename)

    hyperparams_descr = get_hyperparams_descr(filename, model_descr, hidden_layer_sizes, activation_fun, learning_rate,
                                              momentum, adaptive_learning_rate, loss_fn, l2_lambda, features_scaler)

    return hidden_layer_sizes, activation_fun, mini_batch_size, learning_rate, momentum, \
           adaptive_learning_rate, loss_fn, l2_lambda, hyperparams_descr, file_abs_path, \
           sep, dtype1, dtype2, header, col_names, features_scaler


def get_config_for_airfoil_dataset_tensorflow():
    import tensorflow as tf
    assets_subdir = 'airfoil'
    filename = 'airfoil_self_noise.dat.csv'
    sep = '\t'
    root_dir = os.getcwd()
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/', assets_subdir, filename)
    dtype1 = {'Freq(Hz)': np.float32, 'AngleOfAttack(deg)': np.float32, 'ChordLgt(mt)': np.float32, 'D': np.float32,
              'E': np.float32, 'F': np.float32}
    dtype2 = {'Freq(Hz)': np.int32, 'AngleOfAttack(deg)': np.int32}
    col_names = ["Freq(Hz)", "AngleOfAttack(deg)", "ChordLgt(mt)", "FreeStreamVel(m/s)", "SSDT(mt)",
                 "ScaledSoundPressLev(db)"]
    target_col_name = 'ScaledSoundPressLev(db)'
    mini_batch_size = 32
    has_header_row = False

    # dev_split_ratio=0.85, train_split_ratio=0.7

    NormalizationClass = tf.keras.layers.experimental.preprocessing.Normalization

    activation_fun = 'relu'
    l2_lambda = 0.001
    regularizer = tf.keras.regularizers.L2(l2_lambda)
    epochs_count = 50
    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = 0.001
    momentum = 0.9
    adaptive_lr = tf.optimizers.Adam(learning_rate=learning_rate)

    layer_sizes = (5, 120, 1)

    return file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, layer_sizes, NormalizationClass, \
           activation_fun, l2_lambda, regularizer, epochs_count, error_fn, learning_rate, adaptive_lr


def get_config_for_airfoil_dataset_pytorch():
    from torch import nn
    dtype1 = {'Freq(Hz)': np.float32, 'AngleOfAttack(deg)': np.float32, 'ChordLgt(mt)': np.float32, 'D': np.float32, 'E': np.float32, 'F': np.float32}
    dtype2 = {'Freq(Hz)': np.int32, 'AngleOfAttack(deg)': np.int32}
    col_names = ["Freq(Hz)", "AngleOfAttack(deg)", "ChordLgt(mt)", "FreeStreamVel(m/s)", "SSDT(mt)", "ScaledSoundPressLev(db)"]

    header = None
    features_scaler = RobustScaler

    # hidden_layer_sizes = (3*32*32, 512, 512, 10)
    hidden_layer_sizes_airflow = (5, 120, 1)  # (5, 10, 10, 10, 1)
    activation_fun = nn.RReLU()  # other options: nn.ReLU(), nn.Tanh, ..
    mini_batch_size = 64
    learning_rate = 0.012
    momentum = 0.9
    adaptive_learning_rate = 'constant'
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()  # BCELoss
    l2_lambda = 0.0014  # 0.003  # 0.005  # 0.01  # 0.001
    model_descr = 'NN MLP regressor (fully connected)'

    assets_subdir = 'airfoil'
    filename = 'airfoil_self_noise.dat.csv'
    sep = '\t'
    return get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes_airflow,
                                  activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                                  loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep)

