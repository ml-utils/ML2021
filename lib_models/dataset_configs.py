import os
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

from lib_models.utils import get_hyperparams_descr
from nn import NeuralNet
import numpy as np
from numpy.random import default_rng
from datetime import datetime
from shutil import rmtree, copytree
import os


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

    from packaging import version
    if version.parse(tf.__version__) < version.parse("2.6.0"):
        NormalizationClass = tf.keras.layers.experimental.preprocessing.Normalization
    else:
        print('doing normalization as per tf v.2.7')
        NormalizationClass = tf.keras.layers.Normalization

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


def get_layers_descr_as_list_tensorflow(input_dim, num_hid_layers, units_per_layer, out_dim, activation_fun,
                                        regularizer, features_normalizer=None):
    import tensorflow as tf
    layers = []

    if features_normalizer is not None:
        layers.append(features_normalizer)
        print('Using normalization layer: ', str(features_normalizer))
    else:
        print('No normalization layer is being used in the model')
    layers.append(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for layer_idx in range(num_hid_layers):
        layers.append(tf.keras.layers.Dense(units=units_per_layer, activation=activation_fun,
                                              bias_regularizer=regularizer, activity_regularizer=regularizer))
    # NB.: not adding an activation function after the last layer
    layers.append(tf.keras.layers.Dense(units=out_dim))

    return layers


def get_config_for_custom_nn():

    # trains basic neural network on the airfoil dataset
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, '..\\datasplitting\\assets\\airfoil\\airfoil_self_noise.dat.csv')
    data = np.loadtxt(data_dir)
    train_ratio = 0.7
    rng = default_rng()
    rng.shuffle(data)
    example_number = data.shape[0]

    lr_eta = 0.01
    alpha_momentum = 0.12
    lambda_regularization = 0.005
    task_type = 'regression'
    net_shape = [5, 8, 1]
    split_id = int(np.round(example_number * train_ratio))
    activation_fun = 'tanh'
    return activation_fun, net_shape, lr_eta,
    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, \
    input_dim, hidden_layers, units_per_hid_layer, out_dim, \
    NormalizationClass, \
    activation_fun, l2_lambda, regularizer, epochs_count, error_fn, learning_rate, adaptive_lr
    test_net = nn.NeuralNet(activation_fun, net_shape, eta=lr_eta, alpha=alpha_momentum, lamda=lambda_regularization,
                            task=task_type)

    test_net.load_training(data[:split_id], 1)
    test_net.load_validation(data[split_id:], 1)

    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))
    test_net.batch_training()
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print('final validation_error = {}'.format(test_net.validate_net()))


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

    from packaging import version
    if version.parse(tf.__version__) < version.parse("2.6.0"):
        NormalizationClass = tf.keras.layers.experimental.preprocessing.Normalization
    else:
        print('doing normalization as per tf v.2.7')
        NormalizationClass = tf.keras.layers.Normalization

    activation_fun = 'relu'
    l2_lambda = 0.001  # 0.001
    regularizer = tf.keras.regularizers.L2(l2_lambda)
    epochs_count = 1500
    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = 0.0001  # 0.001
    momentum = 0.5  # 0.9
    adaptive_lr = tf.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum) # tf.optimizers.Adam(learning_rate=learning_rate)

    layer_sizes = (5, 16, 1)
    input_dim = layer_sizes[0]
    hidden_layers = 1
    units_per_hid_layer = layer_sizes[1]
    out_dim = layer_sizes[2]

    return file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, \
           input_dim, hidden_layers, units_per_hid_layer, out_dim, \
           NormalizationClass, \
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

