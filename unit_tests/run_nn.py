# Standard library imports
import os
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from shutil import rmtree, copytree

# Third party imports
import numpy as np
from numpy.random import default_rng

# Local application imports
from lib_models.utils import get_hyperparams_descr
from nn import NeuralNet
from preprocessing import load_and_preprocess_monk_dataset


def run_nn_only_regression():
    # trains basic neural network on the airfoil dataset
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, '..\\datasplitting\\assets\\airfoil\\airfoil_self_noise.dat.csv')
    data = np.loadtxt(data_dir)
    train_ratio = 0.7
    rng = default_rng()
    rng.shuffle(data)
    example_number = data.shape[0]
    activation = 'tanh'
    mini_batch_size = 20
    error_fn = 'MSE'
    adaptive_lr = 'SGD constant lr'
    lr = 1e-4
    lambda_reg = 5e-7
    alpha_momentum = 5e-4
    net_shape = [5, 8, 1]
    split_id = int(np.round(example_number * train_ratio))

    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task='regression', error=error_fn)

    test_net.load_training(data[:split_id], 1)
    test_net.load_validation(data[split_id:], 1)

    hyperparams_descr = get_hyperparams_descr(data_dir, str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer=adaptive_lr)

    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))
    test_net.batch_training(hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print('final validation_error = {}'.format(test_net.validate_net()))


def run_nn_only_classification():
    root_dir = os.getcwd()
    filename = 'monks-2.train'
    file_path = os.path.join(root_dir, '..\\datasplitting\\assets\\monk\\', filename)

    data = load_and_preprocess_monk_dataset(file_path)

    # import seaborn as sns
    # sns.pairplot(pd.DataFrame(data), diag_kind='kde')
    # plt.show()

    rng = default_rng()
    rng.shuffle(data)
    example_number = data.shape[0]
    train_ratio = 0.813
    split_id = int(np.round(example_number * train_ratio))
    print(f'doing {split_id} samples for training, and {example_number - split_id} for validation')
    mini_batch_size = 4

    print('dataset head after shuffling: ')
    print(data[:5])
    task = 'classification'
    activation = 'tanh'  # 'sigmoid' # 'tanh'
    net_shape = [17, 5, 1]
    lr = 1e-2
    alpha_momentum = 5e-2
    lambda_reg = 0  # 0.001  # 0.005
    stopping_threshold = 0.00005
    max_epochs = 1500
    patience = 50

    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task=task, verbose=True)

    test_net.load_training(data[:split_id], 1, do_normalization=False)
    test_net.load_validation(data[split_id:], 1)

    hyperparams_descr = get_hyperparams_descr(filename, str(net_shape), activation, mini_batch_size,
                                              error_fn='MSE', l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer='SGD constant lr')
    print(f'running training with hyperparams: {hyperparams_descr}')
    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))

    test_net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs, stopping='MSE2_val', patience=patience,
                            verbose=False, hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    final_validation_error, accuracy, vl_misc_rate = test_net.validate_net()
    print('final validation_error = {}'.format(final_validation_error))
    print(f'final validation accuracy = {accuracy}')

    # todo: plot actual vs predicted (as accuracy and as MSE smoothing function)


def run_nn_and_tf():
    # trains basic neural network on the airfoil dataset
    print('Getting dataset from file.. {}'.format(datetime.now()))
    root_dir = os.getcwd()
    file_abs_path = os.path.join(root_dir, '..\\datasplitting\\assets\\airfoil\\airfoil_self_noise.dat.csv')
    dev_data = np.loadtxt(file_abs_path)
    print('numpy dataset loaded from file. {}'.format(datetime.now()))
    rng = default_rng()
    rng.shuffle(dev_data)

    example_number = dev_data.shape[0]
    train_ratio = 0.7
    split_id = int(np.round(example_number * train_ratio))
    training_data = dev_data[:split_id]
    validation_data = dev_data[split_id:]

    custom_nn_shape = [5, 8, 1]
    num_hid_layers_tf = 1
    input_dim_tf = 5
    units_per_layer_tf = 8
    out_cols_count = 1

    activation_fun = 'tanh'
    lr_eta = 0.01
    alpha_momentum = 0.12
    lambda_regularization = 0.005
    task_type = 'regression'
    mini_batch_size = 20
    max_epochs = 300
    early_stopping_threshold = 300

    import tensorflow as tf
    tf_regularizer = tf.keras.regularizers.L2(lambda_regularization)
    tf_features_normalizer = None
    tf_adaptive_lr = tf.optimizers.SGD(learning_rate=lr_eta, nesterov=False, momentum=alpha_momentum)
    tf_error_fn = tf.keras.losses.MeanSquaredError()

    test_net = NeuralNet(activation_fun, custom_nn_shape, eta=lr_eta, alpha=alpha_momentum, lamda=lambda_regularization,
                            task=task_type, mb=mini_batch_size, max_epochs=max_epochs, threshold=max_epochs)

    test_net.load_training(training_data, out_cols_count)
    test_net.load_validation(validation_data, out_cols_count)

    # todo: do normalization also on data for tf
    normalized_tr_set = test_net.training_set
    normalized_vl_set = test_net.validation_set

    normalized_tr_features, normalized_tr_labels = np.array_split(normalized_tr_set, [input_dim_tf], axis=1)
    normalized_vl_features, normalized_vl_labels = np.array_split(normalized_vl_set, [input_dim_tf], axis=1)

    from lib_models.tensorflow_nn import create_and_compile_model, fit_and_evaluate_model, plot_results
    tf_model = create_and_compile_model(input_dim_tf, num_hid_layers_tf, units_per_layer_tf, out_cols_count,
                                     activation_fun, tf_regularizer, tf_features_normalizer,
                                     normalized_tr_features, normalized_tr_labels, tf_adaptive_lr, tf_error_fn)
    '''early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20,
                                                               restore_best_weights=True,
                                                               mode='min', verbose=1, min_delta=0)
    '''
    start_time = datetime.now()
    print('custom nn initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))
    test_net.batch_training(stopping='epochs', threshold=300, max_epochs=500)
    end_time = datetime.now()
    print('custm nn training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print('final validation_error = {}'.format(test_net.validate_net()))

    start_time = datetime.now()
    print('tf nn initialized at {}'.format(start_time))
    history, metrics_results = fit_and_evaluate_model(tf_model, normalized_tr_features, normalized_tr_labels,
                                                      mini_batch_size, epochs_count=max_epochs,
                                                      val_features=normalized_vl_features, val_labels=normalized_vl_labels,
                                                      early_stopping_callback=None, max_epochs=max_epochs)
    end_time = datetime.now()
    print('tf nn training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))

    NormalizationClass = None
    plot_results(tf_model, metrics_results, history, file_abs_path, activation_fun, mini_batch_size,
                 tf_adaptive_lr, tf_error_fn, lambda_regularization, NormalizationClass,
                 normalized_tr_features, normalized_tr_labels,
                 normalized_vl_features, normalized_vl_labels)
    # todo: also plot actual vs predicted for the custom nn model


if __name__ == '__main__':
    # todo: collect training history to plot learning curve
    # run_nn_and_tf()
    # run_nn_only_regression()
    run_nn_only_classification()
