import os

import matplotlib.pyplot as plt


def plot_hyperparams_descr(ax, hyperparams_descr):
    plt.text(0.82, 0.95, hyperparams_descr, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)


def get_hyperparams_descr(file_abs_path, model_descr, activation_fun, mini_batch_size, error_fn, l2_lambda,
                          momentum=None, learning_rate=None, hidden_layer_sizes=None, normalizer=None, optimizer=None,
                          features_scaler=None):
    hyperparams_descr_list = []

    hyperparams_descr_list.append("Dataset: " + os.path.basename(file_abs_path))
    if features_scaler is not None:
        hyperparams_descr_list.append("\n" + "Scaler: " + str(features_scaler))
    hyperparams_descr_list.append("\n" + "Model: " + model_descr)
    if hidden_layer_sizes is not None:
        hyperparams_descr_list.append("\n" + "Layers, nodes: " + str(hidden_layer_sizes))
    hyperparams_descr_list.append("\n" + "Activation_fun: " + str(activation_fun)[0:5])
    hyperparams_descr_list.append("\n"  "Mini_batch_size: " + str(mini_batch_size))
    if optimizer is not None:
        hyperparams_descr_list.append("\n" + "Adaptive_learning_rate: " + optimizer.__class__.__name__)
    if learning_rate is not None:
        hyperparams_descr_list.append("\n" + "Learning_rate: " + str(learning_rate))
    if normalizer is not None:
        hyperparams_descr_list.append("\n" + "Normalizer: " + normalizer.__name__)
    if momentum is not None:
        hyperparams_descr_list.append("\n" + "Momentum: " + str(momentum))
    hyperparams_descr_list.append("\n" + "Error fn: " + error_fn.__class__.__name__)
    hyperparams_descr_list.append("\n" + "L2 lambda: " + str(l2_lambda))

    return ''.join(hyperparams_descr_list)

