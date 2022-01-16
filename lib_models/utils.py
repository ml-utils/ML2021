import os

import matplotlib.pyplot as plt
import numpy as np


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


class CustomNormalizer:

    def __init__(self):
        self.median = None
        self.interq_range = None

    def adapt(self, data):
        # todo: calculate median
        # calculate quartiles .. between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
        if isinstance(data, list):
            sorted_list = sorted(data)
            idx_1st_quartile = round(len(sorted_list) * 0.25) - 1
            idx_median = round(len(sorted_list) * 0.5) - 1
            idx_3rd_quartile = round(len(sorted_list) * 0.75) - 1
            self.median = sorted_list[idx_median]
            self.interq_range = sorted_list[idx_3rd_quartile] - sorted_list[idx_1st_quartile]
        if isinstance(data, np.ndarray):
            self.median = np.median(data)
            self.interq_range = np.percentile(data, 75) - np.percentile(data, 25)

    def normalize(self, data, copy=True):
        # (multiply interquartile range by 2, or by the diff by 100% and perc covered by quartiles?)

        if isinstance(data, list):
            if copy:
                working_data = data.copy
            else:
                working_data = data

            for idx, sample in enumerate(working_data):
                working_data[idx] = (working_data[idx] - self.median) / (2 * self.interq_range)
            return working_data

        if isinstance(data, np.ndarray):
            return (data - self.median) / (2 * self.interq_range)

    def undo_normalization(self, data, copy=True):
        if isinstance(data, list):
            if copy:
                working_data = data.copy
            else:
                working_data = data

            for idx, sample in enumerate(working_data):
                working_data[idx] = (working_data[idx] * 2 * self.interq_range) + self.median
            return working_data
        if isinstance(data, np.ndarray):
            return (data * 2 * self.interq_range) + self.median
