import os
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_hyperparams_descr, get_hyperparams_descr
from dataset_configs import get_config_for_winequality_dataset_tensorflow, get_layers_descr_as_list_tensorflow, \
    get_config_for_airfoil_dataset_tensorflow


def get_features_and_labels(dataset_split, target_col_name):
    features = dataset_split.copy()
    labels = features.pop(target_col_name)
    features = features.to_numpy()  # features = np.array(features, names=True)
    labels = labels.to_numpy()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()  # returns Tuple of Numpy arrays
    print('features, patterns count: ',  len(features), ', head: ')
    print(features[:5])
    print('labels, head: ')
    print(labels[:5])

    return features, labels


def shuffle_pandas_dataframe(df):
    fraction_of_rows = 1.0
    return df.sample(frac=fraction_of_rows)


def get_dataset1(file_abs_path, batch_size, sep, dev_split_ratio=0.85, train_split_ratio=0.7,
                 target_col_name=None, has_header_row=False, col_names=None):
    ds = pd.read_csv(file_abs_path, sep=sep, names=col_names, header=0)  # header=header, dtype=dtype1

    if has_header_row:
        col_names = ds.columns
    elif col_names is not None:
        ds.columns = col_names
    print('null values per column: ')
    print(ds.isna().sum())
    # ds = ds.dropna()
    DATASET_SIZE = len(ds)
    times_repeat_dataset = 1
    print('DATASET_SIZE: ', DATASET_SIZE)
    # header=header_rows_nums, names=col_names, dtype=dtype1
    #
    #     shuffle=True,
    #     shuffle_buffer_size=10000,
    print('type(ds): ', type(ds))

    # print(tf.data.experimental.cardinality(ds)) # INFINITE = -1,
    # UNKNOWN = -2, (e.g. when the
    #   dataset source is a file).
    print('ds:')
    print(ds)

    val_split_ratio = dev_split_ratio - train_split_ratio
    test_split_ratio = 1 - dev_split_ratio
    train_size = int(train_split_ratio * DATASET_SIZE)
    val_size = int(val_split_ratio * DATASET_SIZE)
    test_size = int(test_split_ratio * DATASET_SIZE)

    # full_dataset = tf.data.TFRecordDataset(FLAGS.input_file)
    ds_shuffled = shuffle_pandas_dataframe(ds)
    print('ds_shuffled')
    print(ds_shuffled)

    from sklearn.model_selection import train_test_split
    dev_split, test_split = train_test_split(ds_shuffled, test_size=test_split_ratio)
    train_split, val_split = train_test_split(dev_split, test_size=val_split_ratio)
    print('train_split')
    print(train_split)

    return train_split, val_split, test_split, col_names


def create_and_compile_model(layer_sizes, activation_fun, regularizer, features_normalizer,
                             train_features, train_labels, adaptive_lr, error_fn):
    model = tf.keras.models.Sequential(
        get_layers_descr_as_list_tensorflow(layer_sizes, activation_fun, regularizer, features_normalizer)
    )
    print('model.output_shape:')
    print(model.output_shape)

    tf.keras.utils.plot_model(model=model, to_file="model.png", rankdir="LR", dpi=72, show_shapes=True)

    train_predictions = model(train_features).numpy()
    print('predictions on training set before training, head')
    print(train_predictions[:5])

    print('training error_fn(train_labels, train_predictions)')
    print(error_fn(train_labels, train_predictions).numpy())

    model.compile(optimizer=adaptive_lr,
                  loss=error_fn,
                  metrics=['mse'])
    return model


def fit_and_evaluate_model(model, train_features, train_labels, epochs_count, mini_batch_size,
                           val_features, val_labels):
    history = model.fit(train_features, train_labels, epochs=epochs_count, batch_size=mini_batch_size, verbose=0,
                        validation_data=(val_features, val_labels),)
    metrics_results = model.evaluate(val_features, val_labels, return_dict=True, verbose=0)
    return history, metrics_results


def plot_results(model, metrics_results, history, file_abs_path, activation_fun, mini_batch_size,
                 adaptive_lr, error_fn, l2_lambda, NormalizationClass, train_features, train_labels,
                 val_features, val_labels):
    print('validation, metrics_results:')
    print(model.metrics_names)
    print(metrics_results)
    print(metrics_results.keys())

    # Plot history: MAE
    print('history')
    print(history)
    print(history.params)
    print(history.history.keys())
    print('history.history[val_loss][last_value]')
    print(history.history['val_loss'][-1])
    print('history.history[val_mse][last]')
    print(history.history['val_mse'][-1])

    # model_descr =  str(model.summary())
    model_descr = ''
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            model_descr += str(layer.kernel.shape)
        else:
            model_descr += '' # str(layer)

    hyperparams_descr = get_hyperparams_descr(file_abs_path, model_descr, activation_fun,
                          mini_batch_size, adaptive_learning_rate=adaptive_lr, optimizer=adaptive_lr,
                          error_fn=error_fn, l2_lambda=l2_lambda, normalizer=NormalizationClass)
    print('hyperparams_descr:')
    print(hyperparams_descr)
    import matplotlib.gridspec as gridspec
    f = plt.figure()
    f, (ax2) = plt.subplots(1, sharey=False)
    # gs = gridspec.GridSpec(1, 1, )  # 1, 3, height_ratios=[1, 3], sharey=False,
    # ax1 = plt.subplot(gs[0])
    # ax1.set_axis_off()
    # ax2 = plt.subplot(gs[1])
    plot_hyperparams_descr(ax2, hyperparams_descr)
    ax2.plot(history.history['loss'], label='loss', color='red') # MAE (training data)
    ax2.plot(history.history['mse'], label='mse', color='orange')  # MAE (training data)
    ax2.plot(history.history['val_loss'], label='val_loss', color='purple') # MAE (training data)
    ax2.plot(history.history['val_mse'], label='val_mse', color='dodgerblue')  # MAE (training data)
    # plt.ylim(0, 4)

    plt.title('Train and validation error/loss')
    plt.ylabel('Loss/Error')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    # todo: plot predicted vs actual targets
    # add to lists of predicted and actual targets
    # todo: sort labels and predicted
    train_predictions = model(train_features).numpy()
    # test_predictions = dnn_model.predict(test_features).flatten()
    val_predictions = model(val_features).numpy()
    train_labels, train_predictions = (list(t) for t in zip(*sorted(zip(train_labels, train_predictions))))
    val_labels, val_predictions = (list(t) for t in zip(*sorted(zip(val_labels, val_predictions))))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    l1, l2 = plot_predicted_points(ax1, hyperparams_descr, train_labels, train_predictions, 'blue', 'orange')
    l3, l4 = plot_predicted_points(ax2, hyperparams_descr, val_labels, val_predictions, 'blue', 'orange')
    plt.legend([l1, l2], ["actual_labels", "predicted_labels"])
    plt.show()


def get_hyperparams_config_and_data():
    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, layer_sizes, NormalizationClass, \
    activation_fun, l2_lambda, regularizer, epochs_count, error_fn, learning_rate, adaptive_lr \
        = get_config_for_airfoil_dataset_tensorflow()

    train_split, val_split, test_split, col_names \
        = get_dataset1(file_abs_path, batch_size=mini_batch_size, sep=sep,  # dev_split_ratio=0.85, train_split_ratio=0.7
                                    col_names=col_names, target_col_name=target_col_name, has_header_row=True)  #col_names=None
    # nb, as a side effect, these convert from pandas dataframe to numpy ndarray
    print('getting training set features and labels:')
    train_features, train_labels = get_features_and_labels(train_split, target_col_name)
    print('getting validation set features and labels:')
    val_features, val_labels = get_features_and_labels(val_split, target_col_name)
    print('getting test set features and labels:')
    test_features, test_labels = get_features_and_labels(test_split, target_col_name)

    return file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, layer_sizes, \
           NormalizationClass, activation_fun, l2_lambda, regularizer, epochs_count, error_fn, \
           learning_rate, adaptive_lr, train_split, val_split, test_split, col_names, \
           train_features, train_labels, val_features, val_labels, test_features, test_labels


def do_preprocessing(train_split, col_names, train_features, train_labels, NormalizationClass):
    data_analysis(train_split, col_names)
    # tf.keras.layers.Normalization(axis=-1)  # normalization layer
    features_normalizer = NormalizationClass()
    labels_normalizer = NormalizationClass()
    features_normalizer.adapt(np.array(train_features))
    # nb: this does not actually does the normalization preprocessing, which is done in a NN layer
    labels_normalizer.adapt(np.array(train_labels))
    print('normalized.mean.numpy():')
    print(features_normalizer.mean.numpy())
    with np.printoptions(precision=2, suppress=True):
        print('First example:', train_features[:1])
        print('Normalized:', features_normalizer(train_features[:1]).numpy())
    # train_features = normalize(train_features) this is done as a layer
    # print('train_features after normalization:')
    # print(train_features)

    # training_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    # DATASET_SIZE = tf.data.experimental.cardinality(training_ds).numpy()
    # training_batches = training_ds.shuffle(DATASET_SIZE).batch(mini_batch_size)

    # todo: linear base expansion (x^2) with numpy or pandas

    return features_normalizer, labels_normalizer


def main():
    print("TensorFlow version:", tf.__version__)
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)

    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, layer_sizes, \
    NormalizationClass, activation_fun, l2_lambda, regularizer, epochs_count, error_fn, \
    learning_rate, adaptive_lr, train_split, val_split, test_split, col_names, \
    train_features, train_labels, val_features, val_labels, test_features, test_labels \
        = get_hyperparams_config_and_data()
    features_normalizer, labels_normalizer\
        = do_preprocessing(train_split, col_names, train_features, train_labels, NormalizationClass)

    model = create_and_compile_model(layer_sizes, activation_fun, regularizer, features_normalizer,
                                     train_features, train_labels, adaptive_lr, error_fn)

    history, metrics_results = fit_and_evaluate_model(model, train_features, train_labels, epochs_count, mini_batch_size,
                           val_features, val_labels)

    plot_results(model, metrics_results, history, file_abs_path, activation_fun, mini_batch_size,
                 adaptive_lr, error_fn, l2_lambda, NormalizationClass, train_features, train_labels,
                 val_features, val_labels)


def plot_predicted_points(ax, hyperparams_descr, actual_labels, predicted_labels, actual_color, predicted_color):

    plot_hyperparams_descr(ax, hyperparams_descr)
    l1 = ax.scatter(range(len(actual_labels)), actual_labels, color=actual_color)  #  l2, = ax2.plot(actual_labels, '-o', color='blue')
    l2 = ax.scatter(range(len(predicted_labels)), predicted_labels, color=predicted_color)  # '-o',
    ax.set_title('actual and predicted labels')
    ax.set_xlabel('sample index')
    ax.set_ylabel('sample label value')
    return l1, l2


def model_summary():
    import tf_slim as slim
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def data_analysis(train_split, col_names):
    import seaborn as sns

    sns.pairplot(train_split, diag_kind='kde')  # train_dataset[col_names] ie col names: ['MPG', 'Cylinders', 'Displacement', 'Weight']
    print('train_split.describe().transpose()')
    print(train_split.describe().transpose()[['mean', 'std']])


if __name__ == '__main__':
    main()

