import tensorflow as tf

from . import tensorflow_nn
from hp_names import HP

'''    activation_fun = 'relu'
    l2_lambda = 0.001
    regularizer = tf.keras.regularizers.L2(l2_lambda)
    epochs_count = 50
    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    momentum = 0.9'''


def train_test_model_tf(hparams):
    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, \
    input_dim, num_hid_layers, units_per_layer, out_dim, \
    NormalizationClass, activation_fun, l2_lambda, _, epochs_count, error_fn, _, _ \
        = tensorflow_nn.get_config_for_airfoil_dataset_tensorflow()

    train_split, val_split, test_split, col_names \
        = tensorflow_nn.get_dataset(file_abs_path, batch_size=mini_batch_size, sep=sep,
                                    # dev_split_ratio=0.85, train_split_ratio=0.7
                                    col_names=col_names, target_col_name=target_col_name, has_header_row=has_header_row)

    train_features, train_labels = tensorflow_nn.get_features_and_labels(train_split, target_col_name)
    val_features, val_labels = tensorflow_nn.get_features_and_labels(val_split, target_col_name)

    features_normalizer, labels_normalizer \
        = tensorflow_nn.do_preprocessing(train_split, col_names, train_features, train_labels, NormalizationClass)

    if hparams[HP.ACTIV_FUN].lower() == 'relu':
        activation_fun = tf.nn.relu

    model = tf.keras.models.Sequential(
        tensorflow_nn.get_layers_descr_as_list_tensorflow(input_dim, num_hid_layers, units_per_layer, out_dim,
                                                          activation_fun,
                                                          regularizer = tf.keras.regularizers.L2(hparams[HP.LAMBDA_L2]),
                                                          features_normalizer=features_normalizer)
    )

    opt_args_nesterov = {'learning_rate': hparams[HP.LEARNING_RATE], 'nesterov': True, 'momentum': hparams[HP.MOMENTUM]}
    opt_args_no_nesterov = {'learning_rate': hparams[HP.LEARNING_RATE], 'momentum': hparams[HP.MOMENTUM]}
    opt_args_no_momentum = {'learning_rate': hparams[HP.LEARNING_RATE]}
    if hparams[HP.OPTIMIZER].lower() == 'adam':
        opt = tf.optimizers.Adam(**opt_args_no_momentum)
    elif hparams[HP.OPTIMIZER].lower() == 'adagrad':
        opt = tf.optimizers.Adagrad(**opt_args_no_momentum)
    elif hparams[HP.OPTIMIZER].lower() == 'rmsprop':
        opt = tf.optimizers.RMSprop(**opt_args_no_nesterov)
    elif hparams[HP.OPTIMIZER].lower() == 'sgd':
        opt = tf.optimizers.SGD(**opt_args_nesterov)
    else:
        opt = tf.optimizers.SGD(**opt_args_nesterov)

    # todo: disable call to plots for gridsearch, add boolean param plot
    # todo: check if an additional train/test split is done during this grid search
    # todo: save plot data for later for each run
    # todo: do a plot at the end with all the learning curves (start x axis from 10 epochs for readability)
    # todo: check that there is no extra unwanted data shuffling from tensorflow_nn and dataset_config and utils.py

    model.compile(
      optimizer=opt,
      loss=ERROR_FN,
      metrics=[HP.METRIC_MSE],
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True,
                                                               mode='min', verbose=1, min_delta=0)

    history = model.fit(train_features, train_labels, callbacks=[early_stopping_callback], epochs=900,
                            validation_data=(val_features, val_labels),
                            batch_size=mini_batch_size, verbose=0)
    print(len(history.history['loss']), ' epochs done.')
    metrics_results = model.evaluate(val_features, val_labels, return_dict=True)

    # plot_results(model, metrics_results, history, file_abs_path, activation_fun, mini_batch_size,
    #                  adaptive_lr, error_fn, l2_lambda, NormalizationClass, train_features, train_labels,
    #                  val_features, val_labels)

    '''model.fit(
    ...,
    callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ],)'''

    return metrics_results, history

