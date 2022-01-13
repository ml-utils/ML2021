import os.path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_nn

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 128]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'rmsprop']))  # , 'sgd', 'adagrad'
HP_LEARNING_RATE = hp.HParam('lr', hp.RealInterval(0.001, 0.01))
HP_MOMENTUM = hp.HParam('momentum', hp.Discrete([0.5]))  # hp.RealInterval(0.5, 0.9))
HP_LAMBDA_L2 = hp.HParam('l2_lambda', hp.RealInterval(min_value=0.001, max_value=0.1))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
'''    activation_fun = 'relu'
    l2_lambda = 0.001
    regularizer = tf.keras.regularizers.L2(l2_lambda)
    epochs_count = 50
    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    momentum = 0.9'''
# remove from hyperparams:
# HP_ACTIV_FUN = hp.HParam('activation', hp.Discrete([tf.nn.relu]))  # activation=activation_fun
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 256]))
ERROR_FN = tf.keras.losses.MeanSquaredError()
METRIC_MSE = 'mse'
METRIC_MAE = 'mae'
METRIC_LOSS = 'loss'
HIST_VAL_LOSS = 'val_loss'
HIST_VAL_MSE = 'val_mse'
# 'accuracy'


def grid_search():

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_OPTIMIZER],  # HP_DROPOUT,
            metrics=[hp.Metric(METRIC_MSE, display_name='MSE')],
        )


def run(log_dir, run_name, hparams):

    run_dir = os.path.join(log_dir, run_name)

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams, trial_id=run_name)  # record the values used in this trial
        metrics_results, history = train_test_model(hparams)
        print('val_loss: ', history.history['val_loss'][-1], ', val_mse: ', history.history['val_mse'][-1])
        # print('metrics_results.keys(): ', metrics_results.keys()) # loss, mse
        tf.summary.scalar('epochs', len(history.history['loss']), step=1)
        tf.summary.scalar(HIST_VAL_LOSS, history.history['val_loss'][-1], step=1)
        tf.summary.scalar(HIST_VAL_MSE, history.history['val_mse'][-1], step=1)
        tf.summary.scalar(METRIC_MSE, metrics_results['mse'], step=1)
        tf.summary.scalar(METRIC_LOSS, metrics_results['loss'], step=1)


def train_test_model(hparams):
    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, \
    input_dim, num_hid_layers, units_per_layer, out_dim, \
    NormalizationClass, activation_fun, l2_lambda, _, epochs_count, error_fn, _, _ \
        = tensorflow_nn.get_config_for_airfoil_dataset_tensorflow()

    train_split, val_split, test_split, col_names \
        = tensorflow_nn.get_dataset1(file_abs_path, batch_size=mini_batch_size, sep=sep,
                       # dev_split_ratio=0.85, train_split_ratio=0.7
                       col_names=col_names, target_col_name=target_col_name, has_header_row=has_header_row)

    train_features, train_labels = tensorflow_nn.get_features_and_labels(train_split, target_col_name)
    val_features, val_labels = tensorflow_nn.get_features_and_labels(val_split, target_col_name)

    features_normalizer, labels_normalizer \
        = tensorflow_nn.do_preprocessing(train_split, col_names, train_features, train_labels, NormalizationClass)

    model = tf.keras.models.Sequential(
        tensorflow_nn.get_layers_descr_as_list_tensorflow(input_dim, num_hid_layers, units_per_layer, out_dim,
                                                          activation_fun,
                                                          regularizer = tf.keras.regularizers.L2(hparams[HP_LAMBDA_L2]),
                                                          features_normalizer=features_normalizer)
    )

    opt_args_nesterov = {'learning_rate': hparams[HP_LEARNING_RATE], 'nesterov': True, 'momentum': hparams[HP_MOMENTUM]}
    opt_args_no_nesterov = {'learning_rate': hparams[HP_LEARNING_RATE], 'momentum': hparams[HP_MOMENTUM]}
    opt_args_no_momentum = {'learning_rate': hparams[HP_LEARNING_RATE]}
    if hparams[HP_OPTIMIZER].lower() == 'adam':
        opt = tf.optimizers.Adam(**opt_args_no_momentum)
    elif hparams[HP_OPTIMIZER].lower() == 'adagrad':
        opt = tf.optimizers.Adagrad(**opt_args_no_momentum)
    elif hparams[HP_OPTIMIZER].lower() == 'rmsprop':
        opt = tf.optimizers.RMSprop(**opt_args_no_nesterov)
    elif hparams[HP_OPTIMIZER].lower() == 'sgd':
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
      metrics=[METRIC_MSE],
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


def main():
    session_num = 0
    repeats_per_run = 3
    total_sessions = len(HP_NUM_UNITS.domain.values) \
                     * len((HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value)) \
                     * len(HP_MOMENTUM.domain.values) \
                     * len((HP_LAMBDA_L2.domain.min_value, HP_LAMBDA_L2.domain.max_value)) \
                     * len(HP_OPTIMIZER.domain.values)

    for num_units in HP_NUM_UNITS.domain.values:
        for lr in (HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value):
            for momentum in HP_MOMENTUM.domain.values:
                for l2_lambda in (HP_LAMBDA_L2.domain.min_value, HP_LAMBDA_L2.domain.max_value):
                    for optimizer in HP_OPTIMIZER.domain.values:
                        hparams = {
                            HP_NUM_UNITS: num_units,
                            # HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_LEARNING_RATE: lr,
                            HP_MOMENTUM: momentum,
                            HP_LAMBDA_L2: l2_lambda
                        }
                        for repeat in range(repeats_per_run):
                            run_name = "run-%d.%d" % (session_num+1, repeat+1)
                            print('--- Starting trial: %s' % run_name, ' of %d x %d' % (total_sessions, repeats_per_run))
                            print({h.name: hparams[h] for h in hparams})
                            run('logs/hparam_tuning/', run_name, hparams)
                        session_num += 1

    # todo: for each CV fold, do 3-5 runs, then calculate mean and sd of metric (training loss)
    # todo: group repeat results for each session
    # todo: in metrics, check and label mse used in training vs validation
    #  check diff btw mse and batch_mse, btw historical vs final mse values
    # repeat for each cv fold: total runs = num sessions x repeats x folds
    # todo: explicit use of alghorithm for weights and bias initialization
    # todo: saving initializations, so can be reused the same for each session (and there are as many as the repeats)

    # Visualize the results in TensorBoard's HParams plugin
    # % tensorboard - -logdir logs / hparam_tuning


if __name__ == '__main__':
    main()
