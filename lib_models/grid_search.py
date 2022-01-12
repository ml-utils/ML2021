import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_nn

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'adagrad', 'rmsprop']))
HP_LEARNING_RATE = hp.HParam('lr', hp.RealInterval(0.001, 0.1))
HP_MOMENTUM = hp.HParam('momentum', hp.RealInterval(0.1, 0.9))
HP_LAMBDA_L2 = hp.HParam('l2_lambda', hp.RealInterval(0.001, 0.1))
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
# 'accuracy'


def grid_search():

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_OPTIMIZER],  # HP_DROPOUT,
            metrics=[hp.Metric(METRIC_MSE, display_name='MSE')],
        )


def run(run_dir, hparams):

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        metrics_results = train_test_model(hparams)
        print('metrics_results: ', metrics_results, ', type: ', type(metrics_results))
        # print('metrics_results.keys(): ', metrics_results.keys())
        tf.summary.scalar(METRIC_MSE, metrics_results[1], step=1)


def train_test_model(hparams):
    file_abs_path, has_header_row, sep, col_names, target_col_name, mini_batch_size, layer_sizes, NormalizationClass, \
    activation_fun, l2_lambda, _, epochs_count, error_fn, _, _ \
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
        tensorflow_nn.get_layers_descr_as_list_tensorflow(layer_sizes, activation_fun,
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

    # todo:disable call to plots for gridsearch, add boolean param plot
    # todo: check if an additional train/test split is done during this grid search
    model.compile(
      optimizer=opt,
      loss=ERROR_FN,
      metrics=[METRIC_MSE],
    )

    history = model.fit(train_features, train_labels, epochs=30)  # Run with 1 epoch to speed things up for demo purposes
    metrics_results = model.evaluate(val_features, val_labels)

    '''model.fit(
    ...,
    callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ],)'''

    return metrics_results


def main():
    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for lr in (HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value):
            for momentum in (HP_MOMENTUM.domain.min_value, HP_MOMENTUM.domain.max_value):
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
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning/' + run_name, hparams)
                        session_num += 1

    # Visualize the results in TensorBoard's HParams plugin
    # % tensorboard - -logdir logs / hparam_tuning


if __name__ == '__main__':
    main()
