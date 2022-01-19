# Standard library imports
from datetime import datetime
from itertools import product
import os.path

# Third party imports
import numpy as np
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

# Local application imports
from hp_names import HP, CFG, RES
from lib_models.grid_search_tf import train_test_model_tf
from nn import NeuralNet
from preprocessing import load_and_preprocess_monk_dataset


AIRF_TF_HP_RANGES = {
    HP.NUM_UNITS_PER_HID_LAYER: hp.HParam(HP.NUM_UNITS_PER_HID_LAYER, hp.Discrete([32, 128])),
    HP.OPTIMIZER: hp.HParam(HP.OPTIMIZER, hp.Discrete(['adam', 'rmsprop'])),  # , 'sgd', 'adagrad'
    HP.LEARNING_RATE: hp.HParam(HP.LEARNING_RATE, hp.RealInterval(0.001, 0.01)),
    HP.MOMENTUM: hp.HParam(HP.MOMENTUM, hp.Discrete([0.5])),  # hp.RealInterval(0.5, 0.9)),
    HP.LAMBDA_L2: hp.HParam(HP.LAMBDA_L2, hp.RealInterval(min_value=0.001, max_value=0.1)),
    # HP.DROPOUT: hp.HParam('dropout', hp.RealInterval(0.1, 0.2)),
    HP.MINI_BATCH_SIZE: hp.HParam(HP.MINI_BATCH_SIZE, hp.Discrete([1, 20])),
    HP.ACTIV_FUN: hp.HParam(HP.ACTIV_FUN, hp.Discrete(['relu']))  # activation=activation_fun
}

MONK_CUSTOM_NET_HP_RANGES = {
    HP.NUM_UNITS_PER_HID_LAYER: hp.HParam(HP.NUM_UNITS_PER_HID_LAYER, hp.Discrete([4, 10])),
    HP.NUM_HID_LAYERS: hp.HParam(HP.NUM_HID_LAYERS, hp.Discrete([1])),
    HP.OPTIMIZER: hp.HParam(HP.OPTIMIZER, hp.Discrete(['SGD constant lr'])),
    HP.LEARNING_RATE: hp.HParam(HP.LEARNING_RATE, hp.RealInterval(0.001, 0.1)),
    HP.MOMENTUM: hp.HParam(HP.MOMENTUM, hp.Discrete([0.1, 0.5])),  # hp.RealInterval(0.5, 0.9)),
    HP.LAMBDA_L2: hp.HParam(HP.LAMBDA_L2, hp.RealInterval(min_value=0.0001, max_value=0.1)),
    # HP.DROPOUT: hp.HParam('dropout', hp.RealInterval(0.1, 0.2)),
    HP.MINI_BATCH_SIZE: hp.HParam(HP.MINI_BATCH_SIZE, hp.Discrete([1, 20])),
    HP.ACTIV_FUN: hp.HParam(HP.ACTIV_FUN, hp.Discrete(['tanh', 'sigmoid'])),
    HP.STOPPING_THRESH: hp.HParam(HP.STOPPING_THRESH, hp.Discrete([0.00005])),
    HP.PATIENCE: hp.HParam(HP.PATIENCE, hp.Discrete([50])),
    HP.MAX_EPOCHS: hp.HParam(HP.MAX_EPOCHS, hp.Discrete([1500])),
    HP.ERROR_FN: hp.HParam(HP.ERROR_FN, hp.Discrete(['MSE'])),  # MEE
    HP.EARLY_STOP_ALG: hp.HParam(HP.EARLY_STOP_ALG, hp.Discrete(['MSE2_val'])),
}
# remove from hyperparams:
# ERROR_FN = tf.keras.losses.MeanSquaredError()
# METRIC_MSE = 'mse'
# METRIC_MEE = 'mee'
# METRIC_MR = 'mr'
# METRIC_LOSS = 'loss'
# HIST_VAL_LOSS = 'val_loss'
# HIST_VAL_MSE = 'val_mse'
# 'accuracy'

HP_RANGES = MONK_CUSTOM_NET_HP_RANGES

MONK_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification'
}


def main():
    model_type = 'monk_custom_nn'
    grid_search_name = cur_time = datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_type
    session_num = 0
    repeats_per_run = 3
    iters = [
        HP_RANGES[HP.NUM_UNITS_PER_HID_LAYER].domain.values,
        HP_RANGES[HP.NUM_HID_LAYERS].domain.values,
        (HP_RANGES[HP.LEARNING_RATE].domain.min_value, HP_RANGES[HP.LEARNING_RATE].domain.max_value),
        HP_RANGES[HP.MOMENTUM].domain.values,
        (HP_RANGES[HP.LAMBDA_L2].domain.min_value, HP_RANGES[HP.LAMBDA_L2].domain.max_value),
        HP_RANGES[HP.OPTIMIZER].domain.values,
        HP_RANGES[HP.MINI_BATCH_SIZE].domain.values,
        HP_RANGES[HP.ACTIV_FUN].domain.values,
        HP_RANGES[HP.STOPPING_THRESH].domain.values,
        HP_RANGES[HP.PATIENCE].domain.values,
        HP_RANGES[HP.MAX_EPOCHS].domain.values,
        HP_RANGES[HP.ERROR_FN].domain.values,
        HP_RANGES[HP.EARLY_STOP_ALG].domain.values,
    ]
    total_sessions = len(list(product(*iters)))
    '''len(HP_RANGES[HP.NUM_UNITS_PER_HID_LAYER].domain.values) \
                 * len((HP_RANGES[HP.LEARNING_RATE].domain.min_value, HP_RANGES[HP.LEARNING_RATE].domain.max_value)) \
                 * len(HP_RANGES[HP.MOMENTUM].domain.values) \
                 * len((HP_RANGES[HP.LAMBDA_L2].domain.min_value, HP_RANGES[HP.LAMBDA_L2].domain.max_value)) \
                 * len(HP_RANGES[HP.OPTIMIZER].domain.values='''

    for num_units, num_hid_layers, lr, momentum, l2_lambda, optimizer, mb_size, activation, stop_thresh, \
        patience, max_epochs, error_fn, early_stopping_alg in product(*iters):
        hparams = {
            HP.NUM_UNITS_PER_HID_LAYER: num_units,
            HP.NUM_HID_LAYERS: num_hid_layers,
            HP.LEARNING_RATE: lr,
            HP.MOMENTUM: momentum,
            HP.LAMBDA_L2: l2_lambda,
            HP.OPTIMIZER: optimizer,
            HP.MINI_BATCH_SIZE: mb_size,
            HP.ACTIV_FUN: activation,
            HP.STOPPING_THRESH: stop_thresh,
            HP.PATIENCE: patience,
            HP.MAX_EPOCHS: max_epochs,
            HP.ERROR_FN: error_fn,
            HP.EARLY_STOP_ALG: early_stopping_alg,
        }
        for repeat in range(repeats_per_run):
            run_name = "run-%d.%d" % (session_num+1, repeat+1)
            print(f'--- Starting trial: {run_name} of {total_sessions} x {repeats_per_run} (cfgs x repeats)')
            print({h.name: hparams[h] for h in hparams})
            run(model_type, 'logs/hparam_tuning/', grid_search_name, run_name, hparams, MONK_CUSTOM_NET_CFG)
        session_num += 1

    # todo: for each CV fold, do 3-5 runs, then calculate mean and sd of metric (training loss)
    # todo: group repeat results for each session
    # todo: in metrics, check and label mse used in training vs validation
    #  check diff btw mse and batch_mse, btw historical vs final mse values
    # repeat for each cv fold: total runs = num sessions x repeats x folds
    # todo: explicit use of alghorithm for weights and bias initialization
    # todo: saving initializations, so can be reused the same for each session (and there are as many as the repeats)
    # todo: make a custom normalizer/denormalizer for output targets

    # Visualize the results in TensorBoard's HParams plugin
    # % tensorboard - -logdir logs / hparam_tuning


def run(model_type, log_dir, grid_search_name, run_name, hparams, cfg):

    run_dir = os.path.join(log_dir, run_name)
    print(f'run, hparams: {hparams}')
    # todo: might be better to just export the grid search results to a csv file and visualize it with excel
    # todo: what about the plots? the auto generated png files should have some id to recognize the trial
        # with tf.summary.create_file_writer(run_dir).as_default():
        #hp.hparams(hparams, trial_id=run_name)  # Write hyperparameter values (on the report file) for a single trial.

    if model_type == 'monk_custom_nn':

        best_tr_error, epochs_done, final_validation_error, accuracy = train_test_custom_nn(hparams, cfg)
        results = {
            RES.last_vl_mse: final_validation_error,
            RES.last_tr_mse: best_tr_error,
            RES.accuracy: accuracy,
            RES.epochs_done: epochs_done
        }
    elif model_type == 'airfoil_tf':
        metrics_results, history = train_test_model_tf(hparams, cfg)
        results = {
            RES.last_vl_loss: history.history['val_loss'][-1],
            RES.last_vl_mse: history.history['val_mse'][-1],
            RES.last_tr_loss: metrics_results['loss'],
            RES.last_tr_mse: metrics_results['mse'],
            RES.epochs_done: len(history.history['loss'])
        }

    print('results: ', results)
    append_trial_info_to_report(log_dir, grid_search_name, run_name, hparams, results)


def append_trial_info_to_report(run_dir, grid_search_name, run_name, hparams, results):
    import csv
    file_abs_path = os.path.join(run_dir, grid_search_name + '.csv')

    trial_info = {'trial': run_name}
    trial_info.update(hparams)
    trial_info.update(results)

    print(f'appending trial results to {file_abs_path}')
    file_already_exists = os.path.exists(file_abs_path)
    write_mode = 'a' if file_already_exists else 'w'
    try:
        with open(file_abs_path, write_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trial_info.keys(), delimiter=';')
            if not file_already_exists:
                writer.writeheader()
            writer.writerow(trial_info)

    except IOError:
        print("I/O error")


def train_test_custom_nn(hparams, cfg):
    root_dir = os.getcwd()
    filename = 'monks-2.train'
    file_path = os.path.join(root_dir, '.\\datasplitting\\assets\\monk\\', filename)

    data = load_and_preprocess_monk_dataset(file_path)

    example_number = data.shape[0]
    train_ratio = 0.813
    split_id = int(np.round(example_number * train_ratio))
    print(f'doing {split_id} samples for training, and {example_number - split_id} for validation')

    print('dataset head after shuffling: ')
    print(data[:5])
    task = cfg[CFG.TASK_TYPE]
    mini_batch_size = hparams[HP.MINI_BATCH_SIZE]
    activation = hparams[HP.ACTIV_FUN]  # 'sigmoid' # 'tanh'

    out_dim = cfg[CFG.OUT_DIM]
    input_dim = cfg[CFG.INPUT_DIM]
    num_hid_layers = hparams[HP.NUM_HID_LAYERS]
    num_units_per_hid_layer = hparams[HP.NUM_UNITS_PER_HID_LAYER]
    net_shape = [input_dim]
    for _ in range(num_hid_layers):
        net_shape.append(num_units_per_hid_layer)
    net_shape.append(out_dim)  # ie net_shape = [17, 10, 1]
    lr = hparams[HP.LEARNING_RATE]  # 0.05
    alpha_momentum = hparams[HP.MOMENTUM] # 0.12
    lambda_reg = hparams[HP.LAMBDA_L2]  # 0  # 0.001  # 0.005
    stopping_threshold = hparams[HP.STOPPING_THRESH]  # 0.00005
    max_epochs = hparams[HP.MAX_EPOCHS]  # 1500
    patience = hparams[HP.PATIENCE]  # 50
    error_fn = hparams[HP.ERROR_FN]  # 'MSE'
    optimizer = hparams[HP.OPTIMIZER]  # 'SGD constant lr'
    early_stopping_alg = hparams[HP.EARLY_STOP_ALG]  # 'MSE2_val'

    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task=task, verbose=True)

    test_net.load_training(data[:split_id], out_dim, do_normalization=False)
    test_net.load_validation(data[split_id:], out_dim)

    from lib_models.utils import get_hyperparams_descr
    hyperparams_descr = get_hyperparams_descr(filename, str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer=optimizer)

    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))

    best_tr_error, epochs_done = test_net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs,
                                                         stopping=early_stopping_alg, patience=patience,
                                                         verbose=False, hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    final_validation_error, accuracy, vl_misc_rate = test_net.validate_net()
    print('final validation_error = {}'.format(final_validation_error))
    print(f'final validation accuracy = {accuracy}')

    # todo: plot actual vs predicted (as accuracy and as MSE smoothing function)
    return best_tr_error, epochs_done, final_validation_error, accuracy


if __name__ == '__main__':
    main()
