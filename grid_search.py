# Standard library imports
from datetime import datetime
from itertools import product
import os.path
import statistics
import csv
from decimal import Decimal

# Third party imports
import numpy as np
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

# Local application imports
from hp_names import HP, CFG, RES
from lib_models.grid_search_tf import train_test_model_tf
from nn import NeuralNet
from preprocessing import load_and_preprocess_monk_dataset, get_cup_dev_set_fold_splits


CUP_CUSTOM_NET_HP_RANGES = {
    HP.UNITS_PER_LAYER: hp.HParam(HP.UNITS_PER_LAYER, hp.Discrete([5, 10])),
    HP.N_HID_LAYERS: hp.HParam(HP.N_HID_LAYERS, hp.Discrete([1, 2])),
    HP.OPTIMIZER: hp.HParam(HP.OPTIMIZER, hp.Discrete(['SGD constant lr'])),
    HP.LR: hp.HParam(HP.LR, hp.Discrete([0.001, 0.01])),
    HP.MOMENTUM: hp.HParam(HP.MOMENTUM, hp.Discrete([0.004, 0.04])),
    HP.LAMBDA_L2: hp.HParam(HP.LAMBDA_L2, hp.RealInterval(min_value=0.0005, max_value=0.005)),
    HP.MB: hp.HParam(HP.MB, hp.Discrete([8, 32])),
    HP.ACTIV_FUN: hp.HParam(HP.ACTIV_FUN, hp.Discrete(['tanh', 'sigmoid'])),
    HP.STOPPING_THRESH: hp.HParam(HP.STOPPING_THRESH, hp.Discrete([0.001])),
    HP.PATIENCE: hp.HParam(HP.PATIENCE, hp.Discrete([150])),
    HP.MAX_EPOCHS: hp.HParam(HP.MAX_EPOCHS, hp.Discrete([2000])),
    HP.ERROR_FN: hp.HParam(HP.ERROR_FN, hp.Discrete(['MSE'])),  # MSE
    HP.EARLY_STOP_ALG: hp.HParam(HP.EARLY_STOP_ALG, hp.Discrete(['MSE2_val'])),
    # HP.DROPOUT: hp.HParam('dropout', hp.Discrete([0.1, 0.2])),
}

MONK_CUSTOM_NET_HP_RANGES = {
    HP.UNITS_PER_LAYER: hp.HParam(HP.UNITS_PER_LAYER, hp.Discrete([5, 10])),
    HP.N_HID_LAYERS: hp.HParam(HP.N_HID_LAYERS, hp.Discrete([1])),
    HP.OPTIMIZER: hp.HParam(HP.OPTIMIZER, hp.Discrete(['SGD constant lr'])),
    HP.LR: hp.HParam(HP.LR, hp.Discrete([0.01, 0.1])),
    HP.MOMENTUM: hp.HParam(HP.MOMENTUM, hp.Discrete([0.01, 0.1])),
    HP.LAMBDA_L2: hp.HParam(HP.LAMBDA_L2, hp.RealInterval(min_value=0.0, max_value=0.0001)),
    HP.MB: hp.HParam(HP.MB, hp.Discrete([1, 20])),
    HP.ACTIV_FUN: hp.HParam(HP.ACTIV_FUN, hp.Discrete(['tanh', 'sigmoid'])),
    HP.STOPPING_THRESH: hp.HParam(HP.STOPPING_THRESH, hp.Discrete([0.01])),
    HP.PATIENCE: hp.HParam(HP.PATIENCE, hp.Discrete([50])),
    HP.MAX_EPOCHS: hp.HParam(HP.MAX_EPOCHS, hp.Discrete([2000])),
    HP.ERROR_FN: hp.HParam(HP.ERROR_FN, hp.Discrete(['MSE'])),  # MEE
    HP.EARLY_STOP_ALG: hp.HParam(HP.EARLY_STOP_ALG, hp.Discrete(['MSE2_val'])),
    # HP.DROPOUT: hp.HParam('dropout', hp.Discrete([0.1, 0.2])),
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


CUP_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 2,
    CFG.INPUT_DIM: 10,
    CFG.TASK_TYPE: 'regression',
    CFG.MODEL_TYPE: 'CUP_custom_nn',
    CFG.DATASET_FILENAME: 'dev_split.csv',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\ml-cup21-internal_splits\\',
    CFG.CV_NUM_SPLITS: 3,
    CFG.HP_RANGES: CUP_CUSTOM_NET_HP_RANGES,
}

MONK1_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-1.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
    CFG.HP_RANGES: MONK_CUSTOM_NET_HP_RANGES,
}

MONK2_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-2.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
    CFG.HP_RANGES: MONK_CUSTOM_NET_HP_RANGES,
}

MONK3_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-3.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
    CFG.HP_RANGES: MONK_CUSTOM_NET_HP_RANGES,
}

# todo: pass dataset filename and trial name in the grid search to the batchtraining method to save
# the error plot img file with a more descriptive name
# todo: collect the plot images of the trials of a gridsearch all in the same folder
# todo: aggregate the results of the multiple trial executions of the same configuration (same hyperparams);
#  aggregate by best, mean or median
# todo: optimize: load and preprocess the dataset once, then pass it to the gridsearch as a param


def do_grid_search(cfg, cv_fold=None):
    # todo: add cross validation option
    # with number of folds
    # dataset is split .. and passed accordingly to the load_training and load_validation methods

    model_type = cfg[CFG.MODEL_TYPE]
    if cv_fold is not None:
        fold_affix = '-fold-' + str(cv_fold)
    else:
        fold_affix = ''
    grid_search_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '-gs' + fold_affix + '-' + model_type
    session_num = 0
    repeats_per_trial = 5
    HP_RANGES = cfg[CFG.HP_RANGES]
    iters = [
        HP_RANGES[HP.UNITS_PER_LAYER].domain.values,
        HP_RANGES[HP.N_HID_LAYERS].domain.values,
        HP_RANGES[HP.LR].domain.values,
        HP_RANGES[HP.MOMENTUM].domain.values,
        (HP_RANGES[HP.LAMBDA_L2].domain.min_value, HP_RANGES[HP.LAMBDA_L2].domain.max_value),
        HP_RANGES[HP.OPTIMIZER].domain.values,
        HP_RANGES[HP.MB].domain.values,
        HP_RANGES[HP.ACTIV_FUN].domain.values,
        HP_RANGES[HP.STOPPING_THRESH].domain.values,
        HP_RANGES[HP.PATIENCE].domain.values,
        HP_RANGES[HP.MAX_EPOCHS].domain.values,
        HP_RANGES[HP.ERROR_FN].domain.values,
        HP_RANGES[HP.EARLY_STOP_ALG].domain.values,
    ]
    total_sessions = len(list(product(*iters)))

    for num_units, num_hid_layers, lr, momentum, l2_lambda, optimizer, mb_size, activation, stop_thresh, \
        patience, max_epochs, error_fn, early_stopping_alg in product(*iters):
        hparams = {
            HP.UNITS_PER_LAYER: num_units,
            HP.N_HID_LAYERS: num_hid_layers,
            HP.LR: lr,
            HP.MOMENTUM: momentum,
            HP.LAMBDA_L2: l2_lambda,
            HP.OPTIMIZER: optimizer,
            HP.MB: mb_size,
            HP.ACTIV_FUN: activation,
            HP.STOPPING_THRESH: stop_thresh,
            HP.PATIENCE: patience,
            HP.MAX_EPOCHS: max_epochs,
            HP.ERROR_FN: error_fn,
            HP.EARLY_STOP_ALG: early_stopping_alg,
        }
        group_of_trial_repeats = []
        for repeat in range(repeats_per_trial):
            if cv_fold is not None:
                fold_name = f'cv_fold {cv_fold} of {cfg[CFG.CV_NUM_SPLITS]}, '
            else:
                fold_name = ''
            session_name = f'session-{session_num+1}'
            trial_name = f"trial-{session_num+1}.{repeat+1}"
            print(f'--- Starting trial: {fold_name} {trial_name} of {total_sessions} x {repeats_per_trial} (cfgs x repeats)')
            print({h.name: hparams[h] for h in hparams})

            grid_search_logdir = 'logs/hparam_tuning/'
            trial_results = run_trial(cfg, grid_search_logdir, grid_search_name, trial_name, hparams, cv_fold)
            group_of_trial_repeats.append(trial_results)
            append_trial_info_to_report(grid_search_logdir, grid_search_name, cfg[CFG.DATASET_FILENAME], trial_name,
                                        hparams, trial_results)

        aggregated_results = get_aggregate_results(group_of_trial_repeats, hparams)
        # print(f'aggregated_results: {aggregated_results}')
        append_trial_info_to_report(grid_search_logdir, grid_search_name, cfg[CFG.DATASET_FILENAME], session_name,
                                    hparams, aggregated_results, file_suffix='-aggregated')
        session_num += 1

    # todo: for each CV fold, do 3-5 trials, then calculate mean and sd of metric (training loss)
    # todo: group repeat results for each session
    # todo: in metrics, check and label mse used in training vs validation
    #  check diff btw mse and batch_mse, btw historical vs final mse values
    # repeat for each cv fold: total trials = num sessions x repeats x folds
    # todo: explicit use of alghorithm for weights and bias initialization
    # todo: saving initializations, so can be reused the same for each session (and there are as many as the repeats)
    # todo: make a custom normalizer/denormalizer for output targets

    # Visualize the results in TensorBoard's HParams plugin
    # % tensorboard - -logdir logs / hparam_tuning


def get_aggregate_results(group_of_trial_repeats, hparams):
    # todo, check: currently median of each metric is calculated independently, therefore
    #  the trial that shows the median mse_tr might not be the same with the median mse_vl
    aggregated_results = dict()

    results_to_aggregate = [metric_name.name for metric_name in RES]
    '''results_to_aggregate = [
        RES.mse_vl_last.name,
        RES.mse_tr_last.name,
        RES.accuracy.name,
        RES.epochs_done.name,
        RES.loss_vl_last.name,
        RES.loss_tr_last.name,
    ]'''
    actual_keys_in_results = group_of_trial_repeats[0].keys()
    print(f'group_of_trial_repeats[0]: {group_of_trial_repeats[0]}')
    for metric in list(results_to_aggregate):
        if metric not in actual_keys_in_results:
            # print(f'aggregating results: metric {metric} not found in {actual_keys_in_results}')
            results_to_aggregate.remove(metric)

    for numeric_metric in results_to_aggregate:
        values = []
        metric_is_decimal = False
        for trial_results in group_of_trial_repeats:
            if numeric_metric != RES.crashed.name:
                if type(trial_results[numeric_metric]) == float:
                    trial_results[numeric_metric] = f'{trial_results[numeric_metric]:0.3f}'
                if type(trial_results[numeric_metric]) == str:
                    trial_results[numeric_metric] = Decimal(trial_results[numeric_metric])
                values.append(trial_results[numeric_metric])
                metric_is_decimal = True
        if metric_is_decimal:
            if len(values) == 0:
                print(f'Warning/Error: trial results for {numeric_metric} are empty!')
                print(f'{group_of_trial_repeats}')
            else:
                try:
                    aggregated_results[numeric_metric + '-median'] = f'{statistics.median(values):0.3f}'
                    aggregated_results[numeric_metric + '-sd'] = f'{statistics.stdev(values):0.3f}'
                except Exception as e:
                    print(f'Warning/Error: unable to aggregate trial results for {numeric_metric} are empty!')
                    print(f'{group_of_trial_repeats}')

    crashed_trials_in_session = 0
    for trial_results in group_of_trial_repeats:
        if trial_results[RES.crashed.name]:
            crashed_trials_in_session += 1

    aggregated_results['trials'] = len(group_of_trial_repeats)
    aggregated_results['crashes'] = crashed_trials_in_session

    print(f'aggregated_results: {aggregated_results}')
    return aggregated_results


def run_trial(cfg, grid_search_logdir, grid_search_name, trial_name, hparams, cv_fold):

    # trial_dir = os.path.join(log_dir, trial_name)
    # todo: might be better to just export the grid search results to a csv file and visualize it with excel
    # todo: what about the plots? the auto generated png files should have some id to recognize the trial
        # with tf.summary.create_file_writer(trial_dir).as_default():
        #hp.hparams(hparams, trial_id=trial_name)  # Write hyperparameter values (on the report file) for a single trial.

    if cfg[CFG.MODEL_TYPE] == 'CUP_custom_nn':
        best_tr_error, epochs_done, final_validation_error, accuracy, crashed, final_MEE_error \
            = train_test_custom_nn(hparams, cfg, trial_name=trial_name, grid_search_name=grid_search_name,
                                   cv_num_plits=cfg[CFG.CV_NUM_SPLITS], cv_fold=cv_fold)
        results = {
            RES.epochs_done.name: epochs_done,
            RES.crashed.name: crashed,
        }
        if final_validation_error is not None:
            results[RES.mse_vl_last.name] = final_validation_error
        if best_tr_error is not None:
            results[RES.mse_tr_last.name] = best_tr_error
        if final_MEE_error is not None:
            results[RES.mee_vl_last.name] = final_MEE_error

    elif cfg[CFG.MODEL_TYPE] == 'monk_custom_nn':

        best_tr_error, epochs_done, final_validation_error, accuracy, crashed \
            = train_test_custom_nn(hparams, cfg, trial_name=trial_name, grid_search_name=grid_search_name)
        results = {
            RES.mse_vl_last.name: final_validation_error,
            RES.mse_tr_last.name: best_tr_error,
            RES.accuracy.name: accuracy,
            RES.epochs_done.name: epochs_done,
            RES.crashed.name: crashed,
        }
    elif cfg[CFG.MODEL_TYPE] == 'airfoil_tf':
        metrics_results, history = train_test_model_tf(hparams, cfg)
        results = {
            RES.loss_vl_last.name: history.history['val_loss'][-1],
            RES.mse_vl_last.name: history.history['val_mse'][-1],
            RES.loss_tr_last.name: metrics_results['loss'],
            RES.mse_tr_last.name: metrics_results['mse'],
            RES.epochs_done.name: len(history.history['loss'])
        }

    print('results: ', for_print(results))
    return results


def for_print(dct):
    dct = dct.copy()
    round_decimals_in_dict(dct)
    shorten_dict_keynames(dct)
    return dct


def shorten_dict_keynames(dct):
    for key in list(dct):
        if hasattr(key, 'name'):
            h_value = dct[key]
            del dct[key]
            dct[key.name] = h_value


def round_decimals_in_dict(dct):
    for key, value in dct.items():
        if isinstance(value, float):
            dct[key] = "{:0.3f}".format(value)


def append_trial_info_to_report(trial_dir, grid_search_name, dataset_filename, trial_name, hparams, results,
                                file_suffix=''):
    # todo fixme: some number apperaing incorrectly in excel sometimes
    # todo fixme: some trials go into numerical overflow errors while doing backprop: check with which params

    file_abs_path = os.path.join(trial_dir, grid_search_name + '-' + dataset_filename + file_suffix + '.csv')

    trial_info = {'trial': trial_name}
    trial_info.update(hparams)
    trial_info.update(results)

    print(f'appending {trial_name} results to {file_abs_path}')
    file_already_exists = os.path.exists(file_abs_path)
    write_mode = 'a' if file_already_exists else 'w'
    # shortening field names for csv readability
    # hp_names = [h.name if hasattr(h, 'name') else h for h in trial_info]
    shorten_dict_keynames(trial_info)

    try:
        with open(file_abs_path, write_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trial_info.keys(), delimiter=';')
            if not file_already_exists:
                writer.writeheader()
            writer.writerow(trial_info)

    except IOError:
        print("I/O error")


def train_test_custom_nn(hparams, cfg, trial_name='', grid_search_name='', cv_num_plits=5, cv_fold=1):
    root_dir = os.getcwd()
    filename = cfg[CFG.DATASET_FILENAME]
    file_path = os.path.join(root_dir, cfg[CFG.DATASET_DIR], filename)

    data_descr = 'CUP'
    if cfg[CFG.MODEL_TYPE] == 'monk_custom_nn':
        data = load_and_preprocess_monk_dataset(file_path)
        example_number = data.shape[0]
        train_ratio = 0.813
        split_id = int(np.round(example_number * train_ratio))
        print(f'doing {split_id} samples for training, and {example_number - split_id} for validation')
        training_split, validation_split = data[:split_id], data[split_id:]
    elif data_descr == 'CUP':
        training_split, validation_split = get_cup_dev_set_fold_splits(file_path, cv_num_plits, cv_fold)

    task = cfg[CFG.TASK_TYPE]
    mini_batch_size = hparams[HP.MB]
    activation = hparams[HP.ACTIV_FUN]  # 'sigmoid' # 'tanh'

    out_dim = cfg[CFG.OUT_DIM]
    input_dim = cfg[CFG.INPUT_DIM]
    num_hid_layers = hparams[HP.N_HID_LAYERS]
    num_units_per_hid_layer = hparams[HP.UNITS_PER_LAYER]
    net_shape = [input_dim]
    for _ in range(num_hid_layers):
        net_shape.append(num_units_per_hid_layer)
    net_shape.append(out_dim)  # ie net_shape = [17, 10, 1]

    lr = hparams[HP.LR]  # 0.05
    alpha_momentum = hparams[HP.MOMENTUM] # 0.12
    lambda_reg = hparams[HP.LAMBDA_L2]  # 0  # 0.001  # 0.005
    stopping_threshold = hparams[HP.STOPPING_THRESH]  # 0.00005
    max_epochs = hparams[HP.MAX_EPOCHS]  # 1500
    patience = hparams[HP.PATIENCE]  # 50
    error_fn = hparams[HP.ERROR_FN]  # 'MSE'
    optimizer = hparams[HP.OPTIMIZER]  # 'SGD constant lr'
    early_stopping_alg = hparams[HP.EARLY_STOP_ALG]  # 'MSE2_val'

    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_subdir = cur_time + '-' + trial_name
    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task=task, verbose=True, dir=trial_subdir, grid_search_dir=grid_search_name)

    test_net.load_training(training_split, out_dim, do_normalization=False)
    test_net.load_validation(validation_split, out_dim)

    from lib_models.utils import get_hyperparams_descr
    hyperparams_descr = get_hyperparams_descr(filename, str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer=optimizer)

    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print(f'initial validation_error = {test_net.validate_net()[0]:0.3f}')

    grid_search_dir = os.path.join(os.getcwd(), grid_search_name)
    try:
        best_tr_error, epochs_done, final_MEE_error = test_net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs,
                                                         stopping=early_stopping_alg, patience=patience,
                                                         verbose=False, hyperparams_for_plot=hyperparams_descr,
                                                             trial_name=trial_name, save_plot_to_file=False,
                                                             grid_search_dir_for_plots=grid_search_dir,
                                                             gridsearch_descr=grid_search_name)
        crashed = False
    except Exception as e:
        # todo: also fix/handle these cases (not actually Exceptions, just warnings, but the results
        # RuntimeWarning: overflow encountered in matmul: nextlayer_dEp_dOt = np.matmul(self.weights.transpose(), dEp_dOt)
        # RuntimeWarning: overflow encountered in double_scalars
        # RuntimeWarning: invalid value encountered in multiply
        print(f'batch_training failed with error: {e}')
        crashed = True
        best_tr_error, epochs_done = None, None

    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    final_validation_error, accuracy, vl_misc_rate = test_net.validate_net()
    if accuracy is not None:
        accuracy_info = f', final validation accuracy = {accuracy:0.3f}'
    else:
        accuracy_info = ''
    print(f'final validation_error = {final_validation_error:0.3f}{accuracy_info}')

    # todo: plot actual vs predicted (as accuracy and as MSE smoothing function)
    return best_tr_error, epochs_done, final_validation_error, accuracy, crashed, final_MEE_error


if __name__ == '__main__':

    for cv_fold in range(1, CUP_CUSTOM_NET_CFG[CFG.CV_NUM_SPLITS] + 1):
        do_grid_search(CUP_CUSTOM_NET_CFG, cv_fold=cv_fold)
    # do_grid_search(MONK1_CUSTOM_NET_CFG)
    # do_grid_search(MONK2_CUSTOM_NET_CFG)
    # do_grid_search(MONK3_CUSTOM_NET_CFG)