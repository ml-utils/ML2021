import csv
import json
import os
import numpy as np

current_dir = os.getcwd()
folder_name = '20220122-123947-gs-fold-3-CUP_custom_nn'

general_header = ['trial', 'epochs_done', 'crashed', 'mse_vl_last', 'mse_tr_last', 'mee_vl_last',
                  'UNITS_PER_LAYER', 'N_HID_LAYERS', 'LR', 'MOMENTUM', 'LAMBDA_L2', 'OPTIMIZER', 'MB', 'ACTIV_FUN',
                  'STOPPING_THRESH', 'PATIENCE', 'MAX_EPOCHS', 'ERROR_FN', 'EARLY_STOP_ALG']

summary_header = ['trial', 'mse_vl_last-median', 'mse_vl_last-sd', 'mse_tr_last-median', 'mse_tr_last-sd',
                  'mee_vl_last-median', 'mee_vl_last-sd', 'epochs_done-median', 'epochs_done-sd', 'trials',
                  'crashes', 'UNITS_PER_LAYER', 'N_HID_LAYERS', 'LR', 'MOMENTUM', 'LAMBDA_L2', 'OPTIMIZER', 'MB',
                  'ACTIV_FUN', 'STOPPING_THRESH', 'PATIENCE', 'MAX_EPOCHS', 'ERROR_FN', 'EARLY_STOP_ALG']

path = os.path.join(current_dir, folder_name)

repetition_summary_path = os.path.join(current_dir, folder_name + '-dev_split.csv.csv')
aggregated_summary_path = os.path.join(current_dir, folder_name + '-dev_split.csv-aggregated.csv')

repetition_summary_file = open(repetition_summary_path, 'w')
aggregated_summary_file = open(aggregated_summary_path, 'w')

repetition_writer = csv.writer(repetition_summary_file, delimiter=';', lineterminator='\n')
aggregated_writer = csv.writer(aggregated_summary_file, delimiter=';', lineterminator='\n')

repetition_writer.writerow(general_header)
aggregated_writer.writerow(summary_header)

all_folders = os.listdir(path)


def trial_split(folder_list):
    for configuration in range(0, len(folder_list), 5):
        yield folder_list[configuration:configuration+5]



for j, trials in enumerate(list(trial_split(all_folders))):
    print(trials)
    mse_vl = np.empty(5)
    mse_tr = np.empty(5)
    mee_vl = np.empty(5)
    epochs_done = np.empty(5)
    for i, repetition in enumerate(list(trials)):
        single_summary = os.path.join(path, repetition, 'net_summary.json')
        layers_folder = os.path.join(path, repetition, 'latest')

        with open(single_summary, 'r') as infile:
            summary = json.load(infile)

        trial_name = repetition[16:]

        validation_errors = np.asarray(summary['validation errors'])
        training_errors = np.asarray(summary['training errors'])
        mee_vl[i] = summary['final MEE'][0]
        epochs_done[i] = len(validation_errors)
        best_epoch = np.argmin(validation_errors)
        mse_tr[i] = training_errors[best_epoch]
        mse_vl[i] = validation_errors[best_epoch]

        layer_files = os.listdir(layers_folder)
        hidden_layers = len(layer_files)/2 -1

        layer_path = os.path.join(layers_folder, layer_files[0])
        first_layer = np.loadtxt(layer_path, delimiter=';')

        units_per_layer = first_layer.shape[0]

        hyperparameters = summary['hyperparameters']

        '''trial;epochs_done;crashed;mse_vl_last;mse_tr_last;mee_vl_last;'
        'UNITS_PER_LAYER;N_HID_LAYERS;LR;MOMENTUM;LAMBDA_L2;OPTIMIZER;MB;ACTIV_FUN;'
        'STOPPING_THRESH;PATIENCE;MAX_EPOCHS;ERROR_FN;EARLY_STOP_ALG'''

        repetition_entry = [trial_name, epochs_done[i], 'false', mse_vl[i], mse_tr[i], mee_vl[i], units_per_layer,
                            hidden_layers, hyperparameters['eta'], hyperparameters['alpha'], hyperparameters['lambda'],
                            'SGD constant lr', hyperparameters['mb'], hyperparameters['hidden activation'],
                            '0.001', '100', '2000', 'MSE', 'MSE2_val']

        repetition_writer.writerow(repetition_entry)

    '''trial;mse_vl_last-median;mse_vl_last-sd;mse_tr_last-median;mse_tr_last-sd;'
    'mee_vl_last-median;mee_vl_last-sd;epochs_done-median;epochs_done-sd;trials;'
    'crashes;UNITS_PER_LAYER;N_HID_LAYERS;LR;MOMENTUM;LAMBDA_L2;OPTIMIZER;MB;ACTIV_FUN;STOPPING_THRESH;'
    'PATIENCE;MAX_EPOCHS;ERROR_FN;EARLY_STOP_ALG'''

    session_name = 'session-{}'.format(1+j)

    aggregated_entry = [session_name, np.median(mse_vl), np.std(mse_vl), np.median(mse_tr), np.std(mse_tr),
                        np.median(mee_vl), np.std(mee_vl), np.median(epochs_done), np.std(epochs_done), 5,
                        0, units_per_layer, hidden_layers, hyperparameters['eta'], hyperparameters['alpha'],
                        hyperparameters['lambda'], 'SGD constant lr', hyperparameters['mb'],
                        hyperparameters['hidden activation'], '0.001', '100', '2000', 'MSE', 'MSE2_val']

    aggregated_writer.writerow(aggregated_entry)

aggregated_summary_file.close()
repetition_summary_file.close()
