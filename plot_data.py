import gc
import glob
import json
import os

import numpy as np
from guppy import hpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LAST_TRIAL_PLOTTED_IDX = 'last_trial_ploted_idx'
LAST_TRIAL_PLOTTED_FOLDERNAME = 'last_trial_ploted_foldername'
NUM_TRIALS_IN_FILE = 'num_trials_in_file'
PLOT_DATA = 'plot_data'
LEARNING_TASK = 'learning_task'
SESSION_NUM = 'session_num'
HYPERPARAMS_FOR_PLOT = 'hyperparams_for_plot'
ERROR_FUNC = 'error_func'
LAST_EPOCH = 'last_epoc'
VAL_ERRORS = 'val_errors'
TR_ERRORS = 'tr_errors'
VL_MISCL_RATES = 'vl_miscl_rates'


def append_learning_curve_plot_data_to_file(validation_errors, train_errors, vl_misclassification_rates, epoch,
                                            hyperparams_for_plot, learning_task, error_func, save_to_dir,
                                            session_num=None, trial_name='',
                                            gridsearch_descr=''):
    '''
    this is to be used during grid search loops, to avoid that crashes in opening pyplot stop the grid search.
    The plots of each trial of the grid search can be generated later using this output file.
    :return:
    '''
    # todo: check trial_name is in the form 11.2 for session 11, repeat 2
    # todo: check the save dir passed is that of the grid search, so all trials write/append to same file
    # write all the plot data in one json file, but store the last plot successfully generated, and resume from there

    # convert ndarrays to list
    if isinstance(validation_errors, np.ndarray):
        validation_errors = validation_errors.tolist()
        train_errors = train_errors.tolist()
        vl_misclassification_rates = vl_misclassification_rates.tolist()

    trial_plot_data = {SESSION_NUM: session_num, HYPERPARAMS_FOR_PLOT: hyperparams_for_plot,
                       ERROR_FUNC: error_func,
                       LAST_EPOCH: epoch, VAL_ERRORS: validation_errors, TR_ERRORS: train_errors,
                       VL_MISCL_RATES: vl_misclassification_rates}

    json_filename = gridsearch_descr + '-' + trial_name + '-plotdata.json'
    print(f'save_to_dir: {save_to_dir}, trial_name: {trial_name}')
    trial_subdir = save_to_dir + trial_name
    json_filepath = os.path.join(save_to_dir, trial_subdir, json_filename)
    if not os.path.exists(json_filepath):
        grid_search_plot_data = {NUM_TRIALS_IN_FILE: 0,
                                 LEARNING_TASK: learning_task, PLOT_DATA: []}
    else:
        with open(json_filepath) as json_file:
            grid_search_plot_data = json.load(json_file)

    grid_search_plot_data[PLOT_DATA].append(trial_plot_data)
    grid_search_plot_data['num_trials_in_file'] += 1

    print(f'appending plot data to {json_filepath}')
    with open(json_filepath, 'w') as outfile:
        json.dump(grid_search_plot_data, outfile)


def generate_plots_from_grid_search_output(grid_search_dir):

    resume_info_filename = 'plot_resume_info.json'

    trial_subdirs = [trial_dir for trial_dir in os.listdir(grid_search_dir)
                     if os.path.isdir(os.path.join(grid_search_dir, trial_dir))]

    resume_info_filepath = os.path.join(grid_search_dir, resume_info_filename)
    if os.path.exists(resume_info_filepath):
        with open(resume_info_filepath) as json_file:
            resume_info = json.load(json_file)
    else:
        resume_info = {LAST_TRIAL_PLOTTED_IDX: -1, LAST_TRIAL_PLOTTED_FOLDERNAME: ''}

    # check idx and forlder names match from last run
    if resume_info['last_trial_ploted_idx'] >= 0:
        if resume_info['last_trial_ploted_foldername'] != trial_subdirs[resume_info['last_trial_ploted_idx']]:
            print(f'Error: trying to resume, but there is a mismatch btw folder names and indexes: '
                  f'{resume_info[LAST_TRIAL_PLOTTED_IDX]}-th folder expected to be '
                  f'{resume_info[LAST_TRIAL_PLOTTED_FOLDERNAME]}, but is '
                  f'{trial_subdirs[resume_info[LAST_TRIAL_PLOTTED_IDX]]}')
            return

    for idx, trial_dir in enumerate(trial_subdirs):
        if resume_info['last_trial_ploted_idx'] >= idx:
            pass
        else:
            # each trial folder should contain a json file with its plot data
            # in each trial subfolder, expects one json file, ending with plotdata.json
            trial_dir_full_path = os.path.join(grid_search_dir, trial_dir)
            path_pattern = trial_dir_full_path + r'\*plotdata.json'
            print(f'trial_dir_full_path pattern: {path_pattern}')
            plotDataFilenamesList = glob.glob(path_pattern)
            print(f'glob found: {plotDataFilenamesList}, actual dir contents of {trial_dir_full_path}:')
            print(os.listdir(trial_dir_full_path))
            if len(plotDataFilenamesList) == 0:
                print(f'Warning: no plot data files in {trial_dir_full_path}')
                pass
            if len(plotDataFilenamesList) > 1:
                print(f'Warning: too many ({len(plotDataFilenamesList)}) plot data files in {trial_dir_full_path} , '
                      f'plotting none. Found plot files are: {plotDataFilenamesList}')
                pass

            plot_data_file_full_path = os.path.join(grid_search_dir, plotDataFilenamesList[0])
            generate_plot_from_single_trial_output_file(plot_data_file_full_path)

            resume_info['last_trial_ploted_idx'] = idx
            resume_info['last_trial_ploted_foldername'] = trial_dir
            with open(resume_info_filepath, 'w') as outfile:
                json.dump(resume_info, outfile)


def generate_plot_from_single_trial_output_file(plot_data_file_full_path, save_to_dir, session_num, trial_num,
                                    ylim=None, xlim=None):

    with open(plot_data_file_full_path) as json_file:
        plot_data = json.load(json_file)

    if (plot_data['num_trials_in_file'] > 1):
        print(f'Warning: more than one trial plot data in {plot_data_file_full_path} , plotting just the 1st')

    trial_plot_data = plot_data[PLOT_DATA][0]
    trial_name = f'{session_num}.{trial_num}'
    plot_learning_curve_to_img_file(trial_plot_data[VAL_ERRORS], trial_plot_data[TR_ERRORS],
                                    trial_plot_data[VL_MISCL_RATES], trial_plot_data[LAST_EPOCH],
                                    trial_plot_data[HYPERPARAMS_FOR_PLOT], plot_data[LEARNING_TASK],
                                    trial_plot_data[ERROR_FUNC], save_to_dir, trial_name=trial_name,
                                    ylim=ylim, xlim=xlim)


def generate_all_plots_for_subdirs(main_dir):
    return 0
    # loop all dirs
    # in each looks for a file ending with plotdata.json
    # (parse the fold session trial info from dir name or file name)
    # (also prefix with main dir timestamp)
    # generate the plotfile, with suffix the fold session trial descr
    # (the file is saved in the main dir, if names are unique)


def plot_learning_curve_to_img_file(validate_errors, train_errors, vl_misclassification_rates, epoch,
                                    hyperparams_for_plot, learning_task, error_func, net_dir, trial_name='',
                                    ylim=None, xlim=None):
    # todo, workaround: run this in a separathe thread/process to try fix the crash after 350 plots

    try:
        fig, ax = plt.subplots(1)
        # todo: explain: why specify [:epoch+1]
        ax.plot(validate_errors[:epoch+1], '--', label='validation errors')
        ax.plot(train_errors[:epoch+1], '-', label='training errors')
        ax.plot(vl_misclassification_rates[:epoch + 1], label='vl misclassification rates')
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        plt.xlabel('epoch')

        y_axis_label = 'MSE / miscl rate' if learning_task == 'classification' else error_func
        plt.ylabel(y_axis_label)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='none', label=hyperparams_for_plot))
        ax.legend(handles=handles)
    except Exception as e:
        print(f'Unable to create a new plot {e}')
    except SystemExit as e:
        print(f'Unable to create a new plot, SystemExit: {e}')
    except BaseException as e:
        print(f'Unable to create a new plot, BaseException: {e}')

    # todo: plot of accuracy metric (misclassification rate)
    # todo: evaluate on monk test set files

    filename = 'errors-' + trial_name + '.png'
    error_graph_path = os.path.join(net_dir, filename)

    plt.savefig(error_graph_path)
    fig.clear()
    plt.close(fig) # plt.close('all')
    gc.collect()
    # print(f'Currently there are {plt.get_fignums()} pyplot figures.')
    # print(hpy().heap())


if __name__ == '__main__':
    import sys

    # print(f'args: {sys.argv}')

    plot_data_file_dir = sys.argv[1]
    plot_data_filename = sys.argv[2]
    # optional
    session_num = sys.argv[3] if sys.argv[3] else 'x'
    trial_num = sys.argv[4] if sys.argv[4] else 'x'

    plot_data_file_full_path = os.path.join(plot_data_file_dir, plot_data_filename)
    save_to_dir = plot_data_file_dir

    if not os.path.exists(plot_data_file_dir):
        print(f'Dir does not exist: {plot_data_file_dir} ')
        sys.exit()

    if not os.path.exists(plot_data_file_full_path):
        print(f'File does not exist: {plot_data_file_dir} ')
        sys.exit()

    ylim = (0.08, 0.3)
    xlim = None  # (0, 2000)
    generate_plot_from_single_trial_output_file(plot_data_file_full_path, save_to_dir, session_num, trial_num,
                                                ylim=ylim, xlim=xlim)
