import os
import unittest
import numpy
import random
import shutil, tempfile

from datasplitting.datasplits import DataSplits
from nn import NeuralNet
from plot_data import *

tmp_grid_search_dir_name = '20220120-161153-monk_custom_nn'
trial_1_1_subfolder_name = os.path.join(tmp_grid_search_dir_name, '20220120-161153-trial-1.1')
trial_1_2_subfolder_name = os.path.join(tmp_grid_search_dir_name, '20220120-161205-trial-1.2')


class UnitTestsPlotData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''self.assertFalse(os.path.exists(tmp_grid_search_dir))
        self.assertFalse(os.path.exists(trial_1_1_subfolder))
        self.assertFalse(os.path.exists(trial_1_2_subfolder))'''
        # crate a grid search subfolder
        '''os.mkdir(tmp_grid_search_dir)
        os.mkdir(trial_1_1_subfolder)
        os.mkdir(trial_1_2_subfolder)'''

    @classmethod
    def tearDownClass(cls):
        # clean up tmp files and dirs
        '''os.remove(trial_1_1_subfolder)
        os.remove(trial_1_2_subfolder)
        os.remove(tmp_grid_search_dir)'''
        '''self.assertFalse(os.path.exists(tmp_grid_search_dir))
        self.assertFalse(os.path.exists(trial_1_1_subfolder))
        self.assertFalse(os.path.exists(trial_1_2_subfolder))'''

    def setUp(self):
        # Create a temporary directory
        '''tempfile.TemporaryDirectory()
        self.tmp_grid_search_dir = tempfile.mkdtemp()  #
        self.trial_1_1_subfolder = tempfile.mkdtemp(dir=self.tmp_grid_search_dir)
        self.trial_1_2_subfolder = tempfile.mkdtemp(dir=self.tmp_grid_search_dir)'''

    def tearDown(self):
        # Remove the directory after the test
        '''shutil.rmtree(self.trial_1_1_subfolder)
        shutil.rmtree(self.trial_1_2_subfolder)
        shutil.rmtree(self.tmp_grid_search_dir)'''

    def test_append_learning_curve_plot_data_to_file(self):
        self.trial_1_1_subfolder = trial_1_1_subfolder_name

        session_num = 1
        trial_name = '1.1'
        learning_task = 'classification'
        hyperparams_for_plot = ''
        epoch = 0
        error_func = 'MSE'
        train_errors = np.array([0.1])
        validation_errors = np.array([0.2])
        vl_misclassification_rates = np.array([0.5])
        savedir = self.trial_1_1_subfolder
        gridsearch_descr = 'gridsearch_timestamp' + '-' + 'monk1-nn'

        json_filename = gridsearch_descr + '-' + trial_name + '-plotdata.json'
        json_filepath = os.path.join(savedir, json_filename)

        self.assertFalse(os.path.exists(json_filepath))

        if not os.path.exists(savedir): os.mkdir(savedir)
        append_learning_curve_plot_data_to_file(validation_errors, train_errors, vl_misclassification_rates, epoch,
                                                hyperparams_for_plot, learning_task, error_func, savedir,
                                                session_num=session_num, trial_name=trial_name,
                                                gridsearch_descr=gridsearch_descr)

        self.assertTrue(os.path.exists(json_filepath))

        # load file, check num trials is 1, load/check the other data
        with open(json_filepath) as json_file:
            grid_search_plot_data = json.load(json_file)

        self.assertEqual(1, grid_search_plot_data[NUM_TRIALS_IN_FILE])

        '''
        grid_search_plot_data = {'num_trials_in_file': 0, 'last_plot_successfully_done': -1,
                                 'learning_task': learning_task, 'plot_data': []}

        trial_plot_data = {'session_num': session_num, 'hyperparams_for_plot': hyperparams_for_plot,
                           'error_func': error_func,
                           'last_epoc': epoch, 'val_errors': validation_errors, 'tr_errors': train_errors,
                           'vl_miscl_rates': vl_misclassification_rates}
        grid_search_plot_data['plot_data'].append(trial_plot_data)
        '''
        # delete file after test
        try:
            os.remove(json_filepath)
            os.mkdir(savedir)
        except Exception as e:
            print(e)

    def test_generate_plots_from_grid_search_output2(self):
        return 0

    def test_generate_plots_from_grid_search_output(self):
        self.trial_1_1_subfolder = trial_1_1_subfolder_name
        session_num = 1
        trial_name = '1.1'
        learning_task = 'classification'
        hyperparams_for_plot = ''
        epoch = 4
        error_func = 'MSE'
        train_errors = np.array([0.3686534479668946, 0.36265433344481, 0.3573498647941035, 0.351855206271252])
        validation_errors = np.array([0.131850569198107, 0.13375515528932863, 0.13574274217707402, 0.1378116131103841])
        vl_misclassification_rates = np.array([0.13043478260869565, 0.13043478260869565, 0.13043478260869565, 0.13043478260869565])
        savedir = self.trial_1_1_subfolder
        gridsearch_descr = 'gridsearch_timestamp' + '-' + 'monk1-nn'

        json_filename = gridsearch_descr + '-plotdata.json'
        json_filepath = os.path.join(savedir, json_filename)

        '''self.assertTrue(os.path.exists(self.tmp_grid_search_dir))
        self.assertTrue(os.path.exists(self.trial_1_1_subfolder))
        self.assertTrue(os.path.exists(self.trial_1_2_subfolder))'''
        # create two trial subfolder
        # each with a json file

        self.tmp_grid_search_dir = tmp_grid_search_dir_name
        if not os.path.exists(self.tmp_grid_search_dir): os.mkdir(self.tmp_grid_search_dir)
        append_learning_curve_plot_data_to_file(validation_errors, train_errors, vl_misclassification_rates, epoch,
                                                hyperparams_for_plot, learning_task, error_func, self.tmp_grid_search_dir,
                                                session_num=session_num, trial_name=trial_name,
                                                gridsearch_descr=gridsearch_descr)
        generate_plots_from_grid_search_output(self.tmp_grid_search_dir)
        try:
            os.remove(self.tmp_grid_search_dir)
        except Exception as e:
            print(e)
        # then also test resuming

    def test_generate_plot_from_single_trial_output_file(self):

        plot_data_file_dir = r'..\20220121-143417-gs-monk_custom_nn\20220121-143418-trial-1.1'
        plot_data_filename = r'20220121-143417-gs-monk_custom_nn--plotdata.json'
        plot_data_file_full_path = os.path.join(plot_data_file_dir, plot_data_filename)
        save_to_dir = plot_data_file_dir
        session_num = 1
        trial_num = 1
        generate_plot_from_single_trial_output_file(plot_data_file_full_path, save_to_dir, session_num, trial_num)
