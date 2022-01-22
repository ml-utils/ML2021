import unittest
import numpy
import random
import os
import csv

import numpy as np

from preprocessing import get_cv_cut_points, get_cv_fold_split
from grid_search import trial_info_csv_writer_helper


class UnitTestsGridSearch(unittest.TestCase):

    def test_write_report_to_logdir(self):

        test_dirs = '.\\test_dir\\test_subdir'
        filename = 'some-text-file'
        file_abs_path = os.path.join(test_dirs, filename + '.txt')
        trial_info = {'trial': 'trial_name', 'trial_number': 0}

        # todo verify folder does not already exists
        self.assertFalse(os.path.isdir(test_dirs), msg=f'{os.path.abspath(test_dirs)}')

        trial_info_csv_writer_helper(file_abs_path, test_dirs, trial_info)

        # todo verify folder created
        self.assertTrue(os.path.isdir(test_dirs), msg=f'{os.path.abspath(test_dirs)}')

    def test_get_cv_cut_points(self):

        num_splits = 5
        dataset_size = 120
        which_fold = 1
        first_cut_point, second_cut_point = get_cv_cut_points(num_splits, which_fold, dataset_size)
        self.assertEqual(0, first_cut_point)
        self.assertEqual(24, second_cut_point)
        which_fold = 5
        first_cut_point, second_cut_point = get_cv_cut_points(num_splits, which_fold, dataset_size)
        self.assertEqual(96, first_cut_point)
        self.assertEqual(120, second_cut_point)

        # generate an array with 1200 elements
        column = [i+1 for i in range(120)]
        column_np = np.asarray(column)
        dummy_dataset = np.column_stack((column_np, column_np))
        self.assertEqual(120, len(dummy_dataset))

        print(f'num_splits: {num_splits}')
        for fold_num in range(1, 6):
            print(f'fold_num: {fold_num}')
            # get detaset splits, check that the size is the same as expected and sum as total dataset
            # todo check che la somma dei due split è uguale al totale
            # check che il dataset originale non è mutato
            vl_begin_point, vl_end_point = get_cv_cut_points(num_splits, fold_num, dataset_size)
            print(f'vl_begin_point, vl_end_point: {vl_begin_point}, {vl_end_point}')
            print(f'dummy_dataset.shape: {dummy_dataset.shape}')
            training_split, validation_split = get_cv_fold_split(dummy_dataset, vl_begin_point, vl_end_point)
            print(f'training_split.shape: {training_split.shape}, validation_split.shape: {validation_split.shape}')
            # todo, test the numpy shape of the arrays (num columns), also for more then 1 column
            self.assertEqual(96, len(training_split))
            self.assertEqual(24, len(validation_split))
            self.assertEqual(120, len(dummy_dataset))

            '''print('validation split:')
            print(validation_split)
            print('training_split :')
            print(training_split)'''

    def test_plot_learning_curve_to_img_file(self):
        from nn import NeuralNet

        plots_count = 400 #
        for current in range(plots_count):
            epochs = 500
            hyperparams_for_plot = None
            learning_task = 'classification'
            error_fn='MSE'
            net_dir = '.'
            validate_errors = [random.randint(0, 99) / 100 for epoch in range(epochs)]
            train_errors = [random.randint(0, 99) / 100 for epoch in range(epochs)]
            vl_misclassification_rates = [random.randint(0, 99) / 100 for epoch in range(epochs)]
            NeuralNet.plot_learning_curve_to_img_file(validate_errors, train_errors, vl_misclassification_rates, epochs,
                                    hyperparams_for_plot, learning_task, error_fn, net_dir)

            if not current % 20: #
                print(f'saved {current} image plots so far')

