import unittest
import numpy
import random

from datasplitting.datasplits import DataSplits
from nn import NeuralNet


class UnitTestsGridSearch(unittest.TestCase):

    def test_plot_learning_curve_to_img_file(self):

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

