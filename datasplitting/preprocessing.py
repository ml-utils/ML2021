import numpy
from numpy import loadtxt
import random
import os


class Preprocessing:
    input_dim = 10
    data_folder = './assets/'

    @staticmethod
    def load_the_dataset(filename='', folder=''):
        print('loading data..')
        if not filename:
            filename = 'ML-CUP21-TR.csv'
        if not folder:
            folder = Preprocessing.data_folder
        filepath = os.path.join(folder, filename)
        filepath = os.path.abspath(filepath)
        if os.path.isfile(filepath):
            dataset = loadtxt(filepath, delimiter=',')  # converters = {0: datestr2num}
            print('data loaded.')
        else:
            msg = 'error, file not found: ' + filepath
            raise NameError(msg)

        # split into input (X) and output (y) variables

        ids = dataset[:, 0]
        int_ids = []
        print(type(ids))
        for idx_idx, idx in enumerate(ids):
            int_ids.append(int(idx))

        X = dataset[:, 1:(Preprocessing.input_dim + 1)]
        y = dataset[:, 11:13]
        return int_ids, X, y

    @staticmethod
    def remove_random_sample(input_ids, inX, outy):
        # get a random from 0 to len of remanining ids
        random_idx = random.randrange(len(input_ids))

        return Preprocessing.remove_sample(random_idx, input_ids, inX, outy)

    @staticmethod
    def remove_sample(remove_idx, input_ids, inX, outy):
        sample_id = input_ids.pop(remove_idx)
        sample_x = inX[remove_idx]
        inX = numpy.delete(inX, remove_idx)
        sample_y = outy[remove_idx]
        outy = numpy.delete(outy, remove_idx)
        return sample_id, sample_x, sample_y

    @staticmethod
    def get_random_k_folds(k, input_ids, inX, outy):
        # given k as number of splits
        # generate k splits of the given data, with randomization
        # make sure format of splits is compatible with scikit-learn formats

        datasize = len(input_ids)
        if k > datasize:
            print('error, k > datasize', k, datasize)

        folds = []
        for fold_idx in range(k):
            current_fold_input_ids = []
            remaining_samples = len(input_ids)
            fold_size = Preprocessing.calculate_fold_size(datasize, k, fold_idx)
            x_array_shape = (fold_size, 10)
            y_array_shape = (fold_size, 2)
            print('x_array_shape: ', x_array_shape, ' y_array_shape: ', y_array_shape)
            current_fold_X = numpy.empty(x_array_shape)
            current_fold_y = numpy.empty(y_array_shape)
            for i in range(fold_size):
                # fix: pool of ids, copy x and y values in new array without removing

                sample_id, sample_x, sample_y = Preprocessing.remove_random_sample(input_ids, inX, outy)
                current_fold_input_ids.append(sample_id)
                current_fold_X[i] = sample_x  # numpy.append(current_fold_X, sample_x)
                current_fold_y[i] = sample_y  # numpy.append(current_fold_y, sample_y)
            current_fold = {'ids': current_fold_input_ids, 'X': current_fold_X, 'y': current_fold_y}
            # tuple(current_fold_input_ids, current_fold_X, current_fold_y)
            folds.append(current_fold)
        return folds

    @staticmethod
    def calculate_fold_size(datasetsize, folds_count, fold_idx):
        min_samples_per_fold = datasetsize // folds_count
        fold_size = min_samples_per_fold
        rest_samples_count = datasetsize % folds_count
        if fold_idx < rest_samples_count:
            fold_size += 1
        return fold_size

