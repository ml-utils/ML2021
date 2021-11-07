import numpy
from numpy import loadtxt, savetxt
import random
import os


class DataSplits:
    data_folder = './assets/'
    export_folder = './exports/'

    @staticmethod
    def get_dataset_configuration(dataset_name='MLCUP2021'):
        configurations = dict()
        MLCUP2021datatypes = ['%i'] + (['%f'] * 12)  # first col is integer, the others are floats
        configurations['MLCUP2021'] = {'shortname': 'MLCUP2021', 'filename': 'ML-CUP21-TR.csv',
                                       'input_dim': 10, 'x_begin_col_idx': 1, 'x_end_col_idx': 10,
                                       'output_dim': 2, 'y_begin_col_idx': 11, 'y_end_col_idx': 12,
                                       'datatypes': MLCUP2021datatypes,
                                       'file_extension': '.csv', 'csv_delimiter': ',', 'folder': './assets/'}

        airfoil_datatypes = ['%i'] + (['%f'] * 5)
        configurations['airfoil'] = {'shortname': 'airfoil', 'filename': 'airfoil_self_noise.dat.csv',
                                     'input_dim': 5, 'x_begin_col_idx': 0, 'x_end_col_idx': 4,
                                     'output_dim': 1, 'y_begin_col_idx': 5, 'y_end_col_idx': 5,
                                     'datatypes': airfoil_datatypes,
                                     'file_extension': '.csv', 'csv_delimiter': '\t', 'folder': './assets/airfoil/'}

        return configurations[dataset_name]

    @staticmethod
    def load_the_dataset(config, otherpath=''):
        print('loading data..')
        folder = config['folder']
        if otherpath:
            folder = otherpath
        filepath = os.path.join(folder, config['filename'])
        filepath = os.path.abspath(filepath)
        if os.path.isfile(filepath):
            dataset = loadtxt(filepath, delimiter=config['csv_delimiter'])  # converters = {0: datestr2num}
            print('data loaded, shape: ', dataset.shape)
        else:
            msg = 'error, file not found: ' + filepath
            raise NameError(msg)

        return DataSplits.slice_dataset(dataset, config)

    @staticmethod
    def slice_dataset(dataset, config):
        # split ndarray into: ids list, input (X) and output (y) variables

        int_ids = None
        if config['shortname'] == 'MLCUP2021':
            ids = dataset[:, 0]
            int_ids = []
            for idx_idx, idx in enumerate(ids):
                int_ids.append(int(idx))

        X = dataset[:, config['x_begin_col_idx']:(config['x_end_col_idx'] + 1)]
        y = dataset[:, config['y_begin_col_idx']:(config['y_end_col_idx'] + 1)]

        if int_ids is None:
            datasize = X.shape[0]
            int_ids = [x + 1 for x in range(datasize)]  # list(range(datasize))

        return int_ids, X, y

    @staticmethod
    def concatenate_dataset(X, y, config, int_ids=None):
        if config['filename'] == 'ML-CUP21-TR.csv':
            dataset_lenght = X.shape[0]
            ids_1d = numpy.array(int_ids)  # print('type(int_ids[0]): ', type(int_ids[0]))
            ids = ids_1d.reshape((dataset_lenght, 1))
            # print('type(ids[0][0]): ', type(ids[0][0]))
            # print('shape: ', ids.shape, 'shape: ', X.shape, 'shape: ', y.shape, )
            dataset = numpy.concatenate((ids, X, y), axis=1, dtype=object)
            # print('type(dataset[0][0]): ', type(dataset[0][0]))
        else:
            dataset = numpy.concatenate((X, y), axis=1, dtype=object)

        return dataset

    @staticmethod
    def remove_random_sample(input_ids, inX, outy, config):
        # get a random from 0 to len of remanining ids
        random_idx = random.randrange(len(input_ids))

        return DataSplits.remove_sample(random_idx, input_ids, inX, outy, config)

    @staticmethod
    def remove_sample(remove_idx, input_ids, inX, outy, config):
        sample_id = input_ids.pop(remove_idx)
        sample_x = inX[remove_idx]
        inX = numpy.delete(inX, remove_idx)
        sample_y = outy[remove_idx]
        outy = numpy.delete(outy, remove_idx)
        return sample_id, sample_x, sample_y

    @staticmethod
    def get_random_k_folds(k, inX, outy, config, input_ids=None):
        # given k as number of splits
        # generate k splits of the given data, with randomization
        # make sure format of splits is compatible with scikit-learn formats
        datasize = inX.shape[0]
        DataSplits.check_k_folds_value(k, datasize)

        if not input_ids:
            input_ids = [x+1 for x in range(datasize)]  # list(range(datasize))

        folds = []
        for fold_idx in range(k):
            current_fold_input_ids = []
            # remaining_samples = len(input_ids)
            fold_size = DataSplits.calculate_fold_size(datasize, k, fold_idx)
            x_array_shape = (fold_size, config['input_dim'])
            y_array_shape = (fold_size, config['output_dim'])
            print('x_array_shape: ', x_array_shape, ' y_array_shape: ', y_array_shape)
            current_fold_X = numpy.empty(x_array_shape)
            current_fold_y = numpy.empty(y_array_shape)
            for i in range(fold_size):
                # fix: pool of ids, copy x and y values in new array without removing

                sample_id, sample_x, sample_y = DataSplits.remove_random_sample(input_ids, inX, outy, config)
                current_fold_input_ids.append(sample_id)
                current_fold_X[i] = sample_x  # numpy.append(current_fold_X, sample_x)
                current_fold_y[i] = sample_y  # numpy.append(current_fold_y, sample_y)
            current_fold = {'ids': current_fold_input_ids, 'X': current_fold_X, 'y': current_fold_y}
            # tuple(current_fold_input_ids, current_fold_X, current_fold_y)
            folds.append(current_fold)
        return folds

    @staticmethod
    def merge_folds(folds, config):
        rows_axis = 0
        cols_axis = 1
        folds_as_ndarrays = []
        for fold in folds:
            folds_as_ndarrays.append(DataSplits.concatenate_dataset(fold['X'], fold['y'], config, fold['ids']))
        dataset = numpy.concatenate(folds_as_ndarrays, axis=rows_axis)  # , dtype=object
        return dataset

    @staticmethod
    def check_k_folds_value(k, datasize):
        if k > datasize:
            msg = 'error, k > datasize', k, datasize
            raise NameError(msg)

    @staticmethod
    def calculate_fold_size(datasetsize, folds_count, fold_idx):
        min_samples_per_fold = datasetsize // folds_count
        fold_size = min_samples_per_fold
        rest_samples_count = datasetsize % folds_count
        if fold_idx < rest_samples_count:
            fold_size += 1
        return fold_size

    @staticmethod
    def save_splits_to_files(folds, config, filename_prefix=''):
        print('saving data..')
        folder = os.path.join(DataSplits.export_folder, config['folder'])
        folder = os.path.abspath(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not filename_prefix:
            filename_prefix = config['shortname']
        for fold_idx, fold in enumerate(folds):
            filename_suffix = '_fold'  # _fold1, _fold2, etc.
            filename = filename_prefix + filename_suffix + str(fold_idx) + config['file_extension']
            filepath = os.path.join(folder, filename)
            filepath = os.path.abspath(filepath)

            fold_dataset = DataSplits.concatenate_dataset(fold['X'], fold['y'], config, fold['ids'])
            savetxt(filepath, fold_dataset, fmt=config['datatypes'], delimiter=config['csv_delimiter'])
            print('saved fold to ', filename)
        print('data saved to files.')
        return folder
