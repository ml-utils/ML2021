import os

import numpy as np
import pandas as pd

MLCUP2021datatypes = ['%i'] + (['%f'] * 12)  # first col is integer, the others are floats
CUP_CFG = {'shortname': 'MLCUP2021', 'filename': 'ML-CUP21-TR.csv',
                                       'input_dim': 10, 'x_begin_col_idx': 1, 'x_end_col_idx': 10,
                                       'output_dim': 2, 'y_begin_col_idx': 11, 'y_end_col_idx': 12,
                                       'datatypes': MLCUP2021datatypes}


def get_cup_dev_set_fold_splits(filepath, cv_num_plits=5, which_fold=1):
    sep = ','
    dataset = np.loadtxt(filepath, delimiter=sep)  # , converters=converters, fmt=config['datatypes']
    dataset = remove_id_col(dataset, CUP_CFG)
    print('data loaded, shape: ', dataset.shape)
    print(f'dtype: {dataset.dtype}')

    # dev size = 1200
    dataset_size = dataset.shape[0]

    vl_begin_point, vl_end_point = get_cv_cut_points(cv_num_plits, which_fold, dataset_size)
    training_split, validation_split = get_cv_fold_split(dataset, vl_begin_point, vl_end_point)
    return training_split, validation_split


def get_cv_fold_split(dataset, vl_begin_point, vl_end_point):

    validation_split = dataset[vl_begin_point:vl_end_point]
    training_split = np.concatenate((dataset[0:vl_begin_point], dataset[vl_end_point:]), axis=0)

    return training_split, validation_split


def get_cv_cut_points(num_splits, which_fold, dataset_size):
    # todo: address case of splits with mod != 0 (so some patterns should be discarded)

    which_fold
    pattern_per_split = dataset_size // num_splits  # integer division, todo: support "imperfect" split points
    first_cut_point = (which_fold - 1) * pattern_per_split
    second_cut_point = which_fold * pattern_per_split
    return first_cut_point, second_cut_point


def load_and_preprocess_monk_dataset(filepath):
    print(f'loading {filepath}')
    import pandas as pd
    df = pd.read_csv(filepath, sep="\s+", engine='python')  #
    df.columns = ['class_label', 'head_shape', 'body_shape', 'is_smiling', 'holding', 'jacket_color', 'has_tie', 'id']
    df.drop(columns=['id'], inplace=True)
    wanted_columns_order = ['head_shape', 'body_shape', 'is_smiling', 'holding', 'jacket_color', 'has_tie', 'class_label']
    df = df.reindex(columns=wanted_columns_order)
    categorical_data = np.array(df)

    data = one_hot_encode_multiple_cols(categorical_data, col_indexes_to_encode=[0,1,2,3,4,5],
                                                   cols_to_not_change=[6])
    return data


def one_hot_encode(arr):
    b = np.zeros((arr.size, arr.max() + 1))
    b[np.arange(arr.size), arr] = 1

    idx_cols_all_zeros = np.argwhere(np.all(b[..., :] == 0, axis=0))
    b_no_zero_columns = np.delete(b, idx_cols_all_zeros, axis=1)

    return b_no_zero_columns


def one_hot_encode_multiple_cols(arr, col_indexes_to_encode=None, cols_to_not_change=None):
    _, num_cols = arr.shape

    # todo: remember mapping btw original categorical columns and one hot column, to allow decoding

    if col_indexes_to_encode is None:
        # encode all columns, create a sequence of all col idx
        col_indexes_to_encode = [x for x in range(num_cols)]

    encoded_ds = None
    for col_idx in range(num_cols):
        col_values = arr[:, col_idx]
        if col_idx in col_indexes_to_encode:
            col_one_hot_cols = one_hot_encode(col_values)
            if encoded_ds is None:
                encoded_ds = np.copy(col_one_hot_cols)
            else:
                encoded_ds = np.append(encoded_ds, col_one_hot_cols, axis=1)
        if col_idx in cols_to_not_change:
            encoded_ds = np.concatenate((encoded_ds, col_values[:, None]), axis=1)
        # print('col_idx: ', col_idx, ' encoded_ds.shape: ', encoded_ds.shape)
        # print(encoded_ds[:5])

    return encoded_ds


def split_cup_dataset():

    root_dir = os.getcwd()
    workdir = '.\\datasplitting\\assets\\ml-cup21-internal_splits\\'
    workdir_path = os.path.join(root_dir, workdir)
    input_file_name = 'ML-CUP21-TR.csv'
    input_file_path = os.path.join(workdir_path, input_file_name)
    # print(f'workdir_path dir: {workdir_path}, input_file_path: {input_file_path}')
    sep = ','

    config = CUP_CFG

    '''converters = {0: lambda s: int(s),
                  1: lambda s: float(s),
                  2: lambda s: float(s),
                  3: lambda s: float(s),
                  4: lambda s: float(s),
                  5: lambda s: float(s),
                  6: lambda s: float(s),
                  7: lambda s: float(s),
                  8: lambda s: float(s),
                  9: lambda s: float(s),
                  10: lambda s: float(s),
                  11: lambda s: float(s),
                  12: lambda s: float(s),
                  }  # {0: datestr2num}'''

    # pd.read_csv(filepath, sep=sep)
    dataset = np.loadtxt(input_file_path, delimiter=sep)  # , converters=converters, fmt=config['datatypes']
    print('data loaded, shape: ', dataset.shape)
    print(f'dtype: {dataset.dtype}')

    # check num columns, num patterns

    print('dataset head before shuffling: ')
    print(dataset[:2])
    # shuffle
    np.random.shuffle(dataset)
    print('dataset head after shuffling: ')
    print(dataset[:2])
    print('data loaded, shape: ', dataset.shape)

    # test_set_ratio = 0.2
    # nb per il training set cercare num pattern che sia multiplo .. di minibatch tipici
    # 1477 totali
    # 1200 dev (0.8125) 277 ts (0.1875 perc)
    # cv folds: 240 vl, 960 tr (0.20 perc folds)
    dev_split_idx = 1200

    dev_split_filename = 'dev_split.csv'
    dev_split_file_path = os.path.join(workdir_path, dev_split_filename)
    print(f'saving dataset to {dev_split_file_path}')
    np.savetxt(dev_split_file_path, dataset[:dev_split_idx], fmt=config['datatypes'], delimiter=sep)

    test_split_filename = 'test_split.csv'
    test_split_file_path = os.path.join(workdir_path, test_split_filename)
    print(f'saving dataset to {test_split_file_path}')
    np.savetxt(test_split_file_path, dataset[dev_split_idx:], fmt=config['datatypes'], delimiter=sep)


def remove_id_col(dataset, config):
    if config['shortname'] == 'MLCUP2021':
        ids = dataset[:, 0]
        no_ids = dataset[:, config['x_begin_col_idx']:(config['y_end_col_idx'] + 1)]
    return no_ids


if __name__ == '__main__':
    split_cup_dataset()