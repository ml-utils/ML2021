import os

import numpy as np

MLCUP2021datatypes = ['%i'] + (['%f'] * 12)  # first col is integer, the others are floats
CUP_CFG = {'shortname': 'MLCUP2021', 'filename': 'ML-CUP21-TR.csv', 'sep': ',',
                                       'input_dim': 10, 'x_begin_col_idx': 1, 'x_end_col_idx': 10,
                                       'output_dim': 2, 'y_begin_col_idx': 11, 'y_end_col_idx': 12,
                                       'datatypes': MLCUP2021datatypes}
CUP_DATASETS_DIR = '.\\datasplitting\\assets\\ml-cup21-internal_splits\\'




def get_cupdata_split_fixed_size(filepath, training_size=1200, vl_size=277):
    print(f'get_cupdata_split_fixed_size training_size: {training_size}, filepath: {filepath}')

    dataset = get_cup_dataset_from_file(filepath)
    training_split, validation_split = get_cv_fold_split(dataset, training_size, training_size+vl_size)
    return training_split, validation_split


def get_cup_dev_set_fold_splits(filepath, cv_num_plits=3, which_fold=1):
    print(f'get_cup_dev_set_fold_splits cv_num_plits: {cv_num_plits}, which_fold: {which_fold}, filepath: {filepath}')

    dataset = get_cup_dataset_from_file(filepath)
    # dev size = 1200
    dataset_size = dataset.shape[0]

    vl_begin_point, vl_end_point = get_cv_cut_points(cv_num_plits, which_fold, dataset_size)
    training_split, validation_split = get_cv_fold_split(dataset, vl_begin_point, vl_end_point)
    return training_split, validation_split


def get_cv_fold_split(dataset, vl_begin_point, vl_end_point):

    validation_split = dataset[vl_begin_point:vl_end_point]
    training_split = np.concatenate((dataset[0:vl_begin_point], dataset[vl_end_point:]), axis=0)

    return training_split, validation_split


def get_tr_and_vl_splits_by_size(dataset, vl_size, shuffle=False):

    if shuffle:
        np.random.shuffle(dataset)

    validation_split = dataset[:vl_size]
    training_split = dataset[vl_size:]

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


def get_cup_dataset_from_file(filepath, has_id_col_to_remove=True):
    dataset = np.loadtxt(filepath, delimiter=CUP_CFG['sep'])  # , converters=converters, fmt=config['datatypes']
    if has_id_col_to_remove:
        dataset = remove_id_col(dataset, CUP_CFG)
    print(f'data loaded, shape: {dataset.shape}, dtype: {dataset.dtype}')

    return dataset


def get_cup_dataset_from_file2(filename='ML-CUP21-TR.csv', has_id_col_to_remove=False):
    root_dir = os.getcwd()
    # workdir_path = os.path.join(root_dir, CUP_DATASETS_DIR)
    input_file_path = os.path.join(CUP_DATASETS_DIR, filename)
    return get_cup_dataset_from_file(input_file_path, remove_id_col)


def shuffle_dataset(dataset):
    print('dataset head before shuffling: ')
    print(dataset[:2])
    # shuffle
    np.random.shuffle(dataset)
    print('dataset head after shuffling: ')
    print(dataset[:2])
    print('data loaded, shape: ', dataset.shape)


def save_shuffled_dataset(filename_suffix):
    dataset = get_cup_dataset_from_file2(filename='ML-CUP21-TR.csv')
    shuffle_dataset(dataset)
    # TODO: save to file
    out_filename = 'ML-CUP21-' + filename_suffix + '.csv'
    out_file_path = os.path.join(CUP_DATASETS_DIR, out_filename)
    print(f'saving shuffled dataset to {out_file_path}')
    np.savetxt(out_file_path, dataset, fmt=MLCUP2021datatypes, delimiter=CUP_CFG['sep'])


def split_cup_dataset(dev_split_idx=1200, pt1_name='dev_split', pt2_name='test_split'):
    dataset = get_cup_dataset_from_file(filename='ML-CUP21-TR.csv', workdir_path=CUP_DATASETS_DIR)
    # check num columns, num patterns

    shuffle_dataset(dataset)

    # test_set_ratio = 0.2
    # nb per il training set cercare num pattern che sia multiplo .. di minibatch tipici
    # 1477 totali
    # 1200 dev (0.8125) 277 ts (0.1875 perc)
    # cv folds: 240 vl, 960 tr (0.20 perc folds)

    dev_split_filename = pt1_name + '.csv'
    dev_split_file_path = os.path.join(CUP_DATASETS_DIR, dev_split_filename)
    print(f'saving dataset to {dev_split_file_path}')
    np.savetxt(dev_split_file_path, dataset[:dev_split_idx], fmt=MLCUP2021datatypes, delimiter=CUP_CFG['sep'])

    test_split_filename = pt2_name + '.csv'
    test_split_file_path = os.path.join(CUP_DATASETS_DIR, test_split_filename)
    print(f'saving dataset to {test_split_file_path}')
    np.savetxt(test_split_file_path, dataset[dev_split_idx:], fmt=MLCUP2021datatypes, delimiter=CUP_CFG['sep'])


def remove_id_col(dataset, config):
    if config['shortname'] == 'MLCUP2021':
        ids = dataset[:, 0]
        no_ids = dataset[:, config['x_begin_col_idx']:(config['y_end_col_idx'] + 1)]
    return no_ids


if __name__ == '__main__':
    # split_cup_dataset(dev_split_idx=1400, pt1_name='retradev_split', pt2_name='test_split')
    save_shuffled_dataset(filename_suffix='shuffled-retraining')
