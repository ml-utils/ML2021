import numpy as np


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

