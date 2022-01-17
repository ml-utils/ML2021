import unittest
import numpy as np
from numpy.testing import *
import os
import pandas as pd
from one_hot_encoder import *


class UnitTestsOneHotEncoder(unittest.TestCase):

    def test_one_hot_encode(self):
        a = np.array([1, 0, 3])
        b = one_hot_encode(a)
        assert_array_equal(b, np.array([[0, 1, 0, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 0, 1]]))

        root_dir = os.getcwd()
        file_path = os.path.join(root_dir, '..\\datasplitting\\assets\\monk\\monks-1.train')

        df = pd.read_csv(file_path, sep='\s')  #
        df.columns = ['class_label', 'head_shape', 'body_shape', 'is_smiling', 'holding', 'jacket_color', 'has_tie', 'id']
        df.drop(columns=['id'], inplace=True)
        wanted_columns_order = ['head_shape', 'body_shape', 'is_smiling', 'holding', 'jacket_color', 'has_tie', 'class_label']
        df = df.reindex(columns=wanted_columns_order)
        categorical_data = np.array(df)
        print('np shape before one hot: ', categorical_data.shape)
        print(categorical_data[:5])
        data_as_one_hot = one_hot_encode_multiple_cols(categorical_data, col_indexes_to_encode=[0,1,2,3,4,5],
                                                   cols_to_not_change=[6])
        print('np shape after one hot: ', data_as_one_hot.shape)
        print(data_as_one_hot[:5])

