import unittest
import numpy
from preprocessing import Preprocessing
import os


class UnitTestsPreprocessing(unittest.TestCase):
    def test_remove_random_sample(self):
        config = Preprocessing.get_dataset_configuration('MLCUP2021')
        input_ids, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/')
        element_9 = {'id': input_ids[8], 'x': inX[8], 'y': outy[8]}
        element_10 = {'id': input_ids[9], 'x': inX[9], 'y': outy[9]}
        element_11 = {'id': input_ids[10], 'x': inX[10], 'y': outy[10]}

        initial_lenght = len(input_ids)
        self.assertEqual(initial_lenght, len(inX))
        self.assertEqual(initial_lenght, len(outy))

        print(input_ids[9])
        print(element_10['id'])
        Preprocessing.remove_sample(9, input_ids, inX, outy, config)
        print(input_ids[9])
        print(element_11['id'])

        self.assertEqual(initial_lenght-1, len(input_ids))

        # check element 9, 10, 11 id and values (x, y)
        # remove element 10
        # check 9, 11 present, 10 missing
        numpy.testing.assert_array_equal(inX[9], element_10['x'])
        numpy.testing.assert_array_equal(outy[9], element_10['y'])
        self.assertEqual(input_ids[9], element_11['id'])

    def test_k_folds_split(self):
        config = Preprocessing.get_dataset_configuration('MLCUP2021')
        input_ids, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/')
        initial_lenght = len(input_ids)
        self.assertGreater(initial_lenght, 0)
        self.assertEqual(initial_lenght, len(inX))
        self.assertEqual(initial_lenght, len(outy))

        k = 5
        folds = Preprocessing.get_random_k_folds(k, inX, outy, config, input_ids)
        expected_min_fold_size = initial_lenght//k
        expected_max_fold_size = expected_min_fold_size
        if initial_lenght % k:
            expected_max_fold_size += 1
        # self.assertEqual(len(inX), 0)
        self.assertEqual(len(folds[0]['ids']), expected_max_fold_size)
        self.assertEqual(len(folds[0]['X']), expected_max_fold_size)
        self.assertEqual(len(folds[0]['y']), expected_max_fold_size)
        self.assertEqual(len(folds[k-1]['ids']), expected_min_fold_size)
        self.assertEqual(len(folds[k-1]['X']), expected_min_fold_size)
        self.assertEqual(len(folds[k-1]['y']), expected_min_fold_size)

        total_folds_samples = 0
        for i in range(k):
            total_folds_samples += len(folds[i]['X'])
        self.assertEqual(total_folds_samples, initial_lenght)

    def test_load_the_dataset(self):
        config = Preprocessing.get_dataset_configuration('MLCUP2021')
        input_ids, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/')

        for idx in input_ids:
            idx = idx - 1
            self.assertIsInstance(inX[idx], numpy.ndarray)
            for value in inX[idx]:
                self.assertIsInstance(value, numpy.float64)
            self.assertIsInstance(outy[idx], numpy.ndarray)
            for value in outy[idx]:
                self.assertIsInstance(value, numpy.float64)

        config = Preprocessing.get_dataset_configuration('airfoil')
        _, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/airfoil/')

    def test_concatenate_dataset(self):
        config = Preprocessing.get_dataset_configuration('MLCUP2021')
        input_ids, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/')

        dataset = Preprocessing.concatenate_dataset(inX, outy, config, input_ids)
        actual_shape = dataset.shape
        expected_shape = (1477, 13)
        self.assertEqual(actual_shape, expected_shape)
        self.assertEqual(type(dataset[0][0]), int)

    def test_save_splits_to_files(self):
        config = Preprocessing.get_dataset_configuration('MLCUP2021')
        input_ids, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/')
        k = 5
        folds = Preprocessing.get_random_k_folds(k, inX, outy, config, input_ids)
        folder = Preprocessing.save_splits_to_files(folds, config)

        # todo, improve with glob, list files only
        dir_contents_onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        # dir_contents = os.listdir(folder)
        expected_file_count = k
        self.assertEqual(len(dir_contents_onlyfiles), expected_file_count)

        config = Preprocessing.get_dataset_configuration('airfoil')
        _, inX, outy = Preprocessing.load_the_dataset(config, otherpath='../assets/airfoil/')
        k = 5
        folds = Preprocessing.get_random_k_folds(k, inX, outy, config)
        folder = Preprocessing.save_splits_to_files(folds, config)

        dir_contents = os.listdir(folder)
        expected_file_count = k
        self.assertEqual(len(dir_contents), expected_file_count)


if __name__ == '__main__':
    unittest.main()
