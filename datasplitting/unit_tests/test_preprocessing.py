import unittest

import numpy

from preprocessing import Preprocessing


class UnitTestsPreprocessing(unittest.TestCase):
    def test_remove_random_sample(self):
        input_ids, inX, outy = Preprocessing.load_the_dataset(folder='../assets/')
        element_9 = {'id': input_ids[8], 'x': inX[8], 'y': outy[8]}
        element_10 = {'id': input_ids[9], 'x': inX[9], 'y': outy[9]}
        element_11 = {'id': input_ids[10], 'x': inX[10], 'y': outy[10]}

        initial_lenght = len(input_ids)
        self.assertEqual(initial_lenght, len(inX))
        self.assertEqual(initial_lenght, len(outy))

        print(input_ids[9])
        print(element_10['id'])
        Preprocessing.remove_sample(9, input_ids, inX, outy)
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
        input_ids, inX, outy = Preprocessing.load_the_dataset(folder='../assets/')
        initial_lenght = len(input_ids)
        self.assertGreater(initial_lenght, 0)
        self.assertEqual(initial_lenght, len(inX))
        self.assertEqual(initial_lenght, len(outy))

        k = 5
        folds = Preprocessing.get_random_k_folds(k, input_ids, inX, outy)
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
        input_ids, inX, outy = Preprocessing.load_the_dataset(folder='../assets/')

        for idx in input_ids:
            idx = idx - 1
            self.assertIsInstance(inX[idx], numpy.ndarray)
            for value in inX[idx]:
                self.assertIsInstance(value, numpy.float64)
            self.assertIsInstance(outy[idx], numpy.ndarray)
            for value in outy[idx]:
                self.assertIsInstance(value, numpy.float64)


if __name__ == '__main__':
    unittest.main()
