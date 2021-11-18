import unittest

import numpy

from datasplitting.datasplits import DataSplits
from nn import neuralnet


class UnitTestsNeuralNet(unittest.TestCase):
    def test_init(self):
        nn1 = neuralnet(units=(50, ), activation='tanh')
        config = DataSplits.get_dataset_configuration('airfoil')
        _, inX, outy = DataSplits.load_the_dataset(config, otherpath='../datasplitting/assets/airfoil/')
        for sample in inX:
            idx = 0
            label = outy[idx]
            nn1.pattern_update(sample, label)


if __name__ == '__main__':
    unittest.main()
