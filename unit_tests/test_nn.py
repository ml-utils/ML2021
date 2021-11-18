import unittest

import numpy

from datasplitting.datasplits import DataSplits
from nn import neuralnet


class UnitTestsNeuralNet(unittest.TestCase):
    def test_pattern_update(self):
        nn1 = neuralnet(units=(50, ), activation='tanh')
        config = DataSplits.get_dataset_configuration('airfoil')
        _, inX, outy = DataSplits.load_the_dataset(config, otherpath='../datasplitting/assets/airfoil/')
        for sample in inX:
            idx = 0
            label = outy[idx]
            nn1.pattern_update(sample, label)

    def test_xor_numeric(self):
        nn1 = neuralnet(units=(4,), activation='tanh')
        # samples: x1, x2, y =  000 011 101 110
        samples = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
        for sample in samples:
            input = numpy.array(sample[0:2])
            label = numpy.array(sample[2])
            nn1.pattern_update(input, label)


if __name__ == '__main__':
    unittest.main()
