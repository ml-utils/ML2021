import unittest
from nn import neuralnet


class UnitTestsNeuralNet(unittest.TestCase):
    def test_init(self):
        nn1 = neuralnet(activation='tanh')


if __name__ == '__main__':
    unittest.main()