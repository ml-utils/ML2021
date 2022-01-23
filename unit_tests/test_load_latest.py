import numpy as np
import os
from nn import NeuralNet

current_dir = os.getcwd()
example_path = os.path.join(current_dir, '20220121-194451')

net = NeuralNet.load_latest(example_path)

for layer in net.layers:

    print(layer.weights)

print(net.hyperparameters)