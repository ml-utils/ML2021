import numpy as np
from time import sleep
from datetime import datetime
from shutil import rmtree, copytree
import os

# Activation functions, each followed by their respective derivatives;
# derivatives are expressed as function of the resulting outputs to simplify computations


def sigmoid(x):
    return 1/(1 - np.exp(-x))


def sigmprime(y):
    return sigmoid(y)*(1 - sigmoid(y))


def tanh(a, b):
    return lambda x: a*np.tanh(b*x)


def tanhprime(a, b):
    return lambda y: a*b + b/a*y**2

# "layer" class:
# implements a single layer of an MLP


class Layer:

    def __init__(self, fan_in, fan_out, activation, strtin_range, act_parameters=(1, 1)):

        # initialization inputs:
        # fan_in (int):          number of inputs
        # fan_out (int):         number of outputs
        # activation (string):   name of desired activation function
        # strtin_range (int):    starting possible range of the weights (may implement fancier stuff l8r)
        # act_parameters (list): additional parameters for the activation function, only used for some

        self.fan_in = fan_in
        self.fan_out = fan_out

        # weights are implemented as a (fan_out) x (fan_in + 1) matrix using the convention of writing the bias as
        # the last column of the matrix
        rng = np.random.default_rng()
        self.weights = strtin_range*(2*rng.random((fan_out, fan_in+1)) - 1)
        self.weights /= np.sqrt(fan_in)

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.derivative = sigmprime

        elif activation == 'tanh':
            self.activation = tanh(*act_parameters)
            self.derivative = tanhprime(*act_parameters)

        self.role = 'hidden'  # layer is initialized as hidden unit by default, use switch_role() to change to output
        self.delta_weights = np.zeros([fan_out, fan_in + 1])
        self.delta_weights_old = np.zeros([fan_out, fan_in + 1])
        self.latest_out = np.empty(fan_out)
        self.latest_net = np.empty(fan_out)
        self.latest_in = np.empty(fan_in + 1)
        self.latest_in[-1] = 1  # last column of latest_in is always set to 1 to implement bias-as-matrix-column

    # switch_role() sets role of layer as output unit
    def switch_role(self):
        self.role = 'output'

    # evaluate(): evaluates input using current weights and stores input, net and output as class attributes
    # for later training
    def evaluate(self, entry):
        self.latest_in[:-1] = entry
        self.latest_net = np.matmul(self.weights, entry)
        self.latest_out = self.activation(self.latest_net)
        return self.latest_out

    # calc_local_gradient: calculates the layer's local gradient for the latest output according to
    # the back-propagating error signal from the next unit, calculates actual gradient for weights update,
    # then returns error signal for the previous unit.
    def calc_local_gradient(self, error_signal):
        # if role == output unit:
        # error_signal == o_j - d_j
        # (this assumes error function to be L2 norm; different error functions would require different definitions
        # of "error signal" and potentially other tweaks)

        # if role == hidden unit k:
        # error_signal = sum_(j=1)^fan_out w_lj^(k+1)*delta_l

        p_gradient = error_signal*self.derivative(self.latest_net)
        self.delta_weights += np.outer(p_gradient, self.latest_in)
        output_error_signal = np.matmul(p_gradient, self.weights)

        return output_error_signal

    # update_weights: update weights using both momentum and L2 regularization, updates delta_weights_old
    # for momentum and zeros delta_weights for successive training
    def update_weights(self, eta, alpha, lamda):

        self.weights += eta*self.delta_weights + alpha*self.delta_weights_old - lamda*self.weights
        self.delta_weights_old = self.delta_weights
        self.delta_weights = np.zeros(self.delta_weights.shape)


class NeuralNet:

    def __init__(self, activation, units=(2, 2), eta=0.01, alpha=0.5, lamda=0.2, mb=20, **kwargs):

        # activation (string): name of activation function used
        #                      (TODO: add way to use different functions for different layers)
        # units (list): list of number of units in each layer
        # eta (float): learning rate
        # alpha (float): momentum parameter
        # lamda (float): regularization parameter
        # mb (int): number of patterns in each mini-batch
        # **kwargs: idk it'll be useful for something


        self.layers = []

        for i in range(len(units)-1):
            lay = Layer(units[i], units[i+1], activation, 0.1)
            self.layers.append(lay)

        self.eta = eta
        self.alpha = alpha
        self.lamda = lamda
        self.mb = mb
        self.latest_SE = 0  # squared error on latest training example
        root_dir = os.getcwd()
        if 'dir' in kwargs.keys():
            self.net_dir = os.path.join(root_dir, kwargs['dir'])

        else:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.net_dir = os.path.join(root_dir, cur_time)

    # savestate: save current state of neural net to folder
    # each folder is named after the training epoch it represents and contains one deltas.csv and one weights.csv
    # for each layer in the net; if no epoch number is provided it instead names the folder based on current datetime
    def savestate(self, epoch=-1):

        if epoch >= 0:
            cur_dir = os.path.join(self.net_dir, "epoch_{}".format(epoch))

        else:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            cur_dir = os.path.join(self.net_dir, cur_time)

        latest_dir = os.path.join(self.net_dir, 'latest')
        rmtree(latest_dir, ignore_errors=True)
        os.makedirs(cur_dir, exist_ok=True)


        for i, layer in enumerate(self.layers):
            weight_path = os.path.join(cur_dir, 'layer_{}_weights.csv'.format(i))
            delta_path = os.path.join(cur_dir, 'layer_{}_deltas.csv'.format(i))
            np.savetxt(weight_path, layer.weights, delimiter=',')
            np.savetxt(delta_path, layer.delta_weights, delimiter=',')

        copytree(cur_dir, latest_dir)




    # evaluate: evaluate input using current weights
    def evaluate(self, entry):

        for lay in self.layers:
            entry = lay.evaluate(entry)

        return entry

    # pattern_update: evaluate input using current weights, then back-propagates error through the entire network
    def pattern_update(self, entry, label):

        output = self.evaluate(entry)
        error_signal = output - label
        self.latest_SE = error_signal**2
        for layer in reversed(self.layers):

            error_signal = layer.calc_local_gradient(error_signal)

if __name__ == '__main__':

    test_net = NeuralNet('sigmoid', dir='test')
    test_net.savestate()
    sleep(90)
    test_net.savestate()