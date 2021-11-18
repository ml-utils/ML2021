import numpy as np

def sigmoid(x):
    return 1/(1 - np.exp(-x))

def sigmprime(y):
    return sigmoid(y)*(1 - sigmoid(y))

def tanh(a, b):
    return lambda x : a*np.tanh(b*x)

def tanhprime(a, b):
    return lambda y : a*b + b/a*y**2

class layer:

    def __init__(self, fan_in, fan_out, activation, strtin_range, act_parameters=(1, 1)):
        self.fan_in = fan_in
        self.fan_out = fan_out
        rng = np.random.default_rng()
        self.weights = strtin_range*(2*rng.random([fan_out, fan_in+1])-1) # was: strtin_range*(2*np.random.Generator.random([fan_out, fan_in+1])-1)
        self.weights /= fan_in

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.derivative = sigmprime

        elif activation == 'tanh':
            self.activation = tanh(*act_parameters)
            self.derivative = tanhprime(*act_parameters)

        self.role = 'hidden'
        self.delta_weights = np.zeros([fan_out, fan_in + 1])
        self.delta_weights_old = np.zeros([fan_out, fan_in + 1])
        self.latest_out = np.empty(fan_out)
        self.latest_net = np.empty(fan_out)
        self.latest_in = np.empty(fan_in + 1)
        self.latest_in[-1] = 1

    def switch_role(self):
        self.role = 'output'

    def evaluate(self, input):
        self.latest_in[:-1] = input
        self.latest_net = np.matmul(self.weights, input)
        self.latest_out = self.activation(self.latest_net)
        return self.latest_out

    def calc_local_gradient(self, error_signal):
        # if role == output unit:
        # error_signal == o_j - d_j
        # if role == hidden unit k:
        # error_signal = sum_(j=1)^fan_out w_lj^(k+1)*delta_l

        p_gradient = error_signal*self.derivative(self.latest_net)
        self.delta_weights += np.outer(p_gradient, self.latest_in)
        output_error_signal = np.matmul(p_gradient, self.weights)

        return output_error_signal

    def update_weights(self, eta, alpha, lamda):

        self.weights += eta*self.delta_weights + alpha*self.delta_weights_old - lamda*self.weights
        self.delta_weights_old = self.delta_weights
        self.delta_weights = np.zeros(self.delta_weights.shape)


class neuralnet:

    def __init__(self, units=(2, 2), activation='tanh', eta = 0.01, alpha = 0.5, lamda = 0.2, mb=20, **kwargs):
        self.layers = []

        for i in range(len(units)-1):
            lay = layer(units[i], units[i+1], activation, strtin_range=0.1)
            self.layers.append(lay)

        self.eta = eta
        self.alpha = alpha
        self.lamda = lamda
        self.mb = mb

    def evaluate(self, input):

        entry = input
        for lay in self.layers:
          entry = lay.evaluate(entry)

        return entry

    def pattern_update(self, input, label):

        output = self.evaluate(input)
        error_signal = output - label



