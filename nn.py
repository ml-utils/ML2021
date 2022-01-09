import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from numpy.random import default_rng
from datetime import datetime
from shutil import rmtree, copytree
import os

# Activation functions, each followed by their respective derivatives;
# derivatives are expressed as function of the resulting outputs to simplify computations

def linear(x):
    return x

def linearprime(y):
    return 1

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmprime(y):
    return y*(1-y)


def tanh(a, b):
    return lambda x: a*np.tanh(b*x)


def tanhprime(a, b):
    return lambda y: a*b - b/a*y**2

# "layer" class:
# implements a single layer of an MLP


class Layer:

    def __init__(self, fan_in, fan_out, activation, strtin_range=1, act_parameters=(1.7159, 2/3), **kwargs):

        # initialization inputs:
        # fan_in (int):          number of inputs
        # fan_out (int):         number of outputs
        # activation (string):   name of desired activation function
        # strtin_range (float):    starting possible range of the weights (may implement fancier stuff l8r)
        # act_parameters (list): additional parameters for the activation function, only used for some

        self.fan_in = fan_in
        self.fan_out = fan_out

        if 'weights' not in kwargs.keys():
            # weights are implemented as a (fan_out) x (fan_in + 1) matrix using the convention of writing the bias as
            # the last column of the matrix
            temp_rng = default_rng()
            self.weights = strtin_range*(2*temp_rng.random((fan_out, fan_in+1)) - 1)
            self.weights *= np.sqrt(12/fan_in)/2
            self.delta_weights = np.zeros([fan_out, fan_in + 1])

        else:
            self.weights = kwargs['weights']
            self.delta_weights = kwargs['dweights']

        self.delta_weights_old = np.zeros([fan_out, fan_in + 1])

        if activation == 'linear':
            self.activation = linear
            self.derivative = linearprime

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.derivative = sigmprime

        elif activation == 'tanh':
            self.activation = tanh(*act_parameters)
            self.derivative = tanhprime(*act_parameters)

        self.activation = np.vectorize(self.activation)
        self.derivative = np.vectorize(self.derivative)

        self.role = 'hidden'  # layer is initialized as hidden unit by default, use switch_role() to change to output
        self.latest_out = np.empty(fan_out)
        self.latest_net = np.empty(fan_out)
        self.latest_in = np.empty(fan_in + 1)
        self.latest_in[-1] = 1  # last column of latest_in is always set to 1 to implement bias-as-matrix-column

    @classmethod
    def load_file(cls, base_path, layer_number, activation, act_parameters=(1, 1)):

        weight_path = os.path.join(base_path, 'layer_{}_weights.csv'.format(layer_number))
        delta_path = os.path.join(base_path, 'layer_{}_deltas.csv'.format(layer_number))
        weights = np.loadtxt(weight_path)
        deltas = np.loadtxt(delta_path)
        assert(deltas.shape == weights.shape)
        fan_out = weights.shape[0]
        fan_in = weights.shape[1] - 1

        layer = Layer(fan_in, fan_out, activation, None, act_parameters, weights=weights, deltas=deltas)

        return layer

    # switch_role() sets role of layer as output unit
    def switch_role(self, task):
        self.role = 'output'
        if task == 'regression':
            self.activation = np.vectorize(linear)
            self.derivative = np.vectorize(linearprime)

    # evaluate(): evaluates input using current weights and stores input, net and output as class attributes
    # for later training
    def evaluate(self, entry):
        self.latest_in[:-1] = entry
        self.latest_net = np.matmul(self.weights, self.latest_in)
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

        p_gradient = error_signal*self.derivative(self.latest_out)
        self.delta_weights = self.delta_weights - np.outer(p_gradient, self.latest_in)
        output_error_signal = np.matmul(self.weights.transpose(), p_gradient)

        return output_error_signal[:-1]

    # update_weights: update weights using both momentum and L2 regularization, updates delta_weights_old
    # for momentum and zeros delta_weights for successive training
    def update_weights(self, eta, alpha, lamda):

        self.delta_weights_old = eta*self.delta_weights + alpha*self.delta_weights_old
        self.weights += self.delta_weights_old - lamda*self.weights
        self.delta_weights = np.zeros(self.delta_weights.shape)


class NeuralNet:

    def __init__(self, activation, units=(2, 2), eta=0.1, alpha=0.1, lamda=0.01, mb=20,
                 task='classification', placeholder=False, **kwargs):

        # activation (string): name of activation function used
        #                      (TODO: add way to use different functions for different layers)
        # units (list): list of number of units in each layer
        # eta (float): learning rate
        # alpha (float): momentum parameter
        # lamda (float): regularization parameter
        # mb (int): number of patterns in each mini-batch
        # task (string): task the NN is meant to perform (classification or regression)
        # **kwargs: idk it'll be useful for something

        self.layers = []

        if not placeholder:
            self.fan_in = units[0]
            self.fan_out = units[-1]

            for i in range(len(units)-1):

                lay = Layer(units[i], units[i+1], activation)
                self.layers.append(lay)

            self.layers[-1].switch_role(task)

        self.eta = eta*mb  # learning rate
        self.alpha = alpha*mb  # momentum parameter
        self.lamda = lamda*mb  # regularization parameter
        self.mb = mb # number of patterns in each mini-batch
        self.batch_SE = 0 # squared error on latest mini-batch
        self.epoch_SE = 0 # squared error on latest epoch

        self.validation_set = None  # placeholder for internal validation set
        self.training_set = None    # placeholder for internal training set

        # placeholders for vectors used for variable normalization x'_i = (x_i - <x_i>)/std(x_i)
        self.shift_vector = None    # placeholder for internal validation set
        self.scale_vector = None    # placeholder for internal validation set

        base_dir = os.getcwd()
        if 'dir' in kwargs.keys():
            self.net_dir = os.path.join(base_dir, kwargs['dir'])

        else:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.net_dir = os.path.join(base_dir, cur_time)

        # save the starting conditions, if random:
        self.savestate(0)

    @classmethod
    def load_net(cls, path, activation, eta=0.1, alpha=0.1, lamda=0.01, mb=20, task='classification', **kwargs):

        net = NeuralNet(activation, eta=eta, alpha=alpha, lamda=lamda, mb=mb,
                        task=task, placeholder=True, **kwargs)

        all_files = os.listdir(path)
        layer_files = [file for file in all_files if (file[:6] == 'layer_' and file[-4:] == '.csv')]
        if (len(layer_files) % 2):
            ValueError('Mismatch in number of layer files in folder {}'.format(path))
        num_of_layers = len(layer_files)/2

        for i in range(num_of_layers):
            lay = Layer.load_file(path, i, activation)
            net.layers.append(lay)

        net.layers[-1].switch_role(task)
        net.fan_in = net.layers[0].fan_in
        net.fan_out = net.layers[-1].fan_out

        return net

    # stores training set inside of neural net
    def load_training(self, training_set, out=1):

        train_examples = training_set.shape[0]
        self.eta /= train_examples
        self.lamda /= train_examples
        self.alpha /= train_examples
        inp = training_set.shape[1] - out
        if inp != self.fan_in:
            raise ValueError('Number of input variables in training set doesn\'t match input units!')

        if out != self.fan_out:
            raise ValueError('Number of output variables in training set doesn\'t match output units!')

        # calculates average and standard deviation for each input and output variable over the training set
        # these are then used to facilitate training by normalizing each variable so that it has 0 average and std 1

        self.shift_vector = np.average(training_set, axis=0)
        self.scale_vector = np.std(training_set, axis=0)

        # stores normalized training set

        self.training_set = (training_set - self.shift_vector)/self.scale_vector

        # applies normalization to validation set if self.load_validation() was called before self.load_training()
        if self.validation_set is not None:
            self.validation_set -= self.shift_vector
            self.validation_set /= self.scale_vector

    # stores validation set inside of neural net
    def load_validation(self, validation_set, out=1):

        inp = validation_set.shape[1] - out
        if inp != self.fan_in:
            raise ValueError('Number of input variables in validation set doesn\'t match input units!')

        if out != self.fan_out:
            raise ValueError('Number of output variables in validation set doesn\'t match output units!')

        # stores plain validation set if self.load_training hasn't been called already
        if self.shift_vector is None:
            self.validation_set = validation_set

        # stores normalized validation set if self.load_training has already been called
        else:
            self.validation_set = (validation_set - self.shift_vector)/self.scale_vector

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

    # internal_evaluate: evaluates **normalized** input and returns **normalized** output; only use this on
    # internally-stored data such as training and validation sets; for external data such as assessment sets or
    # deployment data use self.evaluate()

    def internal_evaluate(self, entry):

        for lay in self.layers:
            entry = lay.evaluate(entry)

        return entry

    # evaluate: evaluates non-normalized (original format) inputs and returns non-normalized outputs
    def evaluate(self, entry):

        entry = (entry - self.shift_vector[:self.fan_in])/self.scale_vector[:self.fan_in]
        output = self.internal_evaluate(entry)
        output = output*self.scale_vector[self.fan_in:] + self.shift_vector[self.fan_in:]

        return output

    # pattern_update: evaluate input using current weights, then back-propagates error through the entire network
    def pattern_update(self, entry, label):

        output = self.internal_evaluate(entry)
        error_signal = output - label
        if type(error_signal) is np.ndarray:
            self.epoch_SE += (error_signal**2).sum()

        else:
            self.epoch_SE += error_signal**2

        for i, layer in reversed(list(enumerate(self.layers))):
            error_signal = layer.calc_local_gradient(error_signal)

    # update weights using previously-calculated weight deltas
    def update_weights(self):

        for layer in self.layers:
            layer.update_weights(self.eta, self.alpha, self.lamda)

    # splits training data into batches; returns an iterable
    # NOTE: if self.mb isn't a multiple of the number of training examples the last few will be left out of training
    def batch_split(self):

        for start_id in range(0, self.training_set.shape[0] - self.mb + 1, self.mb):
            batch = self.training_set[start_id:start_id+self.mb]
            yield batch

    # trains neural net for fixed number of epochs using mini-batch method; also saves MSE for each epoch on file
    def batch_training(self, stopping='epochs', threshold=300, max_epochs=500): #TODO: implement actual stopping conditions

        # stopping (string):        name of stopping condition to be used during training
        # threshold (int or float): numerical threshold at which to stop training;
        #                           exact meaning depends on stopping condition
        # max_epochs (int):         maximum number of epochs per training

        # list of implemented stopping conditions:

        # 'epochs': stop training after *threshold* number of epochs
        #           if epoch = threshold: break
        # 'MSE':    stop training once the relative change in the training error drops below *threshold*
        #           if (error[epoch] - error[epoch-1])/error[epoch] < threshold: break

        train_errors = np.empty(max_epochs)     # support array to store training errors
        validate_errors = np.empty(max_epochs)  # support array to store validation errors
        batch_rng = default_rng() # rng object to shuffle training set

        # number of examples used in each epoch (lower than total number of TRAIN examples, see batch_split()
        example_number = self.training_set.shape[0] - self.training_set.shape[0] % self.mb
        for epoch in range(max_epochs):
            self.epoch_SE = 0
            batch_rng.shuffle(self.training_set) # shuffle training set
            for batch in self.batch_split():

                for example in batch:
                    x = example[:self.fan_in]
                    y = example[-self.fan_out:]
                    self.pattern_update(x, y)

                self.update_weights()

            train_errors[epoch] = self.epoch_SE/example_number
            validate_errors[epoch] = self.validate_net()
            if not epoch % 100: # prints training status on console every 100 epochs
                self.savestate(epoch)
                print('epoch {} done (error = {})'.format(epoch, validate_errors[epoch]))

            # check for stopping conditions
            if stopping == 'epochs':
                if epoch == threshold:
                    break

            if stopping == 'MSE':
                relative_change = (train_errors[epoch] - train_errors[epoch-1])/train_errors[epoch-1]
                if relative_change < threshold:
                    break

        train_error_path = os.path.join(self.net_dir, 'training_errors.csv')
        np.savetxt(train_error_path, train_errors[:epoch+1], delimiter=',')  # saves history of training errors on file

        validate_error_path = os.path.join(self.net_dir, 'validation_errors.csv')
        np.savetxt(validate_error_path, validate_errors[:epoch+1], delimiter=',')  # saves history of training errors on file

        plt.plot(validate_errors[:epoch+1], label='validation errors')
        plt.plot(train_errors[:epoch+1], label='training errors')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend()

        error_graph_path = os.path.join(self.net_dir, 'errors.png')
        plt.savefig(error_graph_path)

    # calculates MSE on the validation set with current weights
    def validate_net(self):

        error = 0
        for example in self.validation_set:
            x = example[:self.fan_in]
            y = example[-self.fan_out:]
            if type(y) is np.ndarray:
                error += ((y - self.internal_evaluate(x))**2).sum()

            else:
                error += (y - self.internal_evaluate(x))**2

        error /= self.validation_set.shape[0]
        return error

if __name__ == '__main__':
    # trains basic neural network on the airfoil dataset
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'datasplitting\\assets\\airfoil\\airfoil_self_noise.dat.csv')
    data = np.loadtxt(data_dir)
    train_ratio = 0.7
    rng = default_rng()
    rng.shuffle(data)
    example_number = data.shape[0]

    net_shape = [5, 8, 1]
    split_id = int(np.round(example_number*train_ratio))

    test_net = NeuralNet('tanh', net_shape, eta=0.01, alpha=0.12, lamda=0.005, task='regression')

    test_net.load_training(data[:split_id], 1)
    test_net.load_validation(data[split_id:], 1)

    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print('initial validation_error = {}'.format(test_net.validate_net()))
    test_net.batch_training()
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time-start_time))
    print('final validation_error = {}'.format(test_net.validate_net()))
