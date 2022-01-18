import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from numpy.random import default_rng
from datetime import datetime
from shutil import rmtree, copytree
import os

# Activation functions, each followed by their respective derivatives;
# derivatives are expressed as function of the resulting outputs to simplify computations


def threshold_basic(x):  # nb: the derivative of this is not used because it's zero and we need a smoothing fn for classification
    if x >= 0.5:
        return 1
    else:
        return 0

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
            # by default weights are extracted from a uniform distribution bounded between +- sqrt(6/(fan_out+ fan_in))
            # according to Xavier initialization

            temp_rng = default_rng()
            self.weights = strtin_range*(2*temp_rng.random((fan_out, fan_in+1)) - 1)
            self.weights *= np.sqrt(6/(fan_in+fan_out))
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

        self.role = 'hidden'  # initialized as hidden layer by default, for output layer use switch_role_to_out_layer()
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

    def switch_role_to_out_layer(self, task):
        self.role = 'output'
        if task == 'regression':
            self.activation = np.vectorize(linear)
            self.derivative = np.vectorize(linearprime)
        if task == 'classification':
            self.activation = np.vectorize(threshold_basic)  # threshold_basic
            self.derivative = np.vectorize(linearprime)
            # (todo check this is correct) in classification, we optimize the loss as MSE because it is differentiable,
            #   so we derivate linear

    # evaluate(): evaluates input using current weights and stores input, net and output as class attributes
    # for later training
    def evaluate(self, entry, use_netx_instead_of_hx=False):
        self.latest_in[:-1] = entry
        self.latest_net = np.matmul(self.weights, self.latest_in)
        if use_netx_instead_of_hx:
            return self.latest_net  # nb: no activation function, in classification error fn uses w*x not h(x)
        else:
            self.latest_out = self.activation(self.latest_net)
            return self.latest_out

    # calc_local_gradient: calculates the layer's local gradient for the latest output according to
    # the back-propagating error signal from the next unit (? unit = neuron, layer, or sample?),
    # calculates actual gradient for weights update,
    # then returns error signal for the previous unit.
    def calc_local_gradient(self, error_signal):
        # if role == output unit:
        # error_signal == o_j - d_j
        # (this assumes error function to be L2 norm; different error functions would require different definitions
        # of "error signal" and potentially other tweaks)

        # if role == hidden unit k:
        # error_signal = sum_(j=1)^fan_out w_lj^(k+1)*delta_l

        # todo, check: delta weights should be (negative) derivative (partial, for each w) of E(w)
        p_gradient = error_signal*self.derivative(self.latest_out)
        # we sum local delta weigths for each sample (calc_local_gradient is called once per sample, per layer)
        # nb: calc_local_gradient is being called for a specific layer
        self.delta_weights = self.delta_weights - np.outer(p_gradient, self.latest_in)
        output_error_signal = np.matmul(self.weights.transpose(), p_gradient)

        '''
        if self.latest_out.size <= 2:
            print(f'self.latest_out: {self.latest_out}, der(latest_out): {self.derivative(self.latest_out)}, '
                  f'error_signal: {error_signal}, p_gradient: {p_gradient}')
        else:
            print(f'self.latest_out.shape: {self.latest_out.shape}, '
                  f'der(latest_out).shape: {self.derivative(self.latest_out).shape}, '
                  f'error_signal.shape: {error_signal.shape}, p_gradient.shape: {p_gradient.shape}')
        print(f'self.delta_weights.shape: {self.delta_weights.shape}')  # = (layer_out_dim, layer_nodes + 1_bias)
        # print(self.delta_weights)
        '''
        output_error_signal_up_to_index_1_from_the_end = output_error_signal[:-1]
        #print(f'output_error_signal.shape: {output_error_signal.shape}, '
        #      f'up_to_index_1_from_the_end.shape: {output_error_signal_up_to_index_1_from_the_end.shape}')
        # todo, explain: are we excluding
        # shape is (layer_size + 1_bias,)
        return output_error_signal_up_to_index_1_from_the_end

    # update_weights: update weights using both momentum and L2 regularization, updates delta_weights_old
    # for momentum and zeros delta_weights for successive training
    def update_weights(self, eta, alpha, lamda):

        self.delta_weights_old = eta*self.delta_weights + alpha*self.delta_weights_old

        # nb: lamda*self.weights, regularization component, is a L.. regularization
        # regularization formula (already derivative): -2*lambda*w (from slide 69 of lecture 03 linear knn)
        # so just lamda*self.weights seems to be missing a 2*
        self.weights += self.delta_weights_old - lamda*self.weights
        self.delta_weights = np.zeros(self.delta_weights.shape)


class NeuralNet:

    def __init__(self, activation, units=(2, 2), eta=0.1, alpha=0.1, lamda=0.01, mb=20,
                 task='classification', placeholder=False, verbose=False, **kwargs):

        # activation (string): name of activation function used
        #                      (TODO: add way to use different functions for different layers)
        # units (list): list of number of units in each layer
        # eta (float): learning rate
        # alpha (float): momentum parameter
        # lamda (float): regularization parameter
        # mb (int): number of patterns in each mini-batch
        # task (string): task the NN is meant to perform (classification or regression)
        # **kwargs: idk it'll be useful for something

        if verbose:
            print('creating NN, units: ', units)

        self.layers = []

        if not placeholder:
            self.fan_in = units[0]
            self.fan_out = units[-1]

            for i in range(len(units)-1):

                lay = Layer(units[i], units[i+1], activation)
                self.layers.append(lay)

            self.task = task
            self.layers[-1].switch_role_to_out_layer(task)

        self.layers_weights_best_epoch = []
        self.save_layers_weights_best_epoch()

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

        net.layers[-1].switch_role_to_out_layer(task)
        net.fan_in = net.layers[0].fan_in
        net.fan_out = net.layers[-1].fan_out

        return net

    # stores training set inside of neural net
    def load_training(self, training_set, out=1, do_normalization=True):

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

        if do_normalization:
            self.shift_vector = np.average(training_set, axis=0)
            self.scale_vector = np.std(training_set, axis=0)
        else:
            columns_count = self.fan_in + self.fan_out
            self.shift_vector = np.zeros((columns_count,), dtype=int)
            self.scale_vector = np.full((columns_count,), 1, dtype=int)

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

    def internal_evaluate(self, entry, use_netx_instead_of_hx=False):

        for lay in self.layers:
            is_last_layer = lay == self.layers[-1]
            if not is_last_layer:
                entry = lay.evaluate(entry)
            else:
                # if not use_netx_instead_of_hx:
                #   print('using activation/threshold fun in last layer')
                entry = lay.evaluate(entry, use_netx_instead_of_hx=use_netx_instead_of_hx)
        return entry

    # evaluate: evaluates non-normalized (original format) inputs and returns non-normalized outputs
    def evaluate(self, entry):
        print('using evaluate functions with normalization and denormalization')
        entry = (entry - self.shift_vector[:self.fan_in])/self.scale_vector[:self.fan_in]

        output = self.internal_evaluate(entry)

        # nb: index [self.fan_in:] is from after the fan_in count, so it's the last fan_out columns
        # was self.scale_vector[self.fan_in:] + self.shift_vector[self.fan_in:]
        # nb [-N:] means the last N elements
        # output = output * self.scale_vector[self.fan_in:] + self.shift_vector[self.fan_in:]
        output = output*self.scale_vector[-self.fan_out:] + self.shift_vector[-self.fan_out:]
        return output

    # pattern_update: evaluate input using current weights, then back-propagates error through the entire network
    def pattern_update(self, entry, label, verbose=False):
        use_netx_instead_of_hx = self.task == 'classification'  #
        output = self.internal_evaluate(entry, use_netx_instead_of_hx=use_netx_instead_of_hx)
        error_signal = output - label
        # print(f'{round(np.asscalar(output), 2)}; ', end=' ')
        if type(error_signal) is np.ndarray:
            entry_SE = (error_signal**2).sum()
        else:
            entry_SE = error_signal**2

        self.epoch_SE += entry_SE
        if verbose == 2:
            print(f'pattern update, predicted: {output} - label: {label} = error: {error_signal}, '
                  f'entry_SE_ {entry_SE}, updated epoch_SE: {self.epoch_SE}')

        # nb this is done for each pattern in a batch
        # todo, check: shouldn't we also propagate the L2 term (so the loss, not just the error)?
        for i, layer in reversed(list(enumerate(self.layers))):
            # print('calc_local_gradient layer ', i, ':')
            error_signal = layer.calc_local_gradient(error_signal)

        # todo, check: the final error signal is not returned here, so what's the point?

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

    def save_layers_weights_best_epoch(self):
        self.layers_weights_best_epoch = []
        for layer in self.layers:
            self.layers_weights_best_epoch.append(layer.weights.copy())

    def restore_layers_weights_best_epoch(self):
        for idx, layer in enumerate(self.layers):
            layer.weights = self.layers_weights_best_epoch[idx]

    # trains neural net for fixed number of epochs using mini-batch method; also saves MSE for each epoch on file
    def batch_training(self, stopping='epochs', patience=50, threshold=0.01, max_epochs=500,
                       verbose=False, hyperparams_for_plot=''):  # TODO: implement actual stopping conditions

        # stopping (string):        name of stopping condition to be used during training
        # threshold (int or float): numerical threshold at which to stop training;
        #                           exact meaning depends on stopping condition
        # max_epochs (int):         maximum number of epochs per training

        # list of implemented stopping conditions:

        # 'epochs': stop training after *max_epochs* number of epochs
        #           if epoch = max_epochs: break
        # 'MSE1_tr':    stop training once the relative change in the training error drops below *threshold*
        #           if (error[epoch] - error[epoch-1])/error[epoch] < threshold: break
        # 'MSE2_val':    stop training if the epoch with the best validation error is more than *patience* epochs ago,
        #           or after at least *patience* epochs therelative change in the validation error drops below *threshold*

        train_errors = np.empty(max_epochs)     # support array to store training errors
        validate_errors = np.empty(max_epochs)  # support array to store validation errors
        batch_rng = default_rng() # rng object to shuffle training set

        # number of examples used in each epoch (lower than total number of TRAIN examples, see batch_split()
        example_number = self.training_set.shape[0] - self.training_set.shape[0] % self.mb
        best_epoch_for_stopping = 0
        for epoch in range(max_epochs):
            self.epoch_SE = 0
            batch_rng.shuffle(self.training_set) # shuffle training set

            # Notes from slides:
            # delta_W = - (partial derivative E(patterns) over W)
            # = - sum for each pattern ( part.der. of E(pattern) over W ) =def - sum for each pattern ("delta_p_W")
            # for a particular w_tu, and pattern p: delta_p_wtu = - part.der of E(p) over w_tu =

            actual_used_tr_patterms = 0
            for batch in self.batch_split():

                for example in batch:
                    actual_used_tr_patterms += 1
                    x = example[:self.fan_in]
                    y = example[-self.fan_out:]

                    # delta_p_wtu = - part.der of E(p) over w_tu =
                    # = - (part.der of E(p) over net_t) * (part.der of net_t over w_tu)
                    # = delta_t * output_u
                    # out_u = (part.der of Sum_over_j
                    self.pattern_update(x, y)

                self.update_weights()

            train_errors[epoch] = self.epoch_SE/example_number
            validate_errors[epoch], accuracy = self.validate_net(epoch)

            if validate_errors[epoch] < validate_errors[best_epoch_for_stopping]:
                best_epoch_for_stopping = epoch
                self.save_layers_weights_best_epoch()

            if verbose:
                print('epoch train error: ', train_errors[epoch])
            if not epoch % 100: # prints training status on console every 100 epochs
                self.savestate(epoch)
                print(f'epoch {epoch} done (tr error = {train_errors[epoch]}, tr patterns: {example_number}, '
                      f' actual_used_tr_patterms: {actual_used_tr_patterms}, '
                      f'val error = {validate_errors[epoch]}, val patterns: {self.validation_set.shape[0]}, '
                      f'accuracy = {accuracy})')

            # check for stopping conditions
            if self.should_stop_training(stopping, threshold, max_epochs, epoch, best_epoch_for_stopping, patience,
                                         validate_errors, train_errors):
                break

        train_error_path = os.path.join(self.net_dir, 'training_errors.csv')
        np.savetxt(train_error_path, train_errors[:epoch+1], delimiter=',')  # saves history of training errors on file

        validate_error_path = os.path.join(self.net_dir, 'validation_errors.csv')
        np.savetxt(validate_error_path, validate_errors[:epoch+1], delimiter=',')  # saves history of training errors on file

        import matplotlib.patches as mpatches
        f, ax = plt.subplots(1)
        ax.plot(validate_errors[:epoch+1], label='validation errors')
        ax.plot(train_errors[:epoch+1], label='training errors')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='none', label=hyperparams_for_plot))
        ax.legend(handles=handles)

        error_graph_path = os.path.join(self.net_dir, 'errors.png')
        plt.savefig(error_graph_path)

    def should_stop_training(self, stopping, threshold, max_epochs, epoch, best_epoch_for_stopping, patience,
                             validate_errors, train_errors):
        if stopping == 'epochs':
            if epoch == max_epochs:
                return True
        elif stopping == 'MSE2_val':
            if epoch - best_epoch_for_stopping > patience:
                self.restore_layers_weights_best_epoch()
                print(f'Stopping: at epoch {epoch}, '
                      f'the epoch with the best validation error is n. {best_epoch_for_stopping}, '
                      f'more than {patience} epochs ago.')
                return True
            else:
                if epoch > patience:
                    relative_change = abs(validate_errors[epoch] - validate_errors[epoch - 1]) / validate_errors[epoch - 1]
                    if relative_change < threshold:
                        self.restore_layers_weights_best_epoch()
                        print(f'Stopping: at epoch {epoch}, '
                              f'relative_change={relative_change} less than {threshold}.')
                        return True
        elif stopping == 'MSE1_tr':
            relative_change = (train_errors[epoch] - train_errors[epoch - 1]) / train_errors[epoch - 1]
            if relative_change < threshold:
                return True
        return False

    # calculates MSE on the validation set with current weights
    def validate_net(self, epoch=None):
        # todo: add metrics (accuracy) for classification

        error_smooth = 0
        total = 0
        correct = 0
        accu = None

        validation_predicted_outs = []
        for example in self.validation_set:
            x = example[:self.fan_in]
            y = example[-self.fan_out:]

            if self.task == 'regression':
                predicted_y = self.internal_evaluate(x)
            if self.task == 'classification':
                predicted_y = self.internal_evaluate(x, use_netx_instead_of_hx=True)
                discretized_predicted_y = self.internal_evaluate(x, use_netx_instead_of_hx=False)
            # print(f'predicted y:  {predicted_y}, thresholded_predicted_y: {thresholded_predicted_y}, actual y: {y}')
            validation_predicted_outs.append(np.asscalar(predicted_y))
            if type(y) is np.ndarray:
                # todo pass error fn (MSE, MAE, ..) as parameter like for activation functions
                error_smooth += ((y - predicted_y)**2).sum()
            else:
                error_smooth += (y - predicted_y)**2

            if self.task == 'classification':
                # _, predicted = outputs.max(1)
                total += 1
                correct += np.sum(discretized_predicted_y == y)
                # print(f'thresholded_predicted_y: {thresholded_predicted_y}, y: {y}')

        if self.task == 'classification':
            accu = 100. * correct / total  # accuracy scores for classification tasks
            # if isinstance(epoch, int) and not epoch % 100:  # prints every 100 epochs
                # print(f'correct: {correct}, total: {total}')
                # print(f'val predicted: {validation_predicted_outs}')
        error_smooth /= self.validation_set.shape[0]
        return error_smooth, accu


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
