import math
from time import sleep
from json import dump
from datetime import datetime
from shutil import rmtree, copytree
import os

from numpy.random import default_rng
import numpy as np

from plot_data import *


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
            # by default weights are extracted from a uniform distribution bounded between +- sqrt(6/(fan_out+ fan_in))
            # according to Xavier initialization

            temp_rng = default_rng()
            self.weights = strtin_range*(2*temp_rng.random((fan_out, fan_in+1)) - 1)
            self.weights *= np.sqrt(6/(fan_in+fan_out))
            self.delta_weights = np.zeros([fan_out, fan_in + 1])

        else:
            self.weights = kwargs['weights']
            self.delta_weights = kwargs['deltas']

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
    def load_file(cls, base_path, layer_number, activation, act_parameters=(1.7159, 2/3)):

        weight_path = os.path.join(base_path, 'layer_{}_weights.csv'.format(layer_number))
        delta_path = os.path.join(base_path, 'layer_{}_deltas.csv'.format(layer_number))
        weights = np.loadtxt(weight_path, delimiter=';')
        deltas = np.loadtxt(delta_path, delimiter=';')
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
            self.activation = np.vectorize(sigmoid)  # threshold_basic
            self.derivative = np.vectorize(sigmprime)
            # (todo check this is correct) in classification, we optimize the loss as MSE because it is differentiable,
            #   so we use the derivative of the linear fn

    # evaluate(): evaluates input using current weights and stores input, net and output as class attributes
    # for later training
    def evaluate(self, entry):
        self.latest_in[:-1] = entry
        self.latest_net = np.matmul(self.weights, self.latest_in)
        self.latest_out = self.activation(self.latest_net)

        return self.latest_out

    # calc_local_gradient: calculates the layer's local gradient for the latest output according to
    # the back-propagating error signal from the next unit (? unit = neuron, layer, or sample?),
    # calculates actual gradient for weights update,
    # then returns error signal for the previous unit.
    def calc_local_gradient(self, dEp_dOt):
        # we sum local delta weigths for each sample (calc_local_gradient is called once per sample, per layer)
        # nb: calc_local_gradient is being called for a specific layer

        # NB: in the first call, dEp_dOt corresponds to the equation for the output layer
        # if role == output unit:
        # error_signal == o_j - d_j
        # (this assumes error function to be L2 norm; different error functions would require different definitions
        # of "error signal" and potentially other tweaks)

        # if role == hidden unit k:
        # error_signal = sum_(j=1)^fan_out w_lj^(k+1)*delta_l
        # error_signal or delta_t ∀ unit t
        error_signal = dEp_dOt*self.derivative(self.latest_out)
        # nb: we use latest_out instead of latest_net
        # because, for convenience, the derivative functions here are expressed in terms of the out, not of the net

        # error_signal delta_t * output of previous layer
        delta_weights_for_current_pattern = -1 * np.outer(error_signal, self.latest_in)  # this is delta_p_w_tu (∀ tu)
        self.delta_weights += delta_weights_for_current_pattern

        # this is the equation of dEp_dOt (for each weight in next layer) for the hidden layer case
        nextlayer_dEp_dOt = np.matmul(self.weights.transpose(), dEp_dOt)

        nextlayer_dEp_dOt_without_the_biases = nextlayer_dEp_dOt[:-1]
        # print(f'nextlayer_dEp_dOt.shape: {nextlayer_dEp_dOt.shape}, '
        #      f'nextlayer_dEp_dOt_without_the_biases.shape: {nextlayer_dEp_dOt_without_the_biases.shape}')
        # shape is (layer_size + 1_bias,)
        return nextlayer_dEp_dOt_without_the_biases

    # update_weights: update weights using both momentum and L2 regularization, updates delta_weights_old
    # for momentum and zeros delta_weights for successive training
    def update_weights(self, eta, alpha, lamda):

        self.delta_weights_old = eta*self.delta_weights + alpha*self.delta_weights_old

        # nb: lamda*self.weights, regularization component, is a L.. regularization
        # regularization formula (already derivative): -2*lambda*w (from slide 69 of lecture 03 linear knn)
        self.weights += self.delta_weights_old - lamda*self.weights
        self.delta_weights = np.zeros(self.delta_weights.shape)


class NeuralNet:

    def __init__(self, activation, units=(2, 2), eta=0.1, alpha=0.1, lamda=0.01, mb=20,
                 task='classification', error='MSE', placeholder=False, verbose=False, **kwargs):

        # activation (string): name of activation function used
        #                      (TODO: add way to use different functions for different layers)
        # units (list): list of number of units in each layer
        # eta (float): learning rate
        # alpha (float): momentum parameter
        # lamda (float): regularization parameter
        # mb (int): number of patterns in each mini-batch
        # task (string): task the NN is meant to perform (classification or regression)
        # error (string): error function to be minimized during training
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

        self.task = task
        self.error_func = error  # error function
        self.eta = eta  # learning rate
        self.alpha = alpha  # momentum parameter
        self.lamda = lamda  # regularization parameter
        self.mb = mb  # number of patterns in each mini-batch
        self.batch_SE = 0 # squared error on latest mini-batch
        self.epoch_SE = 0 # squared error on latest epoch

        self.hyperparameters = {'task': self.task, 'error': self.error_func, 'hidden activation': activation,
                                'eta': self.eta, 'alpha': self.alpha, 'lambda': self.lamda, 'mb': self.mb}
        if self.task == 'classification':
            self.hyperparameters['output activation'] = 'sigmoid'

        elif self.task == 'regression':
            self.hyperparameters['output activation'] = 'linear'

        self.validation_set = None  # placeholder for internal validation set
        self.training_set = None    # placeholder for internal training set

        # placeholders for vectors used for variable normalization x'_i = (x_i - <x_i>)/std(x_i)
        self.shift_vector = None    # placeholder for internal validation set
        self.scale_vector = None    # placeholder for internal validation set

        base_dir = os.getcwd()
        if 'grid_search_dir' in kwargs:
            grid_search_dir = kwargs.get('grid_search_dir')# kwargs['grid_search_dir']
        else:
            grid_search_dir = ''

        if 'dir' in kwargs:  #.keys()
            trial_dir = kwargs['dir']
        else:
            trial_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.net_dir = os.path.join(base_dir, grid_search_dir, trial_dir)

        # save the starting conditions, if random:
        self.savestate(0)

    @classmethod
    def load_net(cls, path, activation, eta=0.1, alpha=0.1, lamda=0.01, mb=20, task='classification', error='MSE',
                 **kwargs):

        net = NeuralNet(activation, eta=eta, alpha=alpha, lamda=lamda, mb=mb,
                        task=task, error=error, placeholder=True, **kwargs)

        all_files = os.listdir(path)
        layer_files = [file for file in all_files if (file[:6] == 'layer_' and file[-4:] == '.csv')]
        if (len(layer_files) % 2):
            ValueError('Mismatch in number of layer files in folder {}'.format(path))
        num_of_layers = len(layer_files)//2

        for i in range(num_of_layers):
            lay = Layer.load_file(path, i, activation)
            net.layers.append(lay)

        net.layers[-1].switch_role_to_out_layer(task)
        net.fan_in = net.layers[0].fan_in
        net.fan_out = net.layers[-1].fan_out

        return net

    # loads trained net based on latest state + info gathered from net_summary.json
    # path (string): filepath to training run (folder containing net_summary.json, 'latest' and all 'epoch_x')
    @classmethod
    def load_trained(cls, path, state='latest'):

        if state == 'latest':
            state_path = os.path.join(path, 'latest')

        else:
            state_path = os.path.join(path, 'epoch_0')

        summary_path = os.path.join(path, 'net_summary.json')

        with open(summary_path, 'r') as infile:
            summary = json.load(infile)

        hyperpars = summary['hyperparameters']

        net = NeuralNet.load_net(state_path, hyperpars['hidden activation'], hyperpars['eta'], hyperpars['alpha'],
                                 hyperpars['lambda'], hyperpars['mb'], hyperpars['task'], hyperpars['error'])

        return net

    # stores training set inside of neural net
    def load_training(self, training_set, out=1, do_normalization=True):

        train_examples = training_set.shape[0]
        print(f'Loading {train_examples} train examples..')
        inp = training_set.shape[1] - out
        if inp != self.fan_in:
            print(f'out: {out}, training_set.shape[1]: {training_set.shape[1]}')
            raise ValueError(f'Number of input variables in training set ({inp}) doesn\'t match input units ({self.fan_in})!')

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
            np.savetxt(weight_path, layer.weights, delimiter=';')
            np.savetxt(delta_path, layer.delta_weights, delimiter=';')

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
        output = self.internal_evaluate(entry)
        error_pattern = output - label  # NB. dEp_dOt is one of the terms of the error signal delta_t (for each unit t)
        # print(f'{round(np.asscalar(output), 2)}; ', end=' ')
        if type(error_pattern) is np.ndarray:
            entry_SE = (error_pattern**2).sum()
        else:
            entry_SE = error_pattern**2

        if self.error_func == 'MSE':
            dEp_dOt = error_pattern

        elif self.error_func == 'MEE':
            entry_SE = np.sqrt(entry_SE)
            dEp_dOt = error_pattern/entry_SE

        self.epoch_SE += entry_SE

        if verbose == 2:
            print(f'pattern update, predicted: {output} - label: {label} = error: {dEp_dOt}, '
                  f'entry_SE_ {entry_SE}, updated epoch_SE: {self.epoch_SE}')

        # nb this is done for each pattern in a batch
        # todo, check: shouldn't we also propagate the L2 term (so the loss, not just the error)?
        for i, layer in reversed(list(enumerate(self.layers))):
            # print('calc_local_gradient layer ', i, ':')
            dEp_dOt = layer.calc_local_gradient(dEp_dOt)

        # todo, check: the final error signal is not returned here, so what's the point?

    # update weights using previously-calculated weight deltas
    def update_weights(self):

        for layer in self.layers:
            layer.update_weights(self.eta/self.mb, self.alpha/self.mb, self.lamda/self.mb)
        squared_gradient_last_layer = np.square(self.layers[-1].delta_weights_old)
        euclidean_norm_grad_last_layer = math.sqrt(np.sum(squared_gradient_last_layer))
        return euclidean_norm_grad_last_layer


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
    def batch_training(self, stopping='epochs', patience=50, threshold=0.01, max_epochs=500, error_func=None,
                       verbose=False, hyperparams_for_plot='', trial_name='', save_plot_to_file=True,
                       grid_search_dir_for_plots='', gridsearch_descr='single_run'):

        if error_func is None:
            error_func = self.error_func
        # stopping (string):        name of stopping condition to be used during training
        # threshold (int or float): numerical threshold at which to stop training;
        #                           exact meaning depends on stopping condition
        # max_epochs (int):         maximum number of epochs per training
        # error_func (string):      error function to use when evaluating neural net; defaults to same one minimized
        #                           by training but can be chosen to be different
        # list of implemented stopping conditions:

        # 'epochs': stop training after *max_epochs* number of epochs
        #           if epoch = max_epochs: break
        # 'MSE1_tr':    stop training once the relative change in the training error drops below *threshold*
        #           if (error[epoch] - error[epoch-1])/error[epoch] < threshold: break
        # 'MSE2_val':    stop training if the epoch with the best validation error is more than *patience* epochs ago,
        #           or after at least *patience* epochs therelative change in the validation error drops below *threshold*

        train_errors = np.empty(max_epochs)     # support array to store training errors
        validate_errors = np.empty(max_epochs)  # support array to store validation errors
        vl_misclassification_rates = np.empty(max_epochs)  # support array to store miclassification rates
        batch_rng = default_rng()  # rng object to shuffle training set

        # number of examples used in each epoch (lower than total number of TRAIN examples, see batch_split()
        example_number = self.training_set.shape[0] - self.training_set.shape[0] % self.mb
        num_of_batches = example_number / self.mb
        best_epoch_for_stopping = 0
        done_epochs = 0
        for epoch in range(max_epochs):
            self.epoch_SE = 0
            batch_rng.shuffle(self.training_set) # shuffle training set

            # Notes from slides:
            # delta_W = - (partial derivative E(patterns) over W)
            # = - sum for each pattern ( part.der. of E(pattern) over W ) =def - sum for each pattern ("delta_p_W")
            # for a particular w_tu, and pattern p: delta_p_wtu = - part.der of E(p) over w_tu =

            actual_used_tr_patterms = 0
            avg_euclidean_norm_grad_last_layer = 0
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

                avg_euclidean_norm_grad_last_layer += self.update_weights()

            avg_euclidean_norm_grad_last_layer /= num_of_batches
            train_errors[epoch] = self.epoch_SE/example_number
            validate_errors[epoch], accuracy, vl_misclassification_rates[epoch] = self.validate_net(epoch, error_func)

            # todo: if no validation set has been explicitly loaded, automatically split a portion of the training set
            if validate_errors[epoch] < validate_errors[best_epoch_for_stopping]:
                best_epoch_for_stopping = epoch
                self.save_layers_weights_best_epoch()

            if verbose:
                print('epoch train error: ', train_errors[epoch])
            if not epoch % 100 and epoch > 0:  # prints training status on console every 100 epochs
                self.savestate(epoch)
                accuracy_info = f', accuracy = {accuracy:.3f}' if accuracy is not None else ''
                print(f'epoch {epoch} done (tr error = {train_errors[epoch]:.3f}, tr patterns: {example_number} '
                      f'(of which used: {actual_used_tr_patterms}), '
                      f'val error = {validate_errors[epoch]:.3f}, val patterns: {self.validation_set.shape[0]}'
                      f'{accuracy_info})')

            # check for stopping conditions
            if self.should_stop_training(stopping, threshold, max_epochs, epoch, best_epoch_for_stopping, patience,
                                         validate_errors, train_errors, avg_euclidean_norm_grad_last_layer):
                break
            done_epochs = epoch

        print(f'run for {done_epochs + 1} epochs')
        best_tr_error = train_errors[best_epoch_for_stopping]
        epochs_done = best_epoch_for_stopping

        summary_path = os.path.join(self.net_dir, 'net_summary.json')
        final_MEE_error, _, _ = self.validate_net(error_func='MEE')  # .tolist()
        summary = {'hyperparameters': self.hyperparameters, 'training errors': train_errors[:epoch+1].tolist(),
                   'validation errors': validate_errors[:epoch+1].tolist(), 'shift vector': self.shift_vector.tolist(),
                   'scale vector': self.scale_vector.tolist(), 'final MEE': [final_MEE_error]}

        if self.task == 'classification':
            summary['misclassification'] = vl_misclassification_rates[:epoch+1].tolist()

        with open(summary_path, 'w') as outfile:
            dump(summary, outfile)

        train_error_path = os.path.join(self.net_dir, 'training_errors.csv')
        np.savetxt(train_error_path, train_errors[:epoch+1], delimiter=';')  # saves history of training errors on file
        validate_error_path = os.path.join(self.net_dir, 'validation_errors.csv')
        np.savetxt(validate_error_path, validate_errors[:epoch+1], delimiter=';')  # saves history of training errors on file
        vl_misclassification_rates_path = os.path.join(self.net_dir, 'vl_misclassification_rates.csv')
        np.savetxt(vl_misclassification_rates_path, vl_misclassification_rates[:epoch+1], delimiter=';')

        if save_plot_to_file:
            plot_learning_curve_to_img_file(validate_errors, train_errors, vl_misclassification_rates, epoch,
                                                      hyperparams_for_plot, self.task, self.error_func, self.net_dir,
                                                      trial_name)

        append_learning_curve_plot_data_to_file(validate_errors, train_errors, vl_misclassification_rates,
                                                epoch, hyperparams_for_plot, self.task, self.error_func,
                                                self.net_dir, trial_name='',
                                                gridsearch_descr=gridsearch_descr)
        return best_tr_error, epochs_done, final_MEE_error

    def should_stop_training(self, stopping, threshold, max_epochs, epoch, best_epoch_for_stopping, patience,
                             validate_errors, train_errors, euclidean_norm_grad_last_layer):
        if stopping == 'epochs':
            if epoch == max_epochs:
                return True
        elif stopping == 'EuclNormGrad':
            print(f'Epoch {epoch}, euclidean_norm_grad_last_layer : {euclidean_norm_grad_last_layer}')
            if epoch - best_epoch_for_stopping > patience and \
                    euclidean_norm_grad_last_layer < threshold:
                print(f'Stopping: at epoch {epoch}, '
                      f' because euclidean_norm_grad_last_layer < {threshold}')
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
                    past_epochs_to_count = patience
                    mean_relative_change = 0
                    for epoch_idx in range(epoch-past_epochs_to_count+1, epoch+1):
                        mean_relative_change += abs(validate_errors[epoch_idx] - validate_errors[epoch_idx - 1]) \
                                                / validate_errors[epoch_idx - 1]
                    mean_relative_change /= past_epochs_to_count
                    # print(f'mean_relative_change: {mean_relative_change}, threshold: {threshold}')
                    if mean_relative_change < threshold:
                        self.restore_layers_weights_best_epoch()
                        print(f'Stopping: at epoch {epoch}, '
                              f'mean_relative_change={mean_relative_change} (over the last {past_epochs_to_count} epochs) '
                              f'is less than {threshold}.')
                        print(f'restoring weights at epoch {best_epoch_for_stopping}')
                        return True
        elif stopping == 'MSE1_tr':
            relative_change = (train_errors[epoch] - train_errors[epoch - 1]) / train_errors[epoch - 1]
            if relative_change < threshold:
                return True
        return False

    # calculates MSE on the validation set with current weights
    def validate_net(self, epoch=None, error_func=None):
        if error_func is None:
            error_func = self.error_func
        # todo: add metrics (accuracy) for classification

        error_smooth = 0
        total = 0
        correct = 0
        accu = None

        # validation_predicted_outs = []
        for example in self.validation_set:
            x = example[:self.fan_in]
            y = example[-self.fan_out:]

            predicted_y = self.internal_evaluate(x)

            if self.task == 'classification':
                discretized_predicted_y = np.around(predicted_y)
            # print(f'predicted y:  {predicted_y}, thresholded_predicted_y: {thresholded_predicted_y}, actual y: {y}')

            # todo: adapt this for outdim > 2, useful to later plot actual vs predicted graph
            # validation_predicted_outs.append(np.asscalar(predicted_y))
            if type(y) is np.ndarray:
                # todo pass error fn (MSE, MAE, ..) as parameter like for activation functions
                current_error = ((y - predicted_y)**2).sum()
            else:
                current_error = (y - predicted_y)**2

            if error_func == 'MSE':
                error_smooth += current_error

            if error_func == 'MEE':
                error_smooth += np.sqrt(current_error)

            if self.task == 'classification':
                # _, predicted = outputs.max(1)
                total += 1
                correct += np.sum(discretized_predicted_y == y)
                # print(f'thresholded_predicted_y: {thresholded_predicted_y}, y: {y}')

        misclassification_rate = None
        if self.task == 'classification':
            accu = 100. * correct / total  # accuracy scores for classification tasks
            misclassification_rate = (total - correct) / total
            # if isinstance(epoch, int) and not epoch % 100:  # prints every 100 epochs
                # print(f'correct: {correct}, total: {total}')
                # print(f'val predicted: {validation_predicted_outs}')
        error_smooth /= self.validation_set.shape[0]
        return error_smooth, accu, misclassification_rate


    # evaluates MEE or MSE on chosen dataset using original scale rather than normalized values;
    # set (string or np.ndarray): dataset to evaluate mean error on, must be either string for validation or
    # training sets or numpy array containing NOT-NORMALIZED entries
    def evaluate_original_error(self, set='validation', error_fn=None):

        if error_fn is None:
            error_fn = self.error_func

        if set == 'validation':
            data_set = self.validation_set

        elif set == 'training':
            data_set = self.training_set

        else:
            data_set = set

        error = 0
        set_size = data_set.shape[0]

        for entry in data_set:

            x = entry[:self.fan_in]
            y = entry[-self.fan_out:]

            if (set == 'validation') or (set == 'training'):

                predicted_y = self.internal_evaluate(x)
                distance = self.scale_vector[-self.fan_out:]*(y - predicted_y)

            else:

                predicted_y = self.evaluate(x)
                distance = y - predicted_y

            if type(distance) is np.ndarray:
                SE = (distance**2).sum()

            else:
                SE = distance**2

            if error_fn == 'MSE':
                error += SE

            elif error_fn == 'MEE':
                error += np.sqrt(SE)

        error /= set_size

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
    print(f'initial validation_error = {test_net.validate_net()[0]:.3f}')
    test_net.batch_training()
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print(f'final validation_error = {test_net.validate_net()[0]}')