import os
from collections import OrderedDict
from datetime import datetime

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler


class NeuralNetwork(nn.Module):
    """Multi-layer Perceptron regressor.
        Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    """
    def __init__(self, hidden_layer_sizes, activation_fun):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()  # what does this do?

        hidden_layer_sizes_as_list = NeuralNetwork.__get_layers_descr_as_list(hidden_layer_sizes, activation_fun)
        self.linear_relu_stack = nn.Sequential(OrderedDict(hidden_layer_sizes_as_list))  # this actually creates the NN

    @staticmethod
    def __get_layers_descr_as_list(hidden_layer_sizes, activation_fun):
        layers = []
        last_layer_idx = len(hidden_layer_sizes) - 2
        print('last_layer_idx: ', last_layer_idx)
        for layer_idx in range(len(hidden_layer_sizes) - 1):
            print('layer: ', hidden_layer_sizes[layer_idx], ', ', hidden_layer_sizes[layer_idx + 1])
            linear_layer = ('linear' + str(layer_idx), nn.Linear(hidden_layer_sizes[layer_idx],
                                                                      hidden_layer_sizes[layer_idx + 1]))
            layers.append(linear_layer)
            is_last_layer = layer_idx >= last_layer_idx
            if not is_last_layer:  # NB.: not adding an activation function after the last layer
                layers.append((str(activation_fun)[0:5] + str(layer_idx), activation_fun))
        return layers

    def forward(self, x):
        # todo: clarify how does this work/is implemented
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_in_training_mode(self):
        super(NeuralNetwork, self).train()

    def set_in_validation_mode(self):
        super(NeuralNetwork, self).eval()


class FeaturesDataset(Dataset):
    def __init__(self, filename, dtype1, sep=',', dtype2=None, header='infer', col_names=None,
                 scaler=None):  # , rows_count=None, cols_count=None
        file_out = pd.read_csv(filename, sep=sep, header=header, names=col_names, dtype=dtype1)
        # print(file_out.to_string())
        print(file_out.info())
        print(file_out.head())

        if dtype2 is not None:
            file_out = file_out.astype(dtype=dtype2)
            print('after astype: ')
            print(file_out.info())
            print(file_out.head())

        # torch.utils.data.random_split(dataset, lengths)
        rows_count = len(file_out)
        cols_count = len(file_out.columns) - 1
        print('rows_count: ', rows_count, ', cols_count: ', cols_count)
        x = file_out.iloc[0:rows_count, 0:cols_count].values
        x = x.astype(np.float32)
        print('x type: ', type(x), 'x shape: ', x.shape, ' head:')
        print(x[:5])
        y = file_out.iloc[0:rows_count, cols_count].values
        print('y type: ', type(y), 'y shape: ', y.shape, ' head:')
        print(y[:50])

        # todo: linear base expansion
        # add x^2 ..for each input

        y_reshaped = y.reshape(-1, 1)  # todo: adapt also for when output is more than one feature/dimension
        print('y shape:', y.shape, ', y_reshaped type: ', type(y_reshaped), ', shape: ',
              y_reshaped.shape)  # numpy.ndarray
        print('y_reshaped head:')
        print(y_reshaped[:5])

        #Feature scaling
        if scaler is not None:
            # todo: option to have no feature scaling
            #  pass the scaler with the config hyperparams
            # todo: also pass the torch dtype (es torch.float32) to be used for x, y after feature scaling
            # check type before and after scaling
            # before scaling: x, y are numpy ..(nd array?)
            # after scaling: x_train is ..ndarray

            self.features_scaler = scaler()  
            x_train = self.features_scaler.fit_transform(x)
            self.labels_scaler = scaler()
            y_train = self.labels_scaler.fit_transform(y_reshaped)  # = y
            print('x head after rescaling:')
            print(x_train[:5])
            print('y head after rescaling:')
            print(y_train[:5])
        else:
            self.features_scaler = None
            self.labels_scaler = None
            x_train = x
            y_train = y_reshaped
            
        # converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


def get_L2_norm_for_regularization(model):
    return sum(p.pow(2.0).sum()
        for p in model.parameters())


def train(trainloader, device, model, loss_fn, train_accuracy_all_epochs, train_errors_all_epochs,
          train_losses_all_epochs, optimizer, l2_lambda):

    model.set_in_training_mode()

    running_error_this_epoch = 0
    running_loss_this_epoch = 0
    correct = 0
    total = 0

    for batch_of_data in tqdm(trainloader):
        inputs, labels = batch_of_data[0].to(device), batch_of_data[1].to(device)

        optimizer.zero_grad()  # clear previous gradients
        outputs = model(inputs)
        # print('outputs this batch: ')
        # print(outputs.tolist())
        # print('labels this batch: ')
        # print(labels.tolist())

        # print("outputs type:", outputs.type())  # outputs type: torch.cuda.FloatTensor
        # print("labels type:", labels.type())  # labels type: torch.cuda.FloatTensor
        # outputs = outputs.to(device, dtype=torch.float64)
        # labels = labels.to(device, dtype=torch.float64)

        error_this_batch = loss_fn(outputs, labels)  # NB: type(error_this_batch): torch.Tensor
        running_error_this_epoch += error_this_batch.item()

        # Replaces pow(2.0) with abs() for L1 regularization
        l2_norm = get_L2_norm_for_regularization(model)
        loss_this_batch = error_this_batch + (l2_lambda * l2_norm)

        # todo: check how it's done (backprop, weights update, gradient descent, derivative of loss)
        loss_this_batch.backward()  # backward pass (why called on loss/error object? (which is a tensor))
        optimizer.step()
        running_loss_this_epoch += loss_this_batch.item()

        # accuracy scores for classification tasks
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    final_train_error_this_epoch = running_error_this_epoch / len(trainloader)
    final_train_loss_this_epoch = running_loss_this_epoch / len(trainloader)

    accu = 100. * correct / total  # accuracy scores for classification tasks

    train_errors_all_epochs.append(final_train_error_this_epoch)
    train_losses_all_epochs.append(final_train_loss_this_epoch)
    train_accuracy_all_epochs.append(accu)
    print('On training with backprop after each minibatch in epoch: Loss : %.3f | Error: %.3f | Accuracy: %.3f' % (final_train_loss_this_epoch, final_train_error_this_epoch, accu))


def evaluate(dataloader, device, model, loss_fn, accuracy_all_epochs, errors_all_epochs):

    model.set_in_validation_mode()

    running_error_this_epoch = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_of_data in tqdm(dataloader):
            # print('validation data batch, as nested lists, len/size: ',
            #      len(batch_of_data[0][0]), 'x', len(batch_of_data[0]), ', 1 x', len(batch_of_data[1]))  # 5x64, 1x64
            inputs, labels = batch_of_data[0].to(device), batch_of_data[1].to(device)

            outputs = model(inputs)

            error_this_batch = loss_fn(outputs, labels)
            running_error_this_epoch += error_this_batch.item()

            # accuracy scores for classification tasks
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    final_test_error_this_epoch = running_error_this_epoch / len(dataloader)
    accu = 100. * correct / total # accuracy scores for classification tasks

    errors_all_epochs.append(final_test_error_this_epoch)
    accuracy_all_epochs.append(accu)

    print('Error at the end of epoch: %.3f | Accuracy: %.3f' % (final_test_error_this_epoch, accu))


def get_dev_data_from_file(mini_batch_size, filename, dtype1, sep, dtype2=None, header='infer', col_names=None, 
                           scaler=None):  # , rows_count=None, cols_count=None
    from torch.utils.data import DataLoader
    print('getting data from ', filename, '..')
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dev_set = FeaturesDataset(filename, dtype1, sep, dtype2, header, col_names, scaler)
    train_fraction = 0.8
    train_count = int(len(dev_set) * train_fraction)
    split_ratios = [train_count, len(dev_set) - train_count]
    print('split ratios: ', split_ratios)
    train_set, validation_set = torch.utils.data.dataset.random_split(dev_set, split_ratios)
    train_set_loader = DataLoader(train_set, batch_size=mini_batch_size, shuffle=True, num_workers=2)
    validation_set_loader = DataLoader(validation_set, batch_size=mini_batch_size, shuffle=True, num_workers=2)

    return train_set_loader, validation_set_loader, train_set, validation_set, dev_set


def get_dev_data_from_library(mini_batch_size):
    from torch.utils.data import DataLoader

    print('getting data..')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(train_set, batch_size=mini_batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_set, batch_size=mini_batch_size, shuffle=False, num_workers=2)

    return trainloader, validation_loader


def train_and_validate_NN_model(device, hidden_layer_sizes, activation_fun, learning_rate, momentum, l2_lambda,
                                loss_fn, trainloader, validationloader, adaptive_learning_rate='constant'):
    # todo, passing functions as parameter, enable choice btw optimizers, normalization function, ..

    print('creating model and sending to cpu/gpu..')
    model = NeuralNetwork(hidden_layer_sizes, activation_fun).to(device)
    print(model)
    if adaptive_learning_rate.lower() == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif adaptive_learning_rate.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif adaptive_learning_rate.lower() == 'AdaGrad'.lower():
        # other params:
        # lr_decay=0 (learning rate decay),
        # weight_decay=0 (L2 penalty),
        # initial_accumulator_value=0,
        # eps=1e-10 (added to the denominator to improve numerical stability)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif adaptive_learning_rate.lower() == 'RMSProp'.lower():
        # other params:
        # alpha=0.99 (smoothing constant),
        # eps=1e-08 (added to the denominator to improve numerical stability),
        # weight_decay=0 (L2 penalty)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # model1
    print('running model 1:')
    train_losses_all_epochs = []
    train_errors_variable_by_batches = []
    train_accuracy_all_epochs = []
    train_accuracy2_all_epochs = []
    train_errors2_all_epochs = []
    eval_errors_all_epochs = []
    eval_accu_all_epochs = []

    # todo: add ability to save model to an epoch and then resume training later
    # todo: add ability for "live" learning curve graph (updated during training, at each epoch)

    epochs_count = 10
    epochs_sequence = range(1, epochs_count + 1)
    print('starting training and validation epochs..')
    epochs_startime = datetime.now()
    for epoch in epochs_sequence:
        print('\nEpoch : %d' % epoch)
        train(trainloader, device, model, loss_fn, train_accuracy_all_epochs, train_errors_variable_by_batches,
              train_losses_all_epochs, optimizer, l2_lambda)
        print('On training set:')
        evaluate(trainloader, device, model, loss_fn, train_accuracy2_all_epochs, train_errors2_all_epochs)
        print('On validation set:')
        evaluate(validationloader, device, model, loss_fn, eval_accu_all_epochs, eval_errors_all_epochs)
    epochs_completetime = datetime.now()  # fromtimestamp(os.path.getmtime("x.cache"))
    epochs_total_duration = epochs_completetime - epochs_startime
    print('Completed training and validation epochs in ', epochs_total_duration.total_seconds(), ' seconds.')
    return model, epochs_sequence, train_losses_all_epochs, train_errors2_all_epochs, train_accuracy2_all_epochs, \
           eval_errors_all_epochs, eval_accu_all_epochs, train_errors_variable_by_batches


def plot_hyperparams_descr(ax, hyperparams_descr):
    plt.text(0.01, 0.95, hyperparams_descr, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)


def plot_learning_curves(epochs_sequence, train_losses, train_errors, eval_errors, train_errors_variable_by_batches,
                         hyperparams_descr):
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    plot_hyperparams_descr(ax1, hyperparams_descr)

    l1, = ax1.plot([x - 0.5 for x in epochs_sequence], train_losses, '-o', color='purple')
    l4, = ax1.plot([x - 0.5 for x in epochs_sequence], train_errors_variable_by_batches, '-o', color='dodgerblue')
    ax1.set_title('train_losses and errors (during training, weights/params change within epoch after each batch)')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss/error')
    ax1.legend([l1, l4], ["train_losses (during epoch)", "train_errors (during epoch)"])

    l2, = ax2.plot(epochs_sequence, train_errors, '-o', color='blue')
    l3, = ax2.plot(epochs_sequence, eval_errors, '-o', color='orange')
    ax2.set_title('train and validation errors (at the end of each epoch)')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('error')
    ax2.legend([l2, l3], ["train_errors (end of epoch)", "eval_errors (end of epoch)"])
    plt.show()

    # plt.plot(train_losses, '-o')
    # plt.plot(train_errors, '-o')
    # plt.plot(eval_errors, '-o')
    # plt.xlabel('epoch')
    #plt.ylabel('losses')
    # plt.legend(['Train losses', 'Tr errors', 'Valid errors'])
    # plt.title('Train vs Valid Error/Losses')
    #plt.show()

    # plot_accuracy(train_accu, eval_accu)
    print('End of model 1.')


def plot_accuracy(train_accu, eval_accu):
    plt.plot(train_accu, '-o')
    plt.plot(eval_accu, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.show()


def plot_predicted_points(device, model, validationloader, dev_set, subtitle, hyperparams_descr):
    # todo: plot lines for labels and one for predicted by model
    # inverse transform of the RobustScaler
    # sort the labels (x axis is for the index 1-n), plot values as y
    # (nb: must sort the whole validation dataset, including the features)
    # plot corresponding values predicted by model
    # do the same (separate subplot) for the training data
    model.eval()
    actual_labels_normalized = None
    predicted_labels_normalized = None

    with torch.no_grad():
        for data in tqdm(validationloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            # print('type(labels): ', type(labels), ', size: ', labels.size())  # torch tensor, torch.Size([64, 1])
            # print('type(outputs): ', type(outputs), ', size: ', outputs.size())

            if actual_labels_normalized is None:
                actual_labels_normalized = labels
                predicted_labels_normalized = outputs
            else:
                actual_labels_normalized = torch.cat((actual_labels_normalized, labels))
                predicted_labels_normalized = torch.cat((predicted_labels_normalized, outputs))

    if dev_set.labels_scaler is not None:
        actual_labels = dev_set.labels_scaler.inverse_transform(actual_labels_normalized.cpu().numpy()).tolist()
        predicted_labels = dev_set.labels_scaler.inverse_transform(predicted_labels_normalized.cpu().numpy()).tolist()
    else:
        actual_labels = actual_labels_normalized.tolist()
        predicted_labels = predicted_labels_normalized.tolist()

    #sort
    actual_labels, predicted_labels = (list(t) for t in zip(*sorted(zip(actual_labels, predicted_labels))))

    f, (ax2) = plt.subplots(1, 1, sharey=False)
    plot_hyperparams_descr(ax2, hyperparams_descr)
    l2 = ax2.scatter(range(len(actual_labels)), actual_labels, color='blue')  #  l2, = ax2.plot(actual_labels, '-o', color='blue')
    l3 = ax2.scatter(range(len(predicted_labels)), predicted_labels, color='orange')  # '-o',
    ax2.set_title('actual and predicted labels' + subtitle)
    ax2.set_xlabel('sample index')
    ax2.set_ylabel('sample label value')
    plt.legend([l2, l3], ["actual_labels", "predicted_labels"])
    plt.show()


def get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes,
                           activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                           loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep):

    root_dir = os.getcwd()
    print('root dir:', root_dir)
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/', assets_subdir, filename)

    hyperparams_descr = ("Dataset: " + filename
                         + "\n" + "Scaler: " + str(features_scaler)
                         + "\n" + "Model: " + model_descr
                         + "\n" + "Layers, nodes: " + str(hidden_layer_sizes)
                         + "\n" + "Activation_fun: " + str(activation_fun)[0:5]
                         + "\n" + "Mini_batch_size: " + str(mini_batch_size)
                         + "\n" + "Learning_rate: " + str(learning_rate)
                         + "\n" + "Momentum: " + str(momentum)
                         + "\n" + "Adaptive_learning_rate: " + adaptive_learning_rate
                         + "\n" + "Error fn: " + str(loss_fn)
                         + "\n" + "L2 lambda: " + str(l2_lambda)
                         )

    return hidden_layer_sizes, activation_fun, mini_batch_size, learning_rate, momentum, \
           adaptive_learning_rate, loss_fn, l2_lambda, hyperparams_descr, file_abs_path, \
           sep, dtype1, dtype2, header, col_names, features_scaler


def get_config_for_airfoil_dataset():
    dtype1 = {'A': np.float32, 'B': np.float32, 'C': np.float32, 'D': np.float32, 'E': np.float32, 'F': np.float32}
    dtype2 = {'A': np.int32, 'B': np.int32}
    col_names = ["A", "B", "C", "D", "E", "F"]
    header = None
    features_scaler = RobustScaler
    
    # hidden_layer_sizes = (3*32*32, 512, 512, 10)
    hidden_layer_sizes_airflow = (5, 120, 1)  # (5, 10, 10, 10, 1)
    activation_fun = nn.RReLU()  # other options: nn.ReLU(), nn.Tanh, ..
    mini_batch_size = 64
    learning_rate = 0.012
    momentum = 0.9
    adaptive_learning_rate = 'constant'
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()  # BCELoss
    l2_lambda = 0.0014  # 0.003  # 0.005  # 0.01  # 0.001
    model_descr = 'NN MLP regressor (fully connected)'

    assets_subdir = 'airfoil'
    filename = 'airfoil_self_noise.dat.csv'
    sep = '\t'
    return get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes_airflow,
                           activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                           loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep)


def get_config_for_winequality_dataset():

    # todo: support more than one target column
    # todo: include here the parameter for test/validation split ratio
    dtype1 = np.float32
    dtype2 = None
    col_names = None
    header = 0
    features_scaler = RobustScaler  # MaxAbsScaler  # MinMaxScaler  # Normalizer  # StandardScaler  #
    
    # hidden_layer_sizes = (3*32*32, 512, 512, 10)
    hidden_layer_sizes = (11, 120, 1)  # (5, 10, 10, 10, 1)
    activation_fun = nn.RReLU()  # other options: nn.ReLU(), nn.Tanh, ..
    mini_batch_size = 64
    learning_rate = 0.0012
    momentum = 0.1
    adaptive_learning_rate = 'constant'
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()  # BCELoss
    l2_lambda = 0.0014  # 0.003  # 0.005  # 0.01  # 0.001
    model_descr = 'NN MLP regressor (fully connected)'

    assets_subdir = 'wine-quality'
    filename = 'winequality-white.csv'
    sep = ';'
    return get_config_for_dataset(dtype1, dtype2, col_names, header, features_scaler, hidden_layer_sizes,
                           activation_fun, mini_batch_size, learning_rate, momentum, adaptive_learning_rate,
                           loss_fn, l2_lambda, model_descr, filename, assets_subdir, sep)


def main():
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    print('setting cpus/gpus..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    print('getting data..')
    hidden_layer_sizes, activation_fun, mini_batch_size, learning_rate, momentum, \
    adaptive_learning_rate, loss_fn, l2_lambda, hyperparams_descr, file_abs_path, \
    sep, dtype1, dtype2, header, col_names, features_scaler = get_config_for_winequality_dataset()

    print('file abs path:', file_abs_path)

    # todo: preprocessing, visualize data: boxplots, ..gaussians for each feature and output dim

    train_set_loader, validation_set_loader, train_set, validation_set, dev_set \
        = get_dev_data_from_file(mini_batch_size, file_abs_path, dtype1, sep, dtype2, header, col_names, features_scaler)

    print('train_and_validate_NN_model..')
    model, epochs_sequence, train_losses, train_errors, train_accu, eval_errors, eval_accu, \
    train_errors_variable_by_batches \
        = train_and_validate_NN_model(device, hidden_layer_sizes, activation_fun,
                                learning_rate, momentum, l2_lambda,
                                loss_fn, train_set_loader, validation_set_loader,
                                      adaptive_learning_rate=adaptive_learning_rate)

    plot_learning_curves(epochs_sequence, train_losses, train_errors, eval_errors, train_errors_variable_by_batches,
                         hyperparams_descr)
    plot_predicted_points(device, model, train_set_loader, dev_set, ' (training set)', hyperparams_descr)
    plot_predicted_points(device, model, validation_set_loader, dev_set, ' (validation set)', hyperparams_descr)


if __name__ == '__main__':
    main()
