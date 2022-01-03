import os
from collections import OrderedDict
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
        self.flatten = nn.Flatten()  # what does this do?

        hidden_layer_sizes_as_list = NeuralNetwork.__get_layers(hidden_layer_sizes, activation_fun)
        self.linear_relu_stack = nn.Sequential(OrderedDict(hidden_layer_sizes_as_list))

    @staticmethod
    def __get_layers(hidden_layer_sizes, activation_fun):
        layers = []
        last_layer_idx = len(hidden_layer_sizes) - 2
        print('last_layer_idx: ', last_layer_idx)
        for layer_idx in range(len(hidden_layer_sizes) - 1):
            print('layer: ', hidden_layer_sizes[layer_idx], ', ', hidden_layer_sizes[layer_idx + 1])
            linear_layer = ('linear' + str(layer_idx), nn.Linear(hidden_layer_sizes[layer_idx],
                                                                      hidden_layer_sizes[layer_idx + 1]))
            layers.append(linear_layer)
            is_last_layer = layer_idx >= last_layer_idx
            if not is_last_layer:
                layers.append((str(activation_fun)[0:5] + str(layer_idx), activation_fun))
        return layers

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_in_training_mode(self):
        super(NeuralNetwork, self).train()


class FeaturesDataset(Dataset):
    def __init__(self, filename, rows_count=None, cols_count=None):
        col_names = ["A", "B", "C", "D", "E", "F"]
        dtype = {'A': np.float32, 'B': np.float32, 'C': np.float32, 'D': np.float32, 'E': np.float32, 'F': np.float32}
        file_out = pd.read_csv(filename, sep='\t', header=None, names=col_names, dtype=dtype)
        # print(file_out.to_string())
        print(file_out.info())
        print(file_out.head())

        file_out = file_out.astype(dtype={'A': np.int32, 'B': np.int32})
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
        print(y[:5])

        # todo: linear base expansion
        # add x^2 ..for each input

        #Feature scaling
        self.features_scaler = RobustScaler()  # MaxAbsScaler()  # MinMaxScaler()  # Normalizer()  # StandardScaler()  #
        x_train = self.features_scaler.fit_transform(x)
        self.labels_scaler = RobustScaler()  # MaxAbsScaler()  # MinMaxScaler()  # Normalizer()  # StandardScaler()  #
        y_reshaped = y.reshape(-1, 1)
        print('y shape:', y.shape, ', y_reshaped type: ', type(y_reshaped), ', shape: ', y_reshaped.shape)  # numpy.ndarray
        print('y_reshaped head:')
        print(y_reshaped[:5])
        y_train = self.labels_scaler.fit_transform(y_reshaped)  # = y

        #converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


def get_L2_norm_for_regularization(model):
    return sum(p.pow(2.0).sum()
        for p in model.parameters())


def train(epoch, trainloader, device, model, loss_fn, optimizer, l2_lambda, train_accu, train_losses, train_errors):
    print('\nEpoch : %d' % epoch)

    model.set_in_training_mode()

    running_loss = 0
    running_error = 0
    correct = 0
    total = 0

    for data in tqdm(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        # print('outputs this batch: ')
        # print(outputs.tolist())
        # print('labels this batch: ')
        # print(labels.tolist())

        # print("outputs type:", outputs.type())  # outputs type: torch.cuda.FloatTensor
        # print("labels type:", labels.type())  # labels type: torch.cuda.FloatTensor
        # outputs = outputs.to(device, dtype=torch.float64)
        # labels = labels.to(device, dtype=torch.float64)

        error = loss_fn(outputs, labels)
        # Replaces pow(2.0) with abs() for L1 regularization
        l2_norm = get_L2_norm_for_regularization(model)
        loss = error + (l2_lambda * l2_norm)

        # todo: check how it's done (backprop, weights update, gradient descent, derivative of loss)
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # backward pass
        optimizer.step()

        running_loss += loss.item()
        running_error += error.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_error = running_error / len(trainloader)
    # If you would like to calculate the loss for each epoch,
    # divide the running_loss by the number of batches and append it to train_losses in each epoch.
    accu = 100. * correct / total

    train_accu.append(accu)
    train_losses.append(train_loss)
    train_errors.append(train_error)
    print('Train Loss: %.3f | Train Error: %.3f | Accuracy: %.3f' % (train_loss, train_error, accu))


def evaluate(epoch, testloader, device, model, loss_fn, eval_errors, eval_accu):
    model.eval()

    running_error = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            error = loss_fn(outputs, labels)

            running_error += error.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_error = running_error / len(testloader)
    accu = 100. * correct / total

    eval_errors.append(test_error)
    eval_accu.append(accu)

    print('Test Error: %.3f | Accuracy: %.3f' % (test_error, accu))


def get_dev_data_from_file(mini_batch_size, filename, rows_count=None, cols_count=None):
    from torch.utils.data import DataLoader
    print('getting data from ', filename, '..')
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dev_set = FeaturesDataset(filename, rows_count, cols_count)
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
                             loss_fn, trainloader, validationloader):
    # todo, passing functions as parameter, enable choice btw optimizers, normalization function, ..

    print('creating model and sending to cpu/gpu..')
    model = NeuralNetwork(hidden_layer_sizes, activation_fun).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # model1
    print('running model 1:')
    train_losses = []
    train_errors = []
    train_accu = []
    eval_errors = []
    eval_accu = []

    epochs_count = 25
    epochs_sequence = range(1, epochs_count + 1)
    for epoch in epochs_sequence:
        train(epoch, trainloader, device, model, loss_fn, optimizer, l2_lambda, train_accu, train_losses, train_errors)
        evaluate(epoch, validationloader, device, model, loss_fn, eval_errors, eval_accu)

    return model, epochs_sequence, train_losses, train_errors, train_accu, eval_errors, eval_accu


def plot_learning_curves(epochs_sequence, train_losses, train_errors, eval_errors):

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    l1, = ax1.plot(epochs_sequence, train_losses, '-o', color='purple')
    ax1.set_title('train_losses')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('losses')

    l2, = ax2.plot(epochs_sequence, train_errors, '-o', color='blue')
    l3, = ax2.plot(epochs_sequence, eval_errors, '-o', color='orange')
    ax2.set_title('train and validation error')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('errors')
    plt.legend([l1, l2, l3], ["train_losses", "train_errors", "eval_errors"])
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


def plot_predicted_points(device, model, validationloader, dev_set, subtitle):
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

    actual_labels = dev_set.labels_scaler.inverse_transform(actual_labels_normalized.cpu().numpy()).tolist()
    predicted_labels = dev_set.labels_scaler.inverse_transform(predicted_labels_normalized.cpu().numpy()).tolist()

    #sort
    actual_labels, predicted_labels = (list(t) for t in zip(*sorted(zip(actual_labels, predicted_labels))))

    f, (ax2) = plt.subplots(1, 1, sharey=False)
    l2 = ax2.scatter(range(len(actual_labels)), actual_labels, color='blue')  #  l2, = ax2.plot(actual_labels, '-o', color='blue')
    l3 = ax2.scatter(range(len(predicted_labels)), predicted_labels, color='orange')  # '-o',
    ax2.set_title('actual and predicted labels' + subtitle)
    ax2.set_xlabel('sample index')
    ax2.set_ylabel('sample label value')
    plt.legend([l2, l3], ["actual_labels", "predicted_labels"])
    plt.show()


def main():
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    # hidden_layer_sizes = (3*32*32, 512, 512, 10)
    hidden_layer_sizes_airflow = (5, 120, 1)  # (5, 10, 10, 10, 1)
    activation_fun = nn.RReLU()  # nn.ReLU() nn.Tanh
    mini_batch_size = 64
    learning_rate = 0.012
    momentum = 0.9
    l2_lambda = 0.0014  # 0.003  # 0.005  # 0.01  # 0.001
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()  # BCELoss

    print('setting cpus/gpus..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    print('getting data..')
    root_dir = os.getcwd()
    print('root dir:', root_dir)
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/airfoil/airfoil_self_noise.dat.csv')
    print('file abs path:', file_abs_path)

    # todo: preprocessing, visualize data: boxplots, ..gaussians for each feature and output dim

    train_set_loader, validation_set_loader, train_set, validation_set, dev_set \
        = get_dev_data_from_file(mini_batch_size, file_abs_path, 1502, 5)

    print('train_and_validate_NN_model..')
    model, epochs_sequence, train_losses, train_errors, train_accu, eval_errors, eval_accu \
        = train_and_validate_NN_model(device, hidden_layer_sizes_airflow, activation_fun,
                                learning_rate, momentum, l2_lambda,
                                loss_fn, train_set_loader, validation_set_loader)

    plot_learning_curves(epochs_sequence, train_losses, train_errors, eval_errors)
    plot_predicted_points(device, model, train_set_loader, dev_set, ' (training set)')
    plot_predicted_points(device, model, validation_set_loader, dev_set, ' (validation set)')


if __name__ == '__main__':
    main()
