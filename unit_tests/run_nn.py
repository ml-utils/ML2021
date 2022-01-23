# Standard library imports
import os
from datetime import datetime

# Third party imports
import numpy as np
from numpy.random import default_rng

# Local application imports
from lib_models.utils import get_hyperparams_descr
from nn import NeuralNet
from preprocessing import load_and_preprocess_monk_dataset, get_cup_dev_set_fold_splits, get_cup_dataset_from_file, \
    get_cupdata_split_fixed_size, get_cup_blind_set, split_cup_dataset, CUP_CFG, MLCUP21_BLINDSET_out_datatypes


def run_nn_cup(which_cv_fold=1):
    from datasets_cfgs import CUP_CUSTOM_NET_CFG
    from hp_names import CFG
    '''
        trains our custom neural network on the cup dataset
    :return:
    '''
    cv_num_plits = CUP_CUSTOM_NET_CFG[CFG.CV_NUM_SPLITS]
    root_dir = os.getcwd()
    input_file_path = os.path.join(root_dir, '..\\datasplitting\\assets\\ml-cup21-internal_splits\\dev_split.csv')
    training_split, validation_split = get_cup_dev_set_fold_splits(input_file_path,
                                                                                 cv_num_plits=cv_num_plits,
                                                                                 which_fold=which_cv_fold)
    error_fn = 'MSE'  # MSE, MEE
    task = 'regression'
    adaptive_lr = 'SGD constant lr'
    out_dim = 2

    net_shape = [10, 10, 10, out_dim]
    lr = 0.01  # 0.01  # 1e-2 # 1e-4
    alpha_momentum = 0.04  # 5e-2
    lambda_reg = 0.0005  # 0.0005  # 0.001  # 0.005 # 5e-7
    mini_batch_size = 32  # 80 # 32
    activation = 'tanh'  # 'sigmoid' # 'tanh'

    stopping_threshold = 0.001  #0.00001  # 0.01
    max_epochs = 2000
    patience = 75  # 75
    early_stopping = 'MSE2_val'  # 'EuclNormGrad'  # 'MSE2_val'

    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task=task, error=error_fn, verbose=True)

    print(f'doing {training_split.shape[0]} samples for training, and {validation_split.shape[0]} for validation')
    test_net.load_training(training_split, out=out_dim)
    test_net.load_validation(validation_split, out=out_dim)

    hyperparams_descr = get_hyperparams_descr('CUP_2021dev', str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer=adaptive_lr)
    print(f'{hyperparams_descr}')
    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print(f'initial validation_error = {test_net.validate_net()[0]:0.3f}')
    test_net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs, stopping=early_stopping, patience=patience,
                            verbose=False, hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print(f'final validation_error = {test_net.validate_net(error_func=error_fn)[0]:0.3f}')
    error_fn2 = 'MEE'
    print(f'final vl MEE error = {test_net.validate_net(error_func=error_fn2)[0]:0.3f}')


def run_nn_only_classification():
    root_dir = os.getcwd()
    filename = 'monks-1.train'
    file_path = os.path.join(root_dir, '..\\datasplitting\\assets\\monk\\', filename)

    data = load_and_preprocess_monk_dataset(file_path)

    # import seaborn as sns
    # sns.pairplot(pd.DataFrame(data), diag_kind='kde')
    # plt.show()

    rng = default_rng()
    rng.shuffle(data)
    example_number = data.shape[0]
    train_ratio = 0.813
    split_id = int(np.round(example_number * train_ratio))
    print(f'doing {split_id} samples for training, and {example_number - split_id} for validation')

    print('dataset head after shuffling: ')
    print(data[:5])
    task = 'classification'
    activation = 'tanh'  # 'sigmoid' # 'tanh'
    net_shape = [17, 5, 1]
    mini_batch_size = 1
    lr = 0.1  # 1e-2
    alpha_momentum = 0.01  # 5e-2
    lambda_reg = 0.00001  # 0.001  # 0.005
    stopping_threshold = 0.01  #0.00001  # 0.01
    max_epochs = 2000
    patience = 50
    early_stopping = 'MSE2_val'  # 'EuclNormGrad'  # 'MSE2_val'
    error_fn = 'MSE'

    test_net = NeuralNet(activation, net_shape, eta=lr, alpha=alpha_momentum, lamda=lambda_reg, mb=mini_batch_size,
                         task=task, verbose=True)

    test_net.load_training(data[:split_id], 1, do_normalization=False)
    test_net.load_validation(data[split_id:], 1)

    hyperparams_descr = get_hyperparams_descr(filename, str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer='SGD constant lr')
    print(f'running training with hyperparams: {hyperparams_descr}')
    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print(f'initial validation_error = {test_net.validate_net()[0]:0.3f}')

    test_net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs, stopping=early_stopping, patience=patience,
                            verbose=False, hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    final_validation_error, accuracy, vl_misc_rate = test_net.validate_net()
    print(f'final validation_error = {final_validation_error:0.3f}')
    print(f'final validation accuracy = {accuracy:0.3f}')

    # todo: plot actual vs predicted (as accuracy and as MSE smoothing function)


def command_line_training():

    from datasets_cfgs import CUP_CUSTOM_NET_CFG
    from hp_names import CFG
    # todo: collect training history to plot learning curve
    # run_nn_and_tf()
    # run_nn_only_regression()
    # run_nn_only_classification()

    if len(sys.argv) <= 1:
        run_nn_cup()
    elif len(sys.argv) == 2 and sys.argv[1].isdigit():
        cv_fold = sys.argv[1]
        cv_fold = int(cv_fold)
        # todo: check is cv_fold btw num folds
        if cv_fold > CUP_CUSTOM_NET_CFG[CFG.CV_NUM_SPLITS]:
            print(f'Error: the cv_fold specified ({cv_fold}) is greather then the number of folds {CUP_CUSTOM_NET_CFG[CFG.CV_NUM_SPLITS]}')
            sys.exit()

        print(f'Command line arg: doing fold n. {cv_fold}')
        run_nn_cup(which_cv_fold=cv_fold)
    else:
        print(f'Unregognized command line args: {sys.argv}')


def command_line_load_and_assess_net(verbose=False):
    # todo:
    # load net from file, with final weight values
    # this is just for testing, need to retrain net with these params, but normalization and denormalization in MEE
    best_weights_path = '..\\report_grid_searches\\chosen_model\\20220123-153347\\latest'
    dev_dataset_path = '..\\datasplitting\\assets\\ml-cup21-internal_splits\\dev_split.csv'
    internal_testset_path = '..\\datasplitting\\assets\\ml-cup21-internal_splits\\test_split.csv'
    print(f'loading net final/best weights from {best_weights_path}')
    print(f'dev set file: {dev_dataset_path}')
    print(f'internal_testset_path file: {internal_testset_path}')

    error_fn = 'MSE'  # MSE, MEE
    task = 'regression'
    out_dim = 2

    activation = 'tanh'  # 'sigmoid' # 'tanh'
    mini_batch_size = 50  # 80 # 32
    lr = 0.1  # 0.01  # 1e-2 # 1e-4
    alpha_momentum = 0.04  # 5e-2
    lambda_reg = 0.0005  # 0.0005  # 0.001  # 0.005 # 5e-7

    cv_num_plits = 3
    which_cv_fold = 1

    net = NeuralNet.load_net(best_weights_path, activation, eta=lr, alpha=alpha_momentum, lamda=lambda_reg,
                             mb=mini_batch_size, task=task, error=error_fn)  # , **kwargs

    # load dataset on which to perform the assessment/evaluation, without normalization
    training_split, validation_split = get_cup_dev_set_fold_splits(dev_dataset_path,
                                                                   cv_num_plits=cv_num_plits,
                                                                   which_fold=which_cv_fold)
    test_set = get_cup_dataset_from_file(internal_testset_path)

    net.load_training(training_split, out_dim)
    error_MEE_tr = net.evaluate_original_error(set=training_split, error_fn='MEE')
    error_MEE_vl = net.evaluate_original_error(set=validation_split, error_fn='MEE')
    error_MSE_tr = net.evaluate_original_error(set=training_split, error_fn='MSE')
    error_MSE_vl = net.evaluate_original_error(set=validation_split, error_fn='MSE')
    print(f'final MEE errors (no normalization): TR: {error_MEE_tr}, VL: {error_MEE_vl}, '  # , TS: {}
          f'final MSE erorors (no normalization): TR: {error_MSE_tr}, VL: {error_MSE_vl}, ')

    error_MEE_testset = net.evaluate_original_error(set=test_set, error_fn='MEE')
    error_MSE_testset = net.evaluate_original_error(set=test_set, error_fn='MSE')

    print(f'The assessment of the model on the test set is: MEE: {error_MEE_testset}, '
          f'MSE: {error_MSE_testset}')
    print(f'TR: MEE: {error_MEE_tr} MSE: {error_MSE_tr}')
    print(f'VL: MEE: {error_MEE_vl} MSE: {error_MSE_vl}')
    print(f'TS: MEE: {error_MEE_testset} MSE: {error_MSE_testset}')

    if verbose:
        net.load_validation(validation_split, out_dim)
        norml_error_MEE_vl, _, _ = net.validate_net(error_func='MEE')
        norml_error_MSE_vl, _, _ = net.validate_net(error_func='MSE')
        print(f'for comparison/double check, the vl errors with normalization are: '
              f'MEE: {norml_error_MEE_vl}, MSE: {norml_error_MSE_vl}')


def generate_cup_retraining_splits():
    split_cup_dataset(dev_split_idx=1200, pt1_name='-tr', pt2_name='-vl',
                      filename='ML-CUP21-shuffled-retraining.csv', shuffle=False)


def command_line_retraining():
    # load a net, with its INITIAL weights,
    print('retraining..')
    initial_weights_path = '..\\report_grid_searches\\chosen_model\\20220123-153347\\epoch_0'
    dataset_path = '..\\datasplitting\\assets\\ml-cup21-internal_splits\\ML-CUP21-shuffled-retraining.csv'
    print(f'loading net initial weights from {initial_weights_path}')
    print(f'dataset_path: {dataset_path}')

    error_fn = 'MSE'  # MSE, MEE
    task = 'regression'
    out_dim = 2
    net_shape = [10, 10, 10, out_dim]
    activation = 'tanh'  # 'sigmoid' # 'tanh'
    mini_batch_size = 50  # 80 # 32
    lr = 0.1  # 0.01  # 1e-2 # 1e-4
    alpha_momentum = 0.04  # 5e-2
    lambda_reg = 0.0005  # 0.0005  # 0.001  # 0.005 # 5e-7
    adaptive_lr = 'SGD constant lr'
    stopping_threshold = 0.001  #0.00001  # 0.01
    max_epochs = 2000
    early_stopping = 'MSE2_val'
    patience = 75
    net = NeuralNet.load_net(initial_weights_path, activation, eta=lr, alpha=alpha_momentum, lamda=lambda_reg,
                             mb=mini_batch_size, task=task, error=error_fn)  # , **kwargs

    # a new tr vl split of the data
    training_size = 1200
    vl_size = 277  # 1200 for TR out of total 1477

    training_split, validation_split = get_cupdata_split_fixed_size(dataset_path, training_size=training_size,
                                                                    vl_size=vl_size)

    # load new tr and vl sets, and start training again
    net.load_training(training_split, out_dim)
    net.load_validation(validation_split, out=out_dim)

    hyperparams_descr = get_hyperparams_descr('CUP_2021dev', str(net_shape), activation, mini_batch_size,
                                              error_fn=error_fn, l2_lambda=lambda_reg, momentum=alpha_momentum,
                                              learning_rate=lr, optimizer=adaptive_lr)
    print(f'{hyperparams_descr}')
    start_time = datetime.now()
    print('net initialized at {}'.format(start_time))
    print(f'initial validation_error = {net.validate_net()[0]:0.3f}')
    net.batch_training(threshold=stopping_threshold, max_epochs=max_epochs, stopping=early_stopping,
                            patience=patience,
                            verbose=False, hyperparams_for_plot=hyperparams_descr)
    end_time = datetime.now()
    print('training completed at {} ({} elapsed)'.format(end_time, end_time - start_time))
    print(f'final validation_error = {net.validate_net(error_func=error_fn)[0]:0.3f}')
    error_fn2 = 'MEE'
    print(f'final vl MEE error = {net.validate_net(error_func=error_fn2)[0]:0.3f}')


def cmd_evaluate_cup_blindset():
    # todo: load blindset, ids included
    # load the final model (the retrained one) with its final weights
    # call net.evaluate(self, entry), passing one entry at a time, but without id
    # collect results, keeping their ids, and save to file
    print(f'Generating blind test predictions..')
    retrained_weights_path = '..\\report_grid_searches\\retrained_model\\20220123-181909\\latest'
    bind_dataset_path = '..\\datasplitting\\assets\\ml-cup21-internal_splits\\ML-CUP21-TS-blindset.csv'
    print(f'loading net retrained_weights from {retrained_weights_path}')
    print(f'bind_dataset_path file: {bind_dataset_path}')


    error_fn = 'MSE'  # MSE, MEE
    task = 'regression'
    out_dim = 2

    activation = 'tanh'  # 'sigmoid' # 'tanh'
    mini_batch_size = 50  # 80 # 32
    lr = 0.1  # 0.01  # 1e-2 # 1e-4
    alpha_momentum = 0.04  # 5e-2
    lambda_reg = 0.0005  # 0.0005  # 0.001  # 0.005 # 5e-7

    cv_num_plits = 3
    which_cv_fold = 1

    net = NeuralNet.load_net(retrained_weights_path, activation, eta=lr, alpha=alpha_momentum, lamda=lambda_reg,
                             mb=mini_batch_size, task=task, error=error_fn)  # , **kwargs

    # this is used only to regenerate the scale and shift vectors used in normalization
    workdir = '..\\datasplitting\\assets\\ml-cup21-internal_splits\\'
    retraining_dataset_path = workdir + 'ML-CUP21-shuffled-retraining.csv'
    training_size = 1200
    vl_size = 277  # 1200 for TR out of total 1477
    training_split, _ = get_cupdata_split_fixed_size(retraining_dataset_path, training_size=training_size,
                                                                    vl_size=vl_size)
    net.load_training(training_split, out_dim)

    blind_set = get_cup_blind_set(bind_dataset_path)
    # todo, check that the dimension is 1 hid col and 10 features columns
    print(f'Blind set size: {blind_set.shape}')

    # to remove an entry id..
    #         ids = dataset[:, 0]
    #         no_ids = dataset[:, config['x_begin_col_idx']:(config['y_end_col_idx'] + 1)]

    predicted_dataset = np.asarray([0, 0, 0])  # np.empty((0, out_dim+1))
    for entry in blind_set:
        entry_id = entry[:1]
        entry_features = entry[-10:]
        predicted = net.evaluate(entry_features)
        id_with_predicted = np.hstack((entry_id, predicted))

        # print(f'entry_id: {entry_id},\n entry_features: {entry_features},\n predicted: {predicted},\n '
        #       f'id_with_predicted: {id_with_predicted}, id_with_predicted.shape: {id_with_predicted.shape}')
        # predicted_dataset = np.append(predicted_dataset, id_with_predicted, axis=0)
        predicted_dataset = np.vstack((predicted_dataset, id_with_predicted))

    # remove first dummpy element:
    predicted_dataset = predicted_dataset[1:]
    print(f'predicted_dataset: {predicted_dataset}')
    print(f'predicted_dataset.shape: {predicted_dataset.shape}')

    # todo: save to csv file with appropriate format for project
    team_name = 'NoCovidNoParty'
    out_file_name = team_name + '_ML-CUP21-TS.csv'
    out_file_path = workdir + out_file_name
    print(f'saving predicted csv file to {out_file_path}')
    np.savetxt(out_file_path, predicted_dataset, fmt=MLCUP21_BLINDSET_out_datatypes, delimiter=CUP_CFG['sep'])


if __name__ == '__main__':
    import sys

    cmd_evaluate_cup_blindset()
    #   command_line_retraining()
    #   command_line_training()
    #   command_line_load_and_assess_net()



