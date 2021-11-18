from datasplits import DataSplits
from feature_analysis import FeaturesAnalysis

if __name__ == '__main__':
    # build_train_and_save_model()
    # load_model()
    # lemon_squeeze_cross_validation()

    dataset_config = DataSplits.get_dataset_configuration('airfoil')
    _, inX, outy = DataSplits.load_the_dataset(dataset_config)
    k = 5
    folds = DataSplits.get_random_k_folds(k, inX, outy, dataset_config)
    # todo: only use the training folds for the correlation matrix
    # todo: use merged folds for the training set
    # todo: use merged folds to do univariate analysis of features

    # FeaturesAnalysis.plot_correlation_matrix2(inX, outy, dataset_config)
    FeaturesAnalysis.plot_densities(inX, outy, dataset_config)

    # todo: print box plots of the same variable along different splits; see also standard deviation
    # (optional: scatterplots btw input variables and inputs)
    # (optional: measures to check if a variable is gaussian)

    # todo, draw some learning curves (regression, scikit library nn, ..)
    # todo gridsearch cross validation

    #todo:
    # activation functions: add relu
    # feature engineering: add x^2 for each numerical input variable

    # steps (try first on a similar dataset for regression, like the airfoil dataset):

    # data split (CV design)
    # feature angineering: linear base expansion (add x^2 and possibly others)
    # feature analysis (ie correlation, distribution) and possibly selection
    # grid search for model selection (dev set)
    ## possible hyperparameters to select:
    ## n. of NN nodes/units and layers
    ## eta, gamma, lambda (learining rate, momentum, regularization)
    ## activation functions (relu, tanh)
    ## loss functions (L2 norm, L1 norm, ..)
    # evaluation (test set)
    # last model selection
    # last training on whole dataset, with the model selected at the previous step

    # possible models to compare:
    # NN with random weights and no training
    # some librariy NN
    # ..
