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

    # FeaturesAnalysis.plot_correlation_matrix2(inX, outy, dataset_config)
    FeaturesAnalysis.plot_densities(inX, outy, dataset_config)

    # todo: print box plots of the same variable along different splits; see also standard deviation
    # (optional: scatterplots btw input variables and inputs)
    # (optional: measures to check if a variable is gaussian)

    # todo, draw some learning curves (regression, scikit library nn, ..)
    # todo gridsearch cross validation
