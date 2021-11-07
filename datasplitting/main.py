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
    # todo: add merge folds method, to merge all the training folds into one

    # FeaturesAnalysis.plot_correlation_matrix2(inX, outy, dataset_config)
    FeaturesAnalysis.plot_boxplots(inX, outy, dataset_config)

    # todo: print box plots of the same variable along different splits; see also standard deviation
    # todo: plot ..histogram/scatterplot distribution of each variable, to see if gaussia, power law, etc.

    # todo, draw some learning curves (regression, scikit library nn, ..)
    # todo gridsearch cross validation
