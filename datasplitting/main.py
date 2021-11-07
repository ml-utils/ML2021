from preprocessing import Preprocessing
from feature_analysis import FeaturesAnalysis

if __name__ == '__main__':
    # build_train_and_save_model()
    # load_model()
    # lemon_squeeze_cross_validation()

    dataset_config = Preprocessing.get_dataset_configuration('airfoil')
    _, inX, outy = Preprocessing.load_the_dataset(dataset_config)
    k = 5
    folds = Preprocessing.get_random_k_folds(k, inX, outy, dataset_config)
    # todo: only use the training folds for the correlation matrix
    # todo: add merge folds method, to merge all the training folds into one

    # FeaturesAnalysis.plot_correlation_matrix2(inX, outy, dataset_config)
    FeaturesAnalysis.plot_boxplots(inX, outy, dataset_config)
