import sklearn
from preprocessing import Preprocessing


class FeaturesAnalysis:
    @staticmethod
    def chi2(X, y, config):
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
        chi2, p_values = sklearn.feature_selection.chi2(X, y)
        f_statistic, p_values = sklearn.feature_selection.f_regression(X, y)
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py

    @staticmethod
    def get_r_regression_coefficients(X, y):
        correlation_coefficients_by_y = []
        num_y_dimensions = y.shape[1]
        for i in range(num_y_dimensions):
            y_dimension = y[:, i:(i+2)]
            correlation_coefficients_by_y.append(sklearn.feature_selection.r_regression(X, y_dimension))
        return correlation_coefficients_by_y

    @staticmethod
    def plot_correlation_matrix(X, y, config):
        import matplotlib.pyplot as plt
        import pandas as pd
        dataframe = pd.DataFrame(Preprocessing.concatenate_dataset(X, y, config), dtype='float') # dataframe = dataframe.astype('float')
        corr = dataframe.corr()
        plt.matshow(corr)
        plt.show()