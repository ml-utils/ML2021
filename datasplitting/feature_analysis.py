import math

import sklearn
from datasplits import DataSplits
import matplotlib.pyplot as plt


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
        corr = FeaturesAnalysis.get_correlation_matrix(X, y, config)
        plt.matshow(corr)
        plt.show()

    @staticmethod
    def plot_correlation_matrix2(X, y, config):
        import seaborn as sns
        # https://www.kaggle.com/sudhirnl7/logistic-regression-with-stratifiedkfold
        corr = FeaturesAnalysis.get_correlation_matrix(X, y, config)
        plt.figure()  # figsize=(12, 6)
        sns.heatmap(corr, cmap='Set1', annot=True)
        plt.show()

    @staticmethod
    def get_correlation_matrix(X, y, config):
        import pandas as pd
        dataframe = pd.DataFrame(DataSplits.concatenate_dataset(X, y, config), dtype='float')
        # dataframe = dataframe.astype('float')
        return dataframe.corr()

    @staticmethod
    def plot_boxplots(X, y, config):
        concatenated_dataset = DataSplits.concatenate_dataset(X, y, config)

        dataset_col_num = concatenated_dataset.shape[1]
        plot_side_lenght = FeaturesAnalysis.get_square_side_leght(dataset_col_num)
        fig = plt.figure()
        for i in range(dataset_col_num):
            ax = fig.add_subplot(plot_side_lenght, plot_side_lenght, i+1)
            column_data = concatenated_dataset[:, i]
            ax.boxplot(column_data, patch_artist=True, notch='True')
        plt.show()

    @staticmethod
    def get_square_side_leght(count):
        sqrt = math.sqrt(count)
        side_lenght = math.ceil(sqrt)
        return side_lenght