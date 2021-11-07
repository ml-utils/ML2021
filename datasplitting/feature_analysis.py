import math

import sklearn
from datasplits import DataSplits
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


class PlotType(Enum):
    BOXPLOT = 1
    HIST = 2
    DIST = 3


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
        FeaturesAnalysis.plot_univariate(X, y, config, PlotType.BOXPLOT)

    @staticmethod
    def plot_histograms(X, y, config):
        FeaturesAnalysis.plot_univariate(X, y, config, PlotType.HIST)

    @staticmethod
    def plot_densities(X, y, config):
        FeaturesAnalysis.plot_univariate(X, y, config, PlotType.DIST)

    @staticmethod
    def plot_univariate(X, y, config, plot_type):
        concatenated_dataset = DataSplits.concatenate_dataset(X, y, config)

        dataset_col_num = concatenated_dataset.shape[1]
        plot_side_lenght = FeaturesAnalysis.get_square_side_leght(dataset_col_num)
        fig = plt.figure()
        for i in range(dataset_col_num):
            ax = fig.add_subplot(plot_side_lenght, plot_side_lenght, i+1)
            column_data = concatenated_dataset[:, i]
            # NB switch/match construct is from python 3.10 only
            if plot_type == PlotType.BOXPLOT:
                ax.boxplot(column_data, patch_artist=True, notch='True')
            elif plot_type == PlotType.HIST:
                binwidth = 10
                bins = int(len(column_data) / binwidth)
                ax.hist(column_data, bins=bins, density=True)
            elif plot_type == PlotType.DIST:
                binwidth = 10
                bins = int(len(column_data) / binwidth)
                sns.distplot(column_data, ax=ax, hist=True)  # , bins=bins
                # sns.displot(data=column_data, ax=ax, kind='hist')  # kind='kde'
        plt.show()

    @staticmethod
    def get_square_side_leght(count):
        sqrt = math.sqrt(count)
        side_lenght = math.ceil(sqrt)
        return side_lenght