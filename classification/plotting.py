import pandas as pd
from matplotlib import pyplot as plt
from numpy import arange, append
from adjustText import adjust_text
from os import path, mkdir

from sympy.codegen.ast import continue_

from classification.preprocessing import group_and_aggregate_df, \
    extract_performances_for_highest_qualities_consistent_representation
from util import get_majority_baseline_performance, get_ratio_baseline_performance

LINE_WIDTH = 3
MARKER_SIZE = 13.5
TICK_FONT_SIZE = 32
AXIS_TITLE_FONT_SIZE = 36
LABEL_SIZE = TICK_FONT_SIZE
MARKERS = {
    'Decision Tree Classification': 'o',
    'Logistic Regression Classification': 'v',
    'Multilayer Perceptron Classification': '^',
    'Pytorch Multilayer Perceptron Classification': '^',
    'Support Vector Machine Classification': 's',
    'Pytorch Support Vector Machine Classification': 's',
    'k-Nearest Neighbors Classification': 'P',
    'Pytorch k-Nearest Neighbors Classification': 'P',
    'Multilayer Perceptron Classification (5 hidden layers)': 'D',
    'Pytorch Multilayer Perceptron Classification (5 hidden layers)': 'X',
    'Multilayer Perceptron Classification (10 hidden layers)': 'X',
    'Pytorch Multilayer Perceptron Classification (10 hidden layers)': 'X',
    'Gradient Boosting Classification': 'p',
    'TabNet Classification': '|'
}
LABELS = {
    'Decision Tree Classification': 'Decision Tree',
    'Logistic Regression Classification': 'Logistic Regression',
    'Multilayer Perceptron Classification': 'Multilayer Perceptron (1)',
    'Pytorch Multilayer Perceptron Classification': 'Multilayer Perceptron (1)',
    'Support Vector Machine Classification': 'Support Vector Machine',
    'Pytorch Support Vector Machine Classification': 'Support Vector Machine',
    'k-Nearest Neighbors Classification': 'k-Nearest Neighbors',
    'Pytorch k-Nearest Neighbors Classification': 'k-Nearest Neighbors',
    'Multilayer Perceptron Classification (5 hidden layers)': 'Multilayer Perceptron (5)',
    'Pytorch Multilayer Perceptron Classification (5 hidden layers)': 'Multilayer Perceptron (5)',
    'Multilayer Perceptron Classification (10 hidden layers)': 'Multilayer Perceptron (10)',
    'Pytorch Multilayer Perceptron Classification (10 hidden layers)': 'Multilayer Perceptron (10)',
    'Gradient Boosting Classification': 'Gradient Boosting',
    'TabNet Classification': 'TabNet'
}

COLORS = {
    'Decision Tree Classification': '#1f77b4',
    'Logistic Regression Classification': '#2ca02c',
    'Multilayer Perceptron Classification': '#d62728',
    'Pytorch Multilayer Perceptron Classification': '#d62728',
    'Support Vector Machine Classification': '#e377c2',
    'Pytorch Support Vector Machine Classification': '#e377c2',
    'k-Nearest Neighbors Classification': '#7f7f7f',
    'Pytorch k-Nearest Neighbors Classification': '#7f7f7f',
    'Multilayer Perceptron Classification (5 hidden layers)': '#8c564b',
    'Pytorch Multilayer Perceptron Classification (5 hidden layers)': '#8c564b',
    'Multilayer Perceptron Classification (10 hidden layers)': '#9467bd',
    'Pytorch Multilayer Perceptron Classification (10 hidden layers)': '#9467bd',
    'Gradient Boosting Classification': '#ff7f0e',
    'TabNet Classification': '#bcbd22'
}

POLLUTION_LABELS = ['0.0', '0.5', '0.8', '1.0']

READABLE_POLLUTERS = {'ConsistentRepresentationPolluter': 'Consistent Representation',
                      'ConsistentRepresentationPolluter_five': 'Consistent Representation (5)',
                      'ConsistentRepresentationPolluter_two': 'Consistent Representation (2)',
                      'CompletenessPolluter': 'Completeness',
                      'UniquenessPolluter_uniform': 'Uniqueness (uniform)',
                      'UniquenessPolluter_normal': 'Uniqueness (normal)',
                      'TargetAccuracyPolluter': 'Target Accuracy',
                      'FeatureAccuracyPolluter': 'Feature Accuracy',
                      'ClassBalancePolluter': 'Class Balance'}


DATASET_BASE_QUALITY = {
    'ClassBalancePolluter': {
        'SouthGermanCredit.csv': 0.4285714285714286,
        'TelcoCustomerChurn.csv': 0.3619988378849506,
        'cmc.data': 0.5294117647058824,
        'covid_data_pre_processed.csv': 0.07861007240872098
    },
    'CompletenessPolluter': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 1.0,
        'cmc.data': 1.0,
        'covid_data_pre_processed.csv': 1.0
    },
    'ConsistentRepresentationPolluter_five': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 1.0,
        'cmc.data': 1.0,
        'covid_data_pre_processed.csv': 1.0
    },
    'ConsistentRepresentationPolluter_two': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 1.0,
        'cmc.data': 1.0,
        'covid_data_pre_processed.csv': 1.0
    },
    'FeatureAccuracyPolluter': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 1.0,
        'cmc.data': 1.0,
        'covid_data_pre_processed.csv': 1.0
    },
    'TargetAccuracyPolluter': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 1.0,
        'cmc.data': 1.0,
        'covid_data_pre_processed.csv': 1.0
    },
    'UniquenessPolluter_normal': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 0.9968709998577727,
        'cmc.data': 0.967391304347826,
        'covid_data_pre_processed.csv': 0.1751429789367615
    },
    'UniquenessPolluter_uniform': {
        'SouthGermanCredit.csv': 1.0,
        'TelcoCustomerChurn.csv': 0.9968709998577727,
        'cmc.data': 0.967391304347826,
        'covid_data_pre_processed.csv': 0.1751429789367615
    }
}

def plot_ordinary_performances(ax, aggregated_plotting_df, performance_measure, with_std):
    if with_std:
        yerr = 'f1-score_std'
    else:
        yerr = 'accuracy_std'

    for algo in aggregated_plotting_df.algorithm.unique():
        marker = MARKERS[algo]
        if with_std:
            aggregated_plotting_df[aggregated_plotting_df.algorithm == algo].plot(x='quality', y=performance_measure,
                                                                                  ax=ax, yerr=yerr,
                                                                                  capsize=4, alpha=0.6,
                                                                                  marker=marker,
                                                                                  markersize=MARKER_SIZE,
                                                                                  linewidth=LINE_WIDTH,
                                                                                  label=LABELS[algo],
                                                                                  color=COLORS[algo])
        else:
            aggregated_plotting_df[aggregated_plotting_df.algorithm == algo].plot(x='quality', y=performance_measure,
                                                                                  ax=ax, marker=marker,
                                                                                  markersize=MARKER_SIZE,
                                                                                  linewidth=LINE_WIDTH,
                                                                                  label=LABELS[algo],
                                                                                  color=COLORS[algo])


def plot_consistent_representation_performances(ax, aggregated_plotting_df, dataset, performance_measure,
                                                with_std, performances_highest_qualities=None):
    """
    The results produced by ConsistentRepresentationPolluter need special treatment to plot them.
    Adds the performance of the models on the original dataset manually.
    Collects the pollution levels of the data points as text labels and returns them.
    """
    texts = []
    for algorithm in aggregated_plotting_df.algorithm.unique():
        marker = MARKERS[algorithm]

        x = list(aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm].quality_mean)
        y = list(aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm][performance_measure])
        pollution_levels = list(aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm].pollution_level)
        pollution_levels.insert(0, '0.0')

        if x[0] != 1.0 and performances_highest_qualities:
            x.insert(0, 1.0)
            highest_performance = performances_highest_qualities[dataset][algorithm][performance_measure]

            y.insert(0, highest_performance)

        quality_diff = pd.Series(x).diff()  # Calculate row-by-row increase
        if (quality_diff > 0).any():  # Check if an increase exists
            first_increase_index = quality_diff[quality_diff > 0].index[0]
            # Slice up to first increase
            x = x[:first_increase_index]
            y = y[:first_increase_index]
            pollution_levels = pollution_levels[:first_increase_index]

        # Plot results with or without error bar
        if with_std:
            std = 'accuracy_std' if 'accuracy' in performance_measure else 'f1-score_std'
            y_err = aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm][std].fillna(0)
            ax.plot(x, y, yerr=y_err, marker=marker, markersize=MARKER_SIZE, capsize=4, alpha=0.6,
                    linewidth=LINE_WIDTH, label=LABELS[algorithm], color=COLORS[algorithm])
        else:
            ax.plot(x, y, marker=marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=LABELS[algorithm], color=COLORS[algorithm])

        # Collect pollution levels of the data points
        for x, y, txt in zip(x, y, pollution_levels):
            if txt in POLLUTION_LABELS and algorithm == 'Decision Tree Classification':
                texts.append(plt.text(x, y, txt, fontsize=LABEL_SIZE))
    return texts


def plot_performances(df, dataset, polluter, scenario, baselines, with_std=False, use_f1=False, save_path=None, show_legend=False):
    """
    :param df:          DataFrame
    :param dataset:     string, name of the dataset
    :param polluter:    string, name of the polluter
    :param scenario:    string, name of the scenario
    :param baselines:   dict, see preprocessing.get_baselines(), ../util.get_majority_baseline_performance and
                        ../util.get_ratio_baseline_performance for details
    :param with_std:    boolean, whether to plot the results with standard deviation as error bars
    :param use_f1:      boolean, whether to use f1-score as the performance metric or accuracy (False)
    :param save_path:   string, path where to save the plots. If it's not given the plots are displayed instead.
    :param show_legend: boolean, whether to show the legend or not
    """
    # Group and aggregate results
    grouped_df = group_and_aggregate_df(df, polluter)

    # Pick the baselines based on the performance measure
    if use_f1:
        baseline_majority = baselines[dataset]['f1-score']['baseline_majority']
        baseline_ratio = baselines[dataset]['f1-score']['baseline_ratio']
    else:
        baseline_majority = baselines[dataset]['accuracy']['baseline_majority']
        baseline_ratio = baselines[dataset]['accuracy']['baseline_ratio']

    # Filter the desired results
    if 'polluter_config' in grouped_df.keys():
        aggregated_plotting_df = grouped_df[
            (grouped_df.dataset == dataset) & (grouped_df.polluter == polluter) & (
                    grouped_df.scenario == scenario)].drop(columns='polluter_config')
    else:
        aggregated_plotting_df = grouped_df[
            (grouped_df.dataset == dataset) & (grouped_df.polluter == polluter) & (
                    grouped_df.scenario == scenario)]

    fig, ax = plt.subplots(figsize=(15, 10))
    if use_f1:
        performance_measure = 'f1-score_mean'
    else:
        performance_measure = 'accuracy_mean'

    # Plot the actual results and baselines
    consistentRepresentationPolluters = ['ConsistentRepresentationPolluter_two',
                                         'ConsistentRepresentationPolluter_five']
    if polluter in consistentRepresentationPolluters:
        # As the ConsistentRepresentation results do not start with 100% quality, they have to be extracted from the
        # 'train_clean_test_clean' scenario and to be added to the plot data later on
        performances_highest_qualities = extract_performances_for_highest_qualities_consistent_representation(df)
        texts = plot_consistent_representation_performances(ax, aggregated_plotting_df, dataset,
                                                            performance_measure, with_std,
                                                            performances_highest_qualities)

        # Get end of baselines and plot baselines
        quality = 'quality_mean' if polluter in consistentRepresentationPolluters else 'quality'
        min_performance_on_smallest_quality = 1.0
        for algorithm in aggregated_plotting_df.algorithm.unique():
            current_min_performance = aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm][
                quality].min()
            if min_performance_on_smallest_quality > current_min_performance:
                min_performance_on_smallest_quality = current_min_performance
        baseline_x = arange(1, min_performance_on_smallest_quality, -0.1)
        baseline_x = append(baseline_x, min_performance_on_smallest_quality)
        baseline_y_majority = [baseline_majority] * len(baseline_x)
        baseline_y_ratio = [baseline_ratio] * len(baseline_x)
        plt.plot(baseline_x, baseline_y_majority, linestyle='--', color='black', linewidth=LINE_WIDTH, label='Majority class baseline')
        plt.plot(baseline_x, baseline_y_ratio, linestyle='-.', color='black', linewidth=LINE_WIDTH, label='Class ratio baseline')

        # Plot some levels of pollution
        adjust_text(texts, only_move={'text': 'y', 'static': 'y', 'explode': 'y', 'pull': 'y'},
                    autoalign='y')

    else:
        print(dataset, polluter, scenario)
        if 'quality' not in aggregated_plotting_df.columns:
            print('Quality not in columns')
            print(aggregated_plotting_df.columns)
            return
        # Plot performance lines and baselines
        plot_ordinary_performances(ax, aggregated_plotting_df, performance_measure, with_std)
        plt.plot(aggregated_plotting_df.quality.unique(),
                 [baseline_majority] * aggregated_plotting_df.quality.unique().shape[0], linestyle='--', color='black',
                 linewidth=LINE_WIDTH, label='Majority class baseline')
        plt.plot(aggregated_plotting_df.quality.unique(),
                 [baseline_ratio] * aggregated_plotting_df.quality.unique().shape[0], linestyle='-.', color='black',
                 linewidth=LINE_WIDTH, label='Class ratio baseline')

    if polluter in ['UniquenessPolluter_normal', 'UniquenessPolluter_uniform']:
        min_quality = aggregated_plotting_df.quality.min()
        max_quality = aggregated_plotting_df.quality.max()
        ax.set_xticks(arange(max_quality, min_quality - 0.1, -0.2))
    else:
        ax.set_xticks(arange(1.0, 0.0, -0.2))

    ax.invert_xaxis()
    if use_f1:
        ax.set_ylabel('F1-Score', fontsize=AXIS_TITLE_FONT_SIZE)
    else:
        ax.set_ylabel('Accuracy', fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlabel(f"{READABLE_POLLUTERS[polluter]} Quality", fontsize=AXIS_TITLE_FONT_SIZE)
    print('dasdsadsa')
    ax.vlines(
        DATASET_BASE_QUALITY[polluter][dataset],
        -0.1,
        1,
        color='black',
        linestyles='dotted',
        label='Original DQ'
    )
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(linewidth=LINE_WIDTH)
    ax.set_ylim([0.15, 0.85])

    custom_order = ['Majority class baseline', 'Class ratio baseline', 'Original DQ', 'Decision Tree', 'Logistic Regression',
                    'Multilayer Perceptron (1)', 'Multilayer Perceptron (5)', 'Multilayer Perceptron (10)', 'TabNet',
                    'Gradient Boosting', 'Support Vector Machine', 'k-Nearest Neighbors']
    handles, labels = ax.get_legend_handles_labels()
    label_handle_dict = dict(zip(labels, handles))
    ordered_handles = [label_handle_dict[label] for label in custom_order if label in label_handle_dict]
    if show_legend:
        # rename legend items
        new_names = ['Majority class baseline', 'Class ratio baseline', 'Original DQ', 'DT', 'LogR',
                     'MLP-1', 'MLP-5', 'MLP-10', 'TN', 'GB', 'SVM', 'KNN']
        #ax.legend(ordered_handles, new_names, fontsize=23)
        ax.legend(ordered_handles, new_names, loc='lower center',  ncol=3,fontsize=26)
        #ax.legend(ordered_handles, custom_order, fontsize=23)
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    for spine in ax.spines.values():
        spine.set_color("lightgray")
    if save_path:
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # replace png with pdf
        save_path = save_path.replace('.png', '.pdf')

        # save_path = save_path.replace('ConsistentRepresentationPolluter', 'CR')
        # save_path = save_path.replace('UniquenessPolluter', 'Uniqu')
        # save_path = save_path.replace('ClassBalancePolluter', 'CB')
        # save_path = save_path.replace('TargetAccuracyPolluter', 'TA')
        # save_path = save_path.replace('CompletenessPolluter', 'Completeness')
        # save_path = save_path.replace('FeatureAccuracyPolluter', 'F_Acc')
        #
        # save_path = save_path.replace('legend', 'leg')
        # save_path = save_path.replace('polluted', 'pol')

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
