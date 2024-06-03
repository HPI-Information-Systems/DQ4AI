from matplotlib import pyplot as plt
from numpy import arange, append
from adjustText import adjust_text
from os import path, mkdir
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
    'Support Vector Machine Classification': 's',
    'k-Nearest Neighbors Classification': 'P',
    'Multilayer Perceptron Classification (5 hidden layers)': 'D',
    'Multilayer Perceptron Classification (10 hidden layers)': 'X',
    'Gradient Boosting Classification': 'p'
}
LABELS = {
    'Decision Tree Classification': 'DT',
    'Logistic Regression Classification': 'LogR',
    'Multilayer Perceptron Classification': 'MLP-1',
    'Support Vector Machine Classification': 'SVM',
    'k-Nearest Neighbors Classification': 'KNN',
    'Multilayer Perceptron Classification (5 hidden layers)': 'MLP-5',
    'Multilayer Perceptron Classification (10 hidden layers)': 'MLP-10',
    'Gradient Boosting Classification': 'GB'
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
                                                                                  label=LABELS[algo])
        else:
            aggregated_plotting_df[aggregated_plotting_df.algorithm == algo].plot(x='quality', y=performance_measure,
                                                                                  ax=ax, marker=marker,
                                                                                  markersize=MARKER_SIZE,
                                                                                  linewidth=LINE_WIDTH,
                                                                                  label=LABELS[algo])


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
        # Plot results with or without error bar
        if with_std:
            std = 'accuracy_std' if 'accuracy' in performance_measure else 'f1-score_std'
            y_err = aggregated_plotting_df[aggregated_plotting_df.algorithm == algorithm][std].fillna(0)
            ax.plot(x, y, yerr=y_err, marker=marker, markersize=MARKER_SIZE, capsize=4, alpha=0.6,
                    linewidth=LINE_WIDTH, label=LABELS[algorithm])
        else:
            ax.plot(x, y, marker=marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=LABELS[algorithm])

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
        # Plot performance lines and baselines
        plot_ordinary_performances(ax, aggregated_plotting_df, performance_measure, with_std)
        plt.plot(aggregated_plotting_df.quality.unique(),
                 [baseline_majority] * aggregated_plotting_df.quality.unique().shape[0], linestyle='--', color='black',
                 linewidth=LINE_WIDTH, label='Majority class baseline')
        plt.plot(aggregated_plotting_df.quality.unique(),
                 [baseline_ratio] * aggregated_plotting_df.quality.unique().shape[0], linestyle='-.', color='black',
                 linewidth=LINE_WIDTH, label='Class ratio baseline')

    ax.invert_xaxis()
    if use_f1:
        ax.set_ylabel('F1-Score', fontsize=AXIS_TITLE_FONT_SIZE)
    else:
        ax.set_ylabel('Accuracy', fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlabel(f"{READABLE_POLLUTERS[polluter]} Quality", fontsize=AXIS_TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(linewidth=LINE_WIDTH)
    ax.set_ylim([0.15, 0.85])

    custom_order = ['Majority class baseline', 'Class ratio baseline', 'DT', 'LogR',
                    'MLP-1', 'MLP-5', 'MLP-10',
                    'GB', 'SVM', 'KNN']
    handles, labels = ax.get_legend_handles_labels()
    label_handle_dict = dict(zip(labels, handles))
    ordered_handles = [label_handle_dict[label] for label in custom_order if label in label_handle_dict]
    if show_legend:
        ax.legend(ordered_handles, custom_order, fontsize=26, ncol=3)
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
