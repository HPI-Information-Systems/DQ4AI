import argparse
import json
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to a non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from pathlib import Path
from tqdm import tqdm

# plot styling config parameters
FONT_SIZE = 49
FONT_SIZE_TICKS = 43
sns.set_style('whitegrid')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.3
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE_TICKS
plt.rcParams["ytick.labelsize"] = FONT_SIZE_TICKS
matplotlib.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['lines.linewidth'] = 4

LABELS = {
    'Decision_Tree_Regression': 'DT',
    'Linear_Regression': 'LR',
    'MLP_Regression': 'MLP-1',
    'pytorch_MLP_Regression': 'MLP-1',
    'Ridge_Regression': 'RR',
    'Random_Forest_Regression': 'RF',
    'MLP_Regression_5_hidden_layers': 'MLP-5',
    'Pytorch_MLP_Regression_5_hidden_layers': 'MLP-5',
    'MLP_Regression_10_hidden_layers': 'MLP-10',
    'Pytorch_MLP_Regression_10_hidden_layers': 'MLP-10',
    'GradientBoosting_Regression': 'GB',
    'TabNet_Regression': 'TN'
}

ALGORITHM_ORDER = [
    'DT',
    'RF',
    'LR',
    'RR',
    'GB',
    'MLP-1',
    'MLP-5',
    'MLP-10',
    'TN',
    'Original DQ'
]

COLORS = {
    'Decision_Tree_Regression': '#1f77b4',
    'Linear_Regression': '#2ca02c',
    'MLP_Regression': '#d62728',
    'pytorch_MLP_Regression': '#d62728',
    'Ridge_Regression': '#7f7f7f',
    'Random_Forest_Regression': '#e377c2',
    'MLP_Regression_5_hidden_layers': '#8c564b',
    'Pytorch_MLP_Regression_5_hidden_layers': '#8c564b',
    'MLP_Regression_10_hidden_layers': '#9467bd',
    'Pytorch_MLP_Regression_10_hidden_layers': '#9467bd',
    'GradientBoosting_Regression': '#ff7f0e',
    'TabNet_Regression': '#bcbd22'
}

MARKERS = {
    'Decision_Tree_Regression': 'o',
    'Linear_Regression': 'v',
    'MLP_Regression': '^',
    'pytorch_MLP_Regression': '^',
    'Ridge_Regression': 's',
    'Random_Forest_Regression': 'P',
    'Pytorch_MLP_Regression_5_hidden_layers': 'D',
    'MLP_Regression_5_hidden_layers': 'X',
    'MLP_Regression_10_hidden_layers': 'X',
    'Pytorch_MLP_Regression_10_hidden_layers': 'X',
    'GradientBoosting_Regression': 'p',
    'TabNet_Regression': '|'
}

READABLE_POLLUTERS = {'ConsistentRepresentationPolluter': 'Consistent Representation',
                      'ConsistentRepresentation_5': 'Consistent Representation',
                      'ConsistentRepresentation_4': 'Consistent Representation',
                      'Completeness': 'Completeness',
                      'TargetAccuracy': 'Target Accuracy',
                      'FeatureAccuracy': 'Feature Accuracy',
                      'Uniqueness_6': 'Uniqueness',
                      'Uniqueness_7': 'Uniqueness',
                      'ClassBalance': 'Class Balance'}

# algorithm baseline quality to add
DATASET_BASE_QUALITY = {
    'ClassBalance': {
        'house_prices_prepared.csv': 0.7992307692307692,
        'vw_prepared.csv': 0.8938263284129043,
        'imdb_prepared.csv': 0.5441269841269841,
        'covid_data_pre_processed_regression.csv': 1.0
    },
    'Completeness': {
        'house_prices_prepared.csv': 0.9976070747355644,
        'vw_prepared.csv': 1.0,
        'imdb_prepared.csv': 0.9979559486067078,
        'covid_data_pre_processed_regression.csv': 1.0
    },
    'ConsistentRepresentation': {
        'house_prices_prepared.csv': 1.0,
        'vw_prepared.csv': 1.0,
        'imdb_prepared.csv': 1.0,
        'covid_data_pre_processed_regression.csv': 1.0
    },
    'FeatureAccuracy': {
        'house_prices_prepared.csv': 1.0,
        'vw_prepared.csv': 1.0,
        'imdb_prepared.csv': 1.0,
        'covid_data_pre_processed_regression.csv': 1.0
    },
    'TargetAccuracy': {
        'house_prices_prepared.csv': 1.0,
        'vw_prepared.csv': 1.0,
        'imdb_prepared.csv': 1.0,
        'covid_data_pre_processed_regression.csv': 1.0
    },
    'Uniqueness': {
        'house_prices_prepared.csv': 1.0,
        'vw_prepared.csv': 0.9825811559778306,
        'imdb_prepared.csv': 0.8134178905206942,
        'covid_data_pre_processed_regression.csv': 1.0
    }
}


def weighted_average_quality(quality_measure, dataset_name, metadata_file='metadata.json'):
    """
    Calculation of the weighted average quality by number of categorical and numerical columns

    :param quality_measure: Quality measure name
    :param dataset_name: Dataset name
    :param metadata_file: Name of the metadata file + path
    """
    with open('../metadata.json', 'r') as f:
        meta = json.load(f)

    ds_meta = meta[dataset_name]

    n_num_cols = len(ds_meta['numerical_cols'])
    n_cat_cols = len(ds_meta['categorical_cols'])
    n_cols = n_num_cols + n_cat_cols

    weighted_measure = n_cat_cols / n_cols * (0 if quality_measure[0] is None else quality_measure[0]) \
                       + n_num_cols / n_cols * \
                       (0 if quality_measure[1] is None else quality_measure[1])

    return round(weighted_measure, 4)


def get_result_dataframe(algo_results, n_seeds):
    """
    Aggregation of results for one algorithm into a dataframe
    """

    res_df = pd.DataFrame(columns=[
        'quality',
        'per_seed_config_idx',
        '$R^2$',
    ])
    per_seed_configs = len(algo_results) // n_seeds

    for i, res in enumerate(algo_results):
        if (any('Pytorch_MLP' in key for key in res.keys())) or (any('pytorch_MLP' in key for key in res.keys())):
            def find_mlp_key(my_dict):
                for key in my_dict.keys():
                    if key.startswith('Pytorch_MLP') or key.startswith('pytorch_MLP'):
                        return key
                return None

            pytorch_mlp_key = find_mlp_key(res)
            res['r2_score'] = res[pytorch_mlp_key]
            res['mean_squared_error'] = res[pytorch_mlp_key]
        if "ConsistentRepresentation" in res['polluter']:
            polluter_config = res['polluter_config']
            # During result preprocessing, an instance of the baseline config was added
            # for each random seed with a numeric index-like value contained in the
            # pollution config string so that those strings were distinguishable as keys.
            if "{}" in polluter_config:
                polluter_config = '_'.join(polluter_config.split('_')[:-1])
            polluter_config = ast.literal_eval(polluter_config)
            if 'percentage_polluted_rows' not in res_df:
                res_df['percentage_polluted_rows'] = None
            res_df = pd.concat(
                [res_df, pd.DataFrame({'quality': res['quality'], 'per_seed_config_idx': i % per_seed_configs,
                                       '$R^2$': res['r2_score'],
                                       'percentage_polluted_rows': polluter_config['percentage_polluted_rows']},
                                      index=[0])], ignore_index=True)
        else:
            res_df = pd.concat(
                [res_df, pd.DataFrame({'quality': res['quality'], 'per_seed_config_idx': i % per_seed_configs,
                                       '$R^2$': res['r2_score']}, index=[0])], ignore_index=True)

    if "ConsistentRepresentation" in algo_results[0]['polluter']:
        res_df['percentage_polluted_rows'] = res_df['percentage_polluted_rows'].astype(float)
        res_df = res_df.groupby('per_seed_config_idx').agg(np.mean)
        res_df = res_df.sort_values(by='percentage_polluted_rows', ascending=True)

        quality_diff = res_df['quality'].diff()
        first_increase_index = quality_diff[quality_diff > 0].index[0]
        res_df = res_df.iloc[:first_increase_index]
    else:
        res_df = res_df.groupby("per_seed_config_idx").agg(np.mean)
        res_df = res_df.sort_values(by='quality', ascending=False)
    #if len(res_df) > 11:
        # keep rows where per_seed_config_idx % 2 == 0
    res_df = res_df[res_df.index % 2 == 0]
    return res_df


def plot_result_dataframe(algorithm_to_metrics, dataset_name, polluter_name, scenario_name, plots_dir, polluters_seen,
                          baseline_perf):
    """
    Plotting a given dataframe and exporting the file as png
    """
    metric_figures = dict()
    y_max = dict()
    y_min = dict()
    algo_name = None
    polluter_title = polluter_name.split("Polluter")[0]
    if "ConsistentRepresentation" in polluter_title or "Uniqueness" in polluter_title:
        polluter_title = f"{polluter_title}_{polluters_seen}"

    for i, (algo, res_df) in enumerate(algorithm_to_metrics.items()):
        algo_name = algo

        for c in res_df.columns.difference(['quality', 'percentage_polluted_rows']):
            if metric_figures.get(c) is None:
                fig, ax = plt.subplots(figsize=(15, 10))
                point_texts = []
                metric_figures[c] = (fig, ax, point_texts)
                y_max[c] = res_df[c].max() + 0.1 * res_df[c].max()
                y_min[c] = res_df[c].min() - 0.1
                # We only plot the polluter name without a possible specific configuration parameter
                x_label = READABLE_POLLUTERS[polluter_title] if polluter_title in READABLE_POLLUTERS else polluter_title
                ax.set_xlabel(x_label + ' Quality')
                ax.set_ylabel(f'{c}')
                if isinstance(res_df.index[0], str):
                    ax.set_xticks(range(len(res_df)))
                    ax.set_xticklabels(
                        res_df['quality'].to_list(), rotation=90)
                else:
                    ax.set_xlim((1.1, -0.1))
                    ax.set_xticks(np.arange(1.0, -0.1, -0.2))
            else:
                fig, ax, point_texts = metric_figures[c]
                y_max[c] = max(y_max[c], res_df[c].max() + 0.1)
                y_min[c] = min(y_min[c], res_df[c].min() - 0.1)

            baseline_res_df = baseline_perf[algo_name]
            p = ax.plot(res_df['quality'].to_list(),
                        res_df[c], marker=MARKERS[algo_name], label=LABELS[algo_name], color=COLORS[algo_name], markersize=15)

            # plot the original dataset performance as dashed line for class balance and uniqueness
            if "ClassBalance" in polluter_title or "Uniqueness" in polluter_title:
                ax.plot(baseline_res_df['quality'].copy(),
                        baseline_res_df[c].copy(), linestyle="dashed", color=COLORS[algo_name], alpha=0.8, ms=17)

            # plot some pollution percentages for Consistent Representation, only for the Random Forest line because
            # this line is mostly at the top
            if "ConsistentRepresentation" in polluter_title and "Random_Forest" in algo_name:
                res_df_percentage_points = res_df[res_df['percentage_polluted_rows'].isin([
                    0, 0.5, 0.8, 1])]
                for x, y, txt in zip(res_df_percentage_points['quality'], res_df_percentage_points[c],
                                     res_df_percentage_points['percentage_polluted_rows']):
                    if 0 <= y <= 1:
                        point_texts.append(
                            plt.text(x, y, txt, fontsize=FONT_SIZE_TICKS))

    for c, (fig, ax, point_texts) in metric_figures.items():
        ax.set_ylim(-0.1, 1)
        ax.set_yticks(np.arange(1.0, -0.1, -0.2))
        fig.tight_layout()
        ax.vlines(
            DATASET_BASE_QUALITY[polluter_title.split("_")[0]][dataset_name],
            -0.1,
            1,
            color='black',
            linestyles='dotted',
            label='Original DQ'
        )

        # special handling for consistent representation plots
        if "ConsistentRepresentation" in polluter_title:
            adjust_text(point_texts, only_move={
                'points': 'y', 'text': 'y', 'objects': 'y', 'explode': 'y', 'static': 'y', 'pull': 'y'}, autoalign='y')

        # Change plotting order
        handles, labels = ax.get_legend_handles_labels()
        order_index = {algo: i for i, algo in enumerate(ALGORITHM_ORDER)}
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: order_index[hl[1]])
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        for spine in ax.spines.values():
            spine.set_color("lightgray")

        figpath = Path(
            plots_dir / f'{polluter_title}/{polluter_title}_{dataset_name.split(".")[0]}_{scenario_name}.pdf')
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, bbox_inches='tight')

        fig.legend(sorted_handles, sorted_labels, loc=(0.18, 0.217), fontsize=35, ncol=3, markerscale=1.5)
        figpath = Path(
            plots_dir / f'{polluter_title}/{polluter_title}_{dataset_name.split(".")[0]}_{scenario_name}_legend.pdf')

        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, bbox_inches='tight')
        plt.close(fig)


def get_header(l):
    """ Helper function generating a LaTeX table header for a list of data qualities. Implemented for technical report.
    """
    header = '    Quality'
    ctr = 1
    for qual in l:
        header += " & \\multicolumn{1}{r" + \
                  ('' if ctr == len(l) else '|') + "}{" + f'{qual:.2f}' + "}"
        ctr += 1

    header += '\\\\\n    \\hline\n'

    return header


def get_row(algo, l):
    """ Helper function generating a LaTeX table row for an algorithm and a list of floats (clustering metrics per data
    quality). Implemented for technical report.
    """
    row = f'    {" ".join(algo.split("_")[:-1])}'
    for val in l:
        row += f' & {val:.4f}'
    row += '\\\\\n'
    return row


def get_table(df_dict, dataset, polluter, scenario):
    """ Helper function generating a LaTeX table for a given dictionary algorithm -> results dataframe.
    Implemented for technical report.
    """
    table = '\\begin{table*}[b]\n'
    table += '    \\caption{Performance of regression algorithms for' + \
             f' {polluter.split("Polluter")[0]} ' + 'dimension and ' + \
             f'{dataset.split(".")[0].capitalize()}' + \
             ' dataset in scenario ' + f'{scenario}' + '.}\n'
    table += '    \\begin{tabular}{l|r|r|r|r|r|r|r|r|r|r|r}\n    & \\multicolumn{11}{c}{$R^2$} \\\\ \n'

    has_head = False
    for algo, df in df_dict.items():
        if not has_head:
            table += get_header(df['quality'].to_list())
            has_head = True
        table += get_row(algo, df['$R^2$'])

    table += '    \\end{tabular}\n\\end{table*}'
    return table


def generate_plots(args):
    """
    Plot generation function

    :param args: CLI args
    """
    with open(args.results, 'r') as f:
        results = json.load(f)
    results = results[next(iter(results))]
    if args.ds_to_consider is not None:
        results = {args.ds_to_consider: results[args.ds_to_consider]}

    for dataset, ds_res in results.items():
        algorithms = set()
        polluters_seen = 0
        for polluter, pollution_res in tqdm(ds_res.items()):
            scenarios = set()
            metrics = dict()
            for run_config, run_res in pollution_res.items():
                quality_point = run_res['quality']
                if isinstance(quality_point, list):
                    quality_point = round(weighted_average_quality(
                        quality_point, dataset), 4) if polluter == 'FeatureAccuracyPolluter' else round(
                        quality_point[1], 4)
                else:
                    quality_point = round(quality_point, 4)

                for scenario, scenario_res in run_res.items():
                    if scenario == 'quality':
                        continue
                    scenarios.add(scenario)
                    for algo, algo_res in scenario_res.items():
                        algorithms.add(algo)
                        if metrics.get(scenario) is None:
                            metrics[scenario] = dict()
                        if metrics[scenario].get(algo) is None:
                            metrics[scenario][algo] = list()
                        metrics[scenario][algo].append(
                            {'quality': quality_point, 'polluter': polluter, 'polluter_config': run_config, **algo_res})
            # generate plot for each scenario
            for scenario in scenarios:
                alg_to_metrics = dict()
                baseline_perf = dict()
                for algo in algorithms:
                    if not args.combine_plots:
                        alg_to_metrics = dict()
                        baseline_perf = dict()
                    alg_to_metrics[algo] = get_result_dataframe(
                        metrics[scenario][algo], args.n_seeds)
                    baseline_perf[algo] = get_result_dataframe(
                        metrics["train_original_test_original"][algo], args.n_seeds)
                    if not args.combine_plots:
                        plot_result_dataframe(
                            alg_to_metrics, dataset, polluter, scenario, args.plots_dir, polluters_seen, baseline_perf)
                if args.combine_plots:
                    alg_to_metrics = dict(sorted(alg_to_metrics.items()))
                    plot_result_dataframe(
                        alg_to_metrics, dataset, polluter, scenario, args.plots_dir, polluters_seen, baseline_perf)
                    # comment in line below to print a LaTeX-formatted table of the original values
                    # print(get_table(alg_to_metrics, dataset, polluter, scenario))

            polluters_seen += 1


if __name__ == '__main__':
    """
    Start script to add described CLI parameters for plot generation
    """

    parser = argparse.ArgumentParser(
        'Script reading a results.json and plotting the metrics recorded.')

    parser.add_argument('--results', required=True, type=Path,
                        help='Path to the results.json to plot from.')
    parser.add_argument('--ds-to-consider', required=False,
                        type=str, help='Which dataset to consider for plotting.')
    parser.add_argument('--plots-dir', required=True, type=Path,
                        help='Base directory for the plot file structure.')
    parser.add_argument('--n-seeds', required=True, type=int,
                        help='Number of different random seeds that were used to create the results.')
    parser.add_argument('--combine-plots', action='store_true',
                        help='Set flag with --combine-plots for combining all algorithms in one plot.')

    generate_plots(parser.parse_args())
