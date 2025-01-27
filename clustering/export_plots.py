""" Script used to visualize the contents of a preprocessed results.json file. For preprocessing of a raw results.json
(produced by running the ``run_experiments.py`` script) refer to the ``preprocess_results_json.py`` script.

Execution and configuration via CLI parameters and by changing the constants defined on top of the code.
"""

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

from matplotlib.lines import Line2D
from tqdm import tqdm

sns.set_style('whitegrid')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25


# Maps the polluter name as found in the results JSON to the name of the "pollution level" parameter from its configuration
# the "pollution level" parameter is a float which indicates the severity of pollution in the given polluter configuration
# this may have the semantic of "pct of samples polluted" or some other, arbitrary meaning
POLL_LEVEL_NAMES = {
    'ClassBalancePolluter': '_imbalance_level',
    'CompletenessPolluter': 'pollution_percentages',
    'ConsistentRepresentationPolluter': 'percentage_polluted_rows',
    'FeatureAccuracyPolluter': None,
    'TargetAccuracyPolluter': 'pollution_level',
    'UniquenessPolluter': 'duplicate_factor'
}


# Maps the clustering quality metric names used in the JSON to human readable representations to be used in plots
QUALITY_NAME_MAP = {
    'adj_mut_info': 'AMI Score',
    'adjusted rand score': 'Adjusted Rand Score',
    'mutual information score': 'Mutual Information Score',
    'n_cluster': 'Number of Clusters Identified',
    'norm_mut_info': 'Normalized Mutual Information Score',
    'rand score': 'Rand Score'
}


# Maps the polluter names as used in the results JSON (without the Polluter part) to human readable to be used in plots
POLLUTER_NAME_MAP = {
    'ClassBalance': 'Target Class Balance',
    'Completeness': 'Completeness',
    'ConsistentRepresentation': 'Consistent Representation',
    'FeatureAccuracy': 'Feature Accuracy',
    'TargetAccuracy': 'Target Accuracy',
    'Uniqueness': 'Uniqueness'
}


# Contains information about the quality of each unpolluted (constant) dataset per quality dimension (polluter)
# These values can be determined using the ``dataset_quality_calculation.py`` script and are constants used in plotting
DATASET_BASE_QUALITY = {
    'ClassBalancePolluter': {
            'letter.arff': 1.0,
            'covtype.csv': 1.0,
            'bank.csv': 1.0,
            'covid.csv': 0.4246734773050562
        },
    'CompletenessPolluter': {
            'letter.arff': 1.0,
            'covtype.csv': 1.0,
            'bank.csv': 0.9977179948781663,
            'covid.csv': 1.0
        },
    'ConsistentRepresentationPolluter': {
            'letter.arff': 1.0,
            'covtype.csv': 1.0,
            'bank.csv': 0.9500240878318416,
            'covid.csv': 1.0
        },
    'FeatureAccuracyPolluter': {
            'letter.arff': 1.0,
            'covtype.csv': 1.0,
            'bank.csv': 1.0,
            'covid.csv': 1.0
        },
    'TargetAccuracyPolluter': {
            'letter.arff': 1.0,
            'covtype.csv': 1.0,
            'bank.csv': 1.0,
            'covid.csv': 1.0
        },
    'UniquenessPolluter': {
        'letter.arff': 0.9665646213230401,
        'covtype.csv': 1.0,
        'bank.csv': 0.3065475396719563,
        'covid.csv': 0.5489979599183967
    }
}


# maps clustering quality metric name to the max value of the ylim to be used when plotting
# for all metrics that are not in this dictionary as keys, ylim is automatically determined
PLOT_YMAX = {
    'adj_mut_info': 0.525
}

# Contains information about the Adjusted Mutual Information Score each algorithm was able to achieve on the unpolluted
# (original) datasets. This information is obtained by modifying the ``run_experiments.py`` script to run without
# dataset pollution and then manual evaluation of the results.json file (averaging of values over all random seeds)
ORIGINAL_DATASET_AMI = {
    'bank.csv': {
        'Agglomerative': 0.15027446841622552,
        'Autoencoder': 0,
        'Gaussian Mixture': 0.13951410006757842,
        'OPTICS': 0.1121429171710097,
        'k-Prototypes': 0.022534711498458038
    },
    'covtype.csv': {
        'Agglomerative': 0.3524127329403385,
        'Autoencoder': 0.1222023806294503,
        'Gaussian Mixture': 0.3540054532526904,
        'OPTICS': 0.024424513222105104,
        'k-Prototypes': 0.19373985701526614
    },
    'letter.arff': {
        'Agglomerative': 0.41007827196187335,
        'Autoencoder': 0.03241023932951963,
        'Gaussian Mixture': 0.44311772631993573,
        'OPTICS': 0.36164281062433934,
        'k-Means': 0.3553878869166296
    },
    'covid.csv': {
        'Agglomerative': 0.00486242679203996281,
        'Autoencoder': 0.3752621170870515,
        'Gaussian Mixture': 0.89087565501779357,
        'OPTICS': 0.121125239774639233,
        'k-Means': 0.45099298911639113
    }
}


# maps algorithm name to color in order to ensure that always the same colors are used for the same algorithm
# k-Prototypes and k-Means are not meant to be plotted at the same time, so their colors may be similar without issue
ALGO_COLOR_MAP = {
    'Agglomerative': 'tab:red',
    'Autoencoder': 'tab:purple',
    'Gaussian Mixture': 'tab:green',
    'OPTICS': 'tab:orange',
    'k-Means': 'tab:blue',
    'k-Prototypes': 'tab:cyan'
}


# maps algorithm name to marker shape in order to ensure uniform usage of marker to distinguish data points
ALGO_MARKER_MAP = {
    'Agglomerative': 'o',
    'Autoencoder': 'v',
    'Gaussian Mixture': '^',
    'OPTICS': 's',
    'k-Means': 'P',
    'k-Prototypes': 'D'
}


def weighted_average_quality(quality_measure, dataset_name, metadata_file='metadata.json'):
    """ Assumes that <quality_measure> is a tuple. Loads the <metadata_file> and reads the section corresponding to the
    given <dataset_name>. Calculates the average of both quality measure values weighted by the number of
    categorical and numerical columns. It is assumed that the <quality_measure> tuple corresponds to
    (measure_cat_cols, measure_num_cols).
    """
    with open('../metadata.json', 'r') as f:
        meta = json.load(f)

    if dataset_name == 'covid.csv':
        temp_dataset_name = 'covid_data_pre_processed_clustering.csv'
        ds_meta = meta[temp_dataset_name]
    else:
        ds_meta = meta[dataset_name]

    n_num_cols = len(ds_meta['numerical_cols'])
    n_cat_cols = len(ds_meta['categorical_cols'])
    n_cols = n_num_cols + n_cat_cols

    weighted_measure = n_cat_cols / n_cols * (0 if quality_measure[0] is None else quality_measure[0]) \
                       + n_num_cols / n_cols * (0 if quality_measure[1] is None else quality_measure[1])

    return round(weighted_measure, 4)


def get_result_dataframe(mets, algo_name, polluter_name, n_seeds):
    """ Converts the given raw metrics dataframe to a results dataframe where results of multiple runs with different
    seeds are averaged. Resulting dataframe will have the dataset quality as index and all clustering quality metrics
    as well as the pollution level as columns. Resulting df will be sorted ascending by its index.
    """
    res_df = pd.DataFrame(columns=[
        'quality',
        'adj_mut_info',
        'mutual information score',
        'norm_mut_info',
        'adjusted rand score',
        'rand score',
        'n_cluster',
        'pollution_level'
    ])

    per_seed_configs = len(mets[algo_name]) // n_seeds

    for i in range(per_seed_configs):
        rd = dict()
        for j in [i + (k * per_seed_configs) for k in range(n_seeds)]:
            try:
                if rd == {}:
                    rd = {
                        'quality': str(mets[algo_name][j]['quality']) if isinstance(mets[algo_name][j]['quality'], list) else mets[algo_name][j]['quality'],
                        'n_cluster': [mets[algo_name][j]['n_cluster']],
                        **{k: [v] for k, v in mets[algo_name][j]['mutual information'].items()},
                        **{k: [v] for k, v in mets[algo_name][j]['rand'].items()}
                    }
                    if POLL_LEVEL_NAMES[polluter_name] is not None:
                        rd['pollution_level'] = float(eval(mets[algo_name][j]['config'])[POLL_LEVEL_NAMES[polluter_name]])
                    else:
                        rd['pollution_level'] = -1.0
                else:
                    sub_rd = {
                        'n_cluster': mets[algo_name][j]['n_cluster'],
                        **mets[algo_name][j]['mutual information'],
                        **mets[algo_name][j]['rand']
                    }
                    for k, v in sub_rd.items():
                        rd[k].append(v)
            except IndexError as e:
                continue
        for k in rd.keys():
            if k == 'quality' or k == 'pollution_level':
                continue
            rd[k] = sum(rd[k]) / len(rd[k])
        res_df = res_df.append(rd, ignore_index=True)

    res_df['is_decreasing'] = res_df['quality'].diff() < 0
    res_df = res_df[res_df['is_decreasing'] | (res_df.index == 0)]
    res_df = res_df.drop(columns=['is_decreasing'])
    res_df = res_df.set_index('quality', drop=True)
    res_df = res_df.sort_index(ascending=False)

    return res_df


def plot_result_dataframe(algorithm_to_metrics, dataset_name, polluter_name, plots_dir):
    """ Takes a previously generated dictionary of results dataframes as <algorithm_to_metrics> and some meta information
    about the origin of these metrics. Generates one plot per metric with all algorithms in it. Saves the plots as
    >dataset_name>/<polluter_name>/<metric_name>.png to the given <plots_dir>.
    """
    metric_figures = dict()
    y_max = dict()

    # special handling of ConsistentRepresentationPolluter as it requires sorting by and annotation of pollution level
    if polluter_name == 'ConsistentRepresentationPolluter':
        for k in algorithm_to_metrics.keys():
            algorithm_to_metrics[k] = algorithm_to_metrics[k].sort_values('pollution_level')

    texts = list()
    for i, (algo, res_df) in enumerate(algorithm_to_metrics.items()):
        for c in res_df.columns:
            if c not in ['n_cluster', 'adj_mut_info']:
                continue
            if c == 'pollution_level':
                continue
            if metric_figures.get(c) is None:
                # create figure for this clustering metric, as it is the first time it is seen
                fig, ax = plt.subplots(figsize=(15, 10))
                metric_figures[c] = (fig, ax)
                y_max[c] = res_df[c].max() + 0.1 * res_df[c].max()
                # ax.set_title(f'Average {QUALITY_NAME_MAP[c]} over {polluter_name.split("Polluter")[0]} quality for dataset {dataset_name}', size=20)
                ax.set_xlabel(f'{POLLUTER_NAME_MAP[polluter_name.split("Polluter")[0]]} Quality', size=36)
                ax.set_ylabel(f'{QUALITY_NAME_MAP[c]}', size=36)
                ax.tick_params(labelsize=32)
                if isinstance(res_df.index[0], str):
                    ax.set_xticks(range(len(res_df)))
                    ax.set_xticklabels(res_df.index.to_list(), rotation=90)
                else:
                    ax.set_xlim((1.1, -0.1))
            else:
                # load already existing figure for this metric, update the ylims as needed
                fig, ax = metric_figures[c]
                y_max[c] = max(y_max[c], res_df[c].max() + 0.1 * res_df[c].max())
                if c in PLOT_YMAX.keys():
                    y_max[c] = PLOT_YMAX[c]

            # keep rows with index 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
            res_df = res_df.iloc[::2, :]
            # plot the data points for this algorithm and clustering metric
            ax.plot(res_df['quality'], res_df[c], color=ALGO_COLOR_MAP[algo], marker=ALGO_MARKER_MAP[algo], label=algo, ms=15)
            # special handling of ConsistentRepresentationPolluter as it requires sorting by and annotation of pollution level
            # if polluter_name == 'ConsistentRepresentationPolluter':
            #     texts.extend([
            #         ax.text(res_df.index.to_list()[idx], res_df[c].tolist()[idx] + y_max[c] / 100, res_df['pollution_level'].tolist()[idx], size=14) for idx in range(len(res_df))
            #     ])

            # if the original dataset quality is not equal to 1.0, plot horizontal lines indicating the
            # algorithm's performance on the original dataset as well
            if (c == 'adj_mut_info') and (algo in ORIGINAL_DATASET_AMI[dataset_name].keys()) and (DATASET_BASE_QUALITY[polluter_name][dataset_name] < 1.0):
                pass
                # ax.hlines(
                #     ORIGINAL_DATASET_AMI[dataset_name][algo],
                #     0,
                #     1,
                #     linestyles='dashed',
                #     color=ALGO_COLOR_MAP[algo],
                #     alpha=0.8,
                #     label=f'{algo} original'
                # )

    # for each plot do some final adjustments, then save to disk
    for c, (fig, ax) in metric_figures.items():
        ax.set_ylim((0, y_max[c]))
        ax.vlines(
            DATASET_BASE_QUALITY[polluter_name][dataset_name],
            0,
            y_max[c],
            color='black',
            linestyles='dotted',
            label='Original DQ'
        )
        fig.tight_layout()
        for spine in ax.spines.values():
            spine.set_color("lightgray")

        figpath = Path(plots_dir / f'{dataset_name.split(".")[0]}/{polluter_name.split("Polluter")[0]}/{c}.pdf')
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, dpi=300)

        _, existing_labels = ax.get_legend_handles_labels()
        if 'k-Means' not in existing_labels:
            kmeans_line = Line2D([], [], color=ALGO_COLOR_MAP['k-Means'], marker=ALGO_MARKER_MAP['k-Means'], label='k-Means', ms=15)
            handles, labels = ax.get_legend_handles_labels()
            handles.insert(4, kmeans_line)
            labels.insert(4, 'k-Means')
            ax.legend(handles, labels, loc='upper right', borderaxespad=1, fontsize=32, ncol=1)
        else:
            fig.legend(loc='upper right', borderaxespad=1, fontsize=32, ncol=1)

        figpath = Path(plots_dir / f'{dataset_name.split(".")[0]}/{polluter_name.split("Polluter")[0]}/{c}_legend.pdf')
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, dpi=300)

        ax.set_ylim((0, 0.12))
        ax.vlines(
            DATASET_BASE_QUALITY[polluter_name][dataset_name],
            0,
            y_max[c],
            color='black',
            linestyles='dotted',
            label='Original DQ'
        )
        fig.tight_layout()

        figpath = Path(plots_dir / f'{dataset_name.split(".")[0]}/{polluter_name.split("Polluter")[0]}/{c}_focused.pdf')
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath, dpi=300)

        plt.close(fig)


def process_quality(quality_value, dataset, polluter):
    """ Helper function to process a given quality dimension quality measure. Can handle all floating point measures as
    well as the special cases of FeatureAccuracyPolluter and ConsistentRepresentationPolluter qualities.
    """
    if isinstance(quality_value, list):
        quality_value = weighted_average_quality(quality_value, dataset) if polluter == 'FeatureAccuracyPolluter' else round(quality_value[1], 4)
    else:
        quality_value = round(quality_value, 4)

    return quality_value


def get_header(l):
    """ Helper function generating a LaTeX table header for a list of data qualities. Implemented for technical report.
    """
    header = '    Quality'
    ctr = 1
    for qual in l:
        header += " & \\multicolumn{1}{r" + ('' if ctr == len(l) else '|')  + "}{" + f'{qual:.2f}' + "}"
        ctr += 1

    header += '\\\\\n    \\hline\n'

    return header


def get_row(algo, l):
    """ Helper function generating a LaTeX table row for an algorithm and a list of floats (clustering metrics per data
    quality). Implemented for technical report.
    """
    row = f'    {algo}'
    for val in l:
        row += f' & {val:.4f}'
    row += '\\\\\n'
    return row


def get_table(df_dict, dataset, polluter):
    """ Helper function generating a LaTeX table for a given dictionary algorithm -> results dataframe.
    Implemented for technical report.
    """
    table = '\\begin{table*}[b]\n'
    table += '    \\caption{Performance of clustering results for' + f' {polluter.split("Polluter")[0]} ' + 'dimension and ' + f'{dataset.split(".")[0].capitalize()}' + ' dataset.}\n'
    table += '    \\begin{tabular}{l|r|r|r|r|r|r|r|r|r|r|r}\n    & \\multicolumn{11}{c}{Adjusted Mutual Information Score} \\\\ \n'

    has_head = False
    for algo, df in df_dict.items():
        if not has_head:
            table += get_header(df.index.to_list())
            has_head = True
        table += get_row(algo, df['adj_mut_info'])

    table += '    \\end{tabular}\n\\end{table*}'
    return table


def main(args):
    with open(args.results, 'r') as f:
        results = json.load(f)

    if args.ds_to_consider is not None:
        results = {args.ds_to_consider: results[args.ds_to_consider]}

    algorithms = set()
    for dataset, ds_res in results.items():
        for polluter, pollution_res in tqdm(ds_res.items()):
            metrics = dict()
            n_seeds = 0
            n_configs = 0
            for run_config, run_res in pollution_res.items():
                # skip meta information we added later on
                if run_config == 'n_seeds':
                    n_seeds = run_res
                    continue
                elif run_config == 'configurations':
                    n_configs = run_res
                    continue
                for algo, algo_res in run_res.items():
                    if algo == 'quality':
                        continue
                    algorithms.add(algo)
                    if metrics.get(algo) is None:
                        metrics[algo] = list()
                    metrics[algo].append({'quality': process_quality(run_res['quality'], dataset, polluter), 'config': run_config, **algo_res})
            alg_to_metrics = dict()
            for algo in algorithms:
                if algo in metrics.keys():
                    alg_to_metrics[algo] = get_result_dataframe(metrics, algo, polluter, n_seeds)
            if polluter == 'ClassBalancePolluter':
                # get the baseline scores for each algorithm to be used for ConsistentRepresentationPolluter
                # this assumes the following things, if they are not true, adaptations need to be made:
                # 1. The ClassBalancePolluter is processed before the ConsistentRepresentation Polluter
                # 2. The original datasets are already balanced. If this is not the case, the class balance polluter
                #    baseline differs from the real baseline and you have to use a different polluter to get the
                #    baseline from (i.e. CompletenessPolluter, FeatureAccuracyPolluter, TargetAccuracyPolluter)
                baseline = {algo: mets.iloc[0] for algo, mets in alg_to_metrics.items()}
            elif polluter == 'ConsistentRepresentationPolluter':
                # this polluter did not run baseline experiments for any algorithm (defined in configurations)
                # therefore, we need to attach the baselines we know to it
                # this is done by simply inserting the baseline at location 1 (for quality=1) in the results dataframe
                for algo in alg_to_metrics.keys():
                    # attach baseline to results at quality = 1
                    alg_to_metrics[algo].loc[1] = baseline[algo]
                    # sort again (important for plotting)
                    alg_to_metrics[algo] = alg_to_metrics[algo].sort_index(ascending=False)
            # re-sort dictionary to ensure that every time the same order of algorithms (and thus coloring) is chosen
            alg_to_metrics_tmp = {'k-Means': alg_to_metrics['k-Means']} if 'k-Means' in alg_to_metrics.keys() else {'k-Prototypes': alg_to_metrics['k-Prototypes'],}
            alg_to_metrics = {
                'Agglomerative': alg_to_metrics['Agglomerative'],
                'Autoencoder': alg_to_metrics['Autoencoder'],
                'Gaussian Mixture': alg_to_metrics['Gaussian Mixture'],
                **alg_to_metrics_tmp,
                'OPTICS': alg_to_metrics['OPTICS'],
            }
            for algo in alg_to_metrics.keys():
                print(f'Plotting {dataset} {polluter} {algo}...')
                current_df = alg_to_metrics[algo]
                alg_to_metrics[algo] = current_df.groupby(['pollution_level', 'quality']).mean().reset_index().copy()

            plot_result_dataframe(alg_to_metrics, dataset, polluter, args.plots_dir)
            # comment in line below to print a LaTeX-formatted table of the original values
            # print(get_table(alg_to_metrics, dataset, polluter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script reading a results.json and plotting the metrics recorded.')

    parser.add_argument('--results', required=True, type=Path, help='Path to the results.json to plot from.')
    parser.add_argument('--ds-to-consider', required=False, type=str, help='Which dataset to consider for plotting.')
    parser.add_argument('--plots-dir', required=True, type=Path, help='Base directory for the plot file structure.')

    main(parser.parse_args())
