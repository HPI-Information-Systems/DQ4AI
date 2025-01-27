import copy
import json
import logging
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datetime import datetime
from pathlib import Path
from polluters import Polluter, CompletenessPolluter, UniquenessPolluter, ClassBalancePolluter, FeatureAccuracyPolluter, \
    TargetAccuracyPolluter, ConsistentRepresentationPolluter
from regression.utils import discretize_column
from dataset_manager import DatasetManager
from util import start_logging

from regression.experiments import LinearRegressionExperiment, RidgeRegressionExperiment, \
    DecisionTreeRegressionExperiment, \
    RandomForestRegressionExperiment, MLPRegressionExperiment, MLPRegression5Experiment, MLPRegression10Experiment, \
    GradientBoostingRegressionExperiment, PytorchMLPRegressionExperiment, PytorchMLPRegression5Experiment, \
    PytorchMLPRegression10Experiment, TabNetRegressionExperiment

"""
This module is intended as an aggregator for the experiments defined in the regression package
"""

# constants
DATA_DIR = Path('data/')
CLEAN_DATA_DIR = DATA_DIR / 'clean/'
POLLUTED_DATA_DIR = DATA_DIR / 'polluted/'
RESULTS_JSON_PATH = 'covid_data_pre_processed_regression_tabnet.json'

# REGRESSION_DATASETS = ["house_prices_prepared.csv",
#                       "vw_prepared.csv", "imdb_prepared.csv"]
REGRESSION_DATASETS = ["covid_data_pre_processed_regression.csv"]
# list of polluter classes
# POLLUTION_METHODS = [CompletenessPolluter, TargetAccuracyPolluter, ClassBalancePolluter, FeatureAccuracyPolluter,
#                     UniquenessPolluter, ConsistentRepresentationPolluter]
POLLUTION_METHODS = [ConsistentRepresentationPolluter]
# list of experiment classes
# EXPERIMENTS = [LinearRegressionExperiment, RidgeRegressionExperiment, DecisionTreeRegressionExperiment,
#                RandomForestRegressionExperiment, PytorchMLPRegressionExperiment, PytorchMLPRegression5Experiment,
#                PytorchMLPRegression10Experiment, GradientBoostingRegressionExperiment]
EXPERIMENTS = [TabNetRegressionExperiment]


def init_results_file(run_timestamp, ds_name, polluter_name, polluter_params, quality):
    # open an existing results.json if it exists, otherwise create a new dictionary
    try:
        with open(RESULTS_JSON_PATH, 'r') as f:
            existing_results = json.load(f)
    except OSError:
        existing_results = {}

    def set_if_none(d, k, v=None):
        if d.get(k) is None:
            d[k] = v if v is not None else dict()

    # initialize results dictionary
    # the results will be saved in a hierarchical structure
    #   |- %Run Start Timestamp%
    #       |- %Dataset Name%
    #           |- %Polluter Class Name%
    #               | - %Pollution parameter dictionary (represented as string to be able to be used as key)%
    #                   | - quality: %Quality Measure%
    #                   | - %Experiment Name% (will not be initialized here)
    #                       | - %Experiment results% (we implement this as a dict)
    set_if_none(existing_results, run_timestamp)
    set_if_none(existing_results[run_timestamp], ds_name)
    set_if_none(existing_results[run_timestamp][ds_name], polluter_name)
    set_if_none(
        existing_results[run_timestamp][ds_name][polluter_name],
        polluter_params,
        {'quality': quality}
    )

    return existing_results


def run_regression_experiments():
    # run start time (for results json key)
    start_timestamp = f'{datetime.now():%Y_%m_%d_%H_%M_%S}'

    if not POLLUTED_DATA_DIR.exists():
        POLLUTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # read dataset metadata file
    with open('../metadata.json', 'r') as f:
        metadata = json.load(f)
    # iterate through your datasets to read, pollute and evaluate them
    for ds_name in REGRESSION_DATASETS:
        # check whether dataset is defined in metadata
        if ds_name not in metadata.keys():
            raise KeyError(
                f'Dataset {ds_name} is not specified in the metadata file.')

        # read dataset, assuming it is a standard csv saved with no index column and a header
        dataset = pd.read_csv(CLEAN_DATA_DIR / ds_name)
        categoricals = metadata[ds_name]["categorical_cols"]

        # stratify on discretized target feature
        target_feature = metadata[ds_name]['target']
        discr_target_feature = "discr_" + target_feature
        discr_step_size = metadata[ds_name]['discr_step_size']
        discr_bins = np.arange(dataset[target_feature].min(
        ), dataset[target_feature].max() + discr_step_size, discr_step_size)

        df_with_discr = discretize_column(
            dataset, target_feature, discr_target_feature, discr_bins)
        if ds_name == 'covid_data_pre_processed_regression.csv':
            class_counts = df_with_discr[discr_target_feature].value_counts()
            valid_classes = class_counts[class_counts >= 2].index
            df_with_discr = df_with_discr[df_with_discr[discr_target_feature].isin(valid_classes)]
        dataset_train_with_discr, dataset_test_with_discr = train_test_split(
            df_with_discr, test_size=0.2, random_state=1, stratify=df_with_discr[discr_target_feature])

        dataset_train_with_discr.reset_index(inplace=True, drop=True)
        dataset_test_with_discr.reset_index(inplace=True, drop=True)

        dataset_train = dataset_train_with_discr.drop(
            columns=discr_target_feature)
        dataset_test = dataset_test_with_discr.drop(
            columns=discr_target_feature)

        for polluter_class in POLLUTION_METHODS:
            # Distinguish between cases: the UniquenessPolluter and ClassBalancePolluter need discretized
            # target features to pollute on. Further, for the ClassBalancePolluter, too small classes
            # have to be discarded because it would result in too small polluted datasets. The classes are
            # then also discarded from the original datasets too (only when using ClassBalancePolluter) to allow a more consistent comparison

            if polluter_class == UniquenessPolluter or polluter_class == ClassBalancePolluter:
                discr_metadata = copy.deepcopy(metadata)
                discr_metadata[ds_name]['target'] = discr_target_feature
                polluters = polluter_class.configure(
                    discr_metadata, df_with_discr, ds_name)

                if polluter_class == ClassBalancePolluter:
                    # We want to pollute on the discretized target feature and small classes should be discarded
                    dataset_train_to_pollute = dataset_train_with_discr[
                        dataset_train_with_discr[discr_target_feature].isin(
                            discr_metadata[ds_name]['class_balance_polluter_classes'])].copy()
                    dataset_test_to_pollute = dataset_test_with_discr[
                        dataset_test_with_discr[discr_target_feature].isin(
                            discr_metadata[ds_name]['class_balance_polluter_classes'])].copy()
                    # The original comparison dataset should not have the discretized column in it anymore
                    original_dataset_train = dataset_train_to_pollute.drop(
                        columns=discr_target_feature)
                    original_dataset_test = dataset_test_to_pollute.drop(
                        columns=discr_target_feature)

                else:
                    # We want to pollute on the discretized target feature
                    dataset_train_to_pollute = dataset_train_with_discr.copy()
                    dataset_test_to_pollute = dataset_test_with_discr.copy()
                    # The original comparison dataset should not have the discretized column in it anymore
                    original_dataset_train = dataset_train.copy()
                    original_dataset_test = dataset_test.copy()
            else:
                # We want to pollute on the dataset as it is
                polluters = polluter_class.configure(
                    metadata, dataset, ds_name)
                original_dataset_train = dataset_train.copy()
                original_dataset_test = dataset_test.copy()
                dataset_train_to_pollute = dataset_train.copy()
                dataset_test_to_pollute = dataset_test.copy()

            for polluter in polluters:
                logging.info(
                    f'Polluting dataset {ds_name} with {polluter.__class__.__name__}')

                poll_dataset_train, quality = DatasetManager.read_or_pollute(
                    POLLUTED_DATA_DIR, dataset_train_to_pollute, 'train_' + ds_name, polluter)
                # We ignore test quality because it is the same as train quality
                poll_dataset_test, _ = DatasetManager.read_or_pollute(
                    POLLUTED_DATA_DIR, dataset_test_to_pollute, 'test_' + ds_name, polluter)

                # drop discretized target column if it is present
                if discr_target_feature in poll_dataset_train:
                    poll_dataset_train = poll_dataset_train.drop(
                        columns=discr_target_feature)
                if discr_target_feature in poll_dataset_test:
                    poll_dataset_test = poll_dataset_test.drop(
                        columns=discr_target_feature)

                DatasetManager.persist_dataset(
                    POLLUTED_DATA_DIR, poll_dataset_train, 'train_' + ds_name, polluter)
                DatasetManager.persist_dataset(
                    POLLUTED_DATA_DIR, poll_dataset_test, 'test_' + ds_name, polluter)
                logging.info(f'Pollution finished, logging results...')
                logging.info(
                    f'Pollution parameters: {polluter.get_pollution_params()}')
                logging.info(f'Quality Measure: {quality}')

                scenarios = {"train_original_test_original": (original_dataset_train, original_dataset_test),
                             "train_polluted_test_original": (poll_dataset_train, original_dataset_test),
                             "train_original_test_polluted": (original_dataset_train, poll_dataset_test),
                             "train_polluted_test_polluted": (poll_dataset_train, poll_dataset_test)}

                existing_results = init_results_file(
                    start_timestamp,
                    ds_name,
                    polluter.__class__.__name__,
                    polluter.get_pollution_params(),
                    quality
                )

                for scenario_name, scenario_datasets in scenarios.items():
                    scenario_train_dataset, scenario_test_dataset = scenario_datasets

                    # Our current experiment loop implementation for the clustering experiments
                    for experiment in EXPERIMENTS:
                        logging.info(
                            f'Starting experiment {experiment} for scenario {scenario_name} and dataset {ds_name} with {polluter.__class__.__name__}')

                        # Instantiate Experiment - need to know categorical features and target feature for encoding and
                        # data split, therefore we just pass the metadata corresponding to the current dataset
                        exp = experiment(scenario_train_dataset,
                                         scenario_test_dataset,
                                         target_feature,
                                         categoricals)
                        # results are metrics (e.g. accuracy) returned as a dictionary
                        results = exp.run()

                        # update the results dictionary with our current experiment
                        if existing_results[start_timestamp][ds_name][polluter.__class__.__name__][
                                polluter.get_pollution_params()].get(scenario_name) is None:
                            existing_results[start_timestamp][ds_name][polluter.__class__.__name__][
                                polluter.get_pollution_params()][scenario_name] = {}

                        existing_results[start_timestamp][ds_name][polluter.__class__.__name__][
                            polluter.get_pollution_params()][scenario_name].update({
                                exp.name: results
                            })

                        def convert_np_to_native(obj):
                            if isinstance(obj, np.float32):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()  # Convert numpy arrays to lists
                            elif isinstance(obj, dict):
                                return {k: convert_np_to_native(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_np_to_native(i) for i in obj]
                            return obj
                        existing_results = convert_np_to_native(existing_results)
                        # persist existing results dictionary after each experiment, in case something fails
                        with open(RESULTS_JSON_PATH, 'w') as f:
                            json.dump(existing_results, f,
                                      indent=4, sort_keys=True)

                        logging.info(f'{exp.name} results: {results}')


if __name__ == "__main__":
    start_logging(cmd_out=True)
    run_regression_experiments()
