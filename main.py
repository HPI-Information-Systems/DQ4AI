import json
import logging
from util import start_logging

import pandas as pd

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from dataset_manager import DatasetManager

"""
This module is intended as an aggregator for the experiments defined in the individual packages
"""

# constants
DATA_DIR = Path('data/')
CLEAN_DATA_DIR = DATA_DIR / 'clean/'
POLLUTED_DATA_DIR = DATA_DIR / 'polluted/'
RESULTS_JSON_PATH = 'results.json'


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


def main():
    # list of polluter classes
    pollution_methods = []
    # list of experiment classes
    experiments = []
    # list of dataset names
    datasets = []
    # run start time (for results json key)
    start_timestamp = f'{datetime.now():%Y_%m_%d_%H_%M_%S}'

    if not POLLUTED_DATA_DIR.exists():
        POLLUTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # read dataset metadata file
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)

    # iterate through your datasets to read, pollute and evaluate them
    for ds_name in datasets:
        # special handling for datasets pre-sampled with a given seed
        # it is assumed that the datasets are named <ds_name>_<random_seed>.csv (e.g. covtype_12345.csv)
        ds_sampled_with_seed = False
        for rseed in metadata['random_seeds']:
            ds_sampled_with_seed = rseed if str(rseed) in ds_name else False
            if ds_sampled_with_seed:
                break

        # special handling for dataset pre-sampled with a given seed
        # if we found that a random seed is in the dataset name
        #   - deepcopy current metadata for later restoring
        #   - modify the metadata to only run polluters with that random seed
        if ds_sampled_with_seed:
            old_metadata = deepcopy(metadata)
            metadata['random_seeds'] = [ds_sampled_with_seed]
            metadata[ds_name] = metadata[''.join(ds_name.split('_' + str(ds_sampled_with_seed)))]

        # check whether dataset is defined in metadata
        if ds_name not in metadata.keys():
            raise KeyError(f'Dataset {ds_name} is not specified in the metadata file.')

        # read dataset, assuming it is a standard csv saved with no index column and a header
        dataset = pd.read_csv(CLEAN_DATA_DIR / ds_name)

        for polluter_class in pollution_methods:
            polluters = polluter_class.configure(metadata, dataset, ds_name)

            for polluter in polluters:
                logging.info(f'Polluting dataset {ds_name} with {polluter.__class__.__name__}')
                polluted_df, quality = DatasetManager.read_or_pollute(POLLUTED_DATA_DIR, dataset, ds_name, polluter)
                DatasetManager.persist_dataset(POLLUTED_DATA_DIR, polluted_df, ds_name, polluter)
                logging.info(f'Pollution finished, logging results...')
                logging.info(f'Pollution parameters: {polluter.get_pollution_params()}')
                logging.info(f'Quality Measure: {quality}')

                existing_results = init_results_file(
                    start_timestamp,
                    ds_name,
                    polluter.__class__.__name__,
                    polluter.get_pollution_params(),
                    quality
                )

                # Our current experiment loop implementation for the clustering experiments
                for experiment in experiments:
                    # Instanciate Experiment - need to know categorical features and target feature for encoding and
                    # data split, therefore we just pass the metadata corresponding to the current dataset
                    exp = experiment(polluted_df, metadata[ds_name])
                    # results are metrics (e.g. accuracy) returned as a dictionary
                    results = exp.run()

                    # update the results dictionary with our current experiment
                    existing_results[start_timestamp][ds_name][polluter.__class__.__name__][polluter.get_pollution_params()].update({
                        exp.name: results
                    })

                    # persist existing results dictionary after each experiment, in case something fails
                    with open(RESULTS_JSON_PATH, 'w') as f:
                        json.dump(existing_results, f, indent=4, sort_keys=True)

                    logging.info(f'{exp.name} results: {results}')

        # special handling for dataset pre-sampled with a given seed
        # reset metadata to original state
        if ds_sampled_with_seed:
            metadata = old_metadata


if __name__ == "__main__":
    start_logging(cmd_out=True)
    main()
