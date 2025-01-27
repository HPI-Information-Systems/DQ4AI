"""
Helper script, used to calculate the quality measure for a given dataset and polluter.
This can be required in multiple cases:
1. Original Datasets were never used in polluter (e.g. Uniqueness Polluter or Class Balance Polluter)
2. Quality measure was not calculated or recorded during experiments and needs to be retroactively determined for
   persisted polluted datasets.
"""
import json
import pandas as pd

from pathlib import Path

from polluters import ClassBalancePolluter, CompletenessPolluter, ConsistentRepresentationPolluter, \
    FeatureAccuracyPolluter, TargetAccuracyPolluter, UniquenessPolluter


DATA_DIR = Path('data/')
CLEAN_DATA_DIR = DATA_DIR / 'clean/'
POLLUTED_DATA_DIR = DATA_DIR / 'polluted/'


def compute_org_ds_quality():
    # list of dataset names
    datasets = ['letter.arff', 'covtype.csv', 'bank.csv', 'covid_data_pre_processed_clustering.csv']
    datasets = ['SouthGermanCredit.csv', 'TelcoCustomerChurn.csv', 'cmc.data', 'covid_data_pre_processed.csv']
    datasets = ['SouthGermanCredit.csv', 'TelcoCustomerChurn.csv', 'cmc.data']

    if not POLLUTED_DATA_DIR.exists():
        POLLUTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # read dataset metadata file
    with open('../metadata.json', 'r') as f:
        metadata = json.load(f)

    # specific to the clustering datasets - they were all pre-sampled with the seeds from metadata
    # the new, pre-sampled datasets have their sampling seed in their name - hence we have 5 datasets for each base
    tmp_datasets = []
    for d in datasets:
        parts = d.split('.')
        tmp_datasets.extend([f'{parts[0]}_{s}.{parts[1]}' for s in metadata['random_seeds']])
        #datasets = tmp_datasets

    qualities = list()
    for ds_name in datasets:
        df = pd.read_csv(CLEAN_DATA_DIR / ds_name)

        # take only the name until the last _
        #ds_name = ds_name.rsplit('_', 1)[0] + '.' + ds_name.split('.')[1]

        # instantiate any version of the polluter you want - no pollution will be done, so params do not matter
        poll = ClassBalancePolluter(0, metadata[ds_name]['target'], None, random_seed=0)
        #poll = UniquenessPolluter.configure(metadata, df, ds_name)[0]
        #poll = CompletenessPolluter.configure(metadata, df, ds_name)[0]
        #poll = ConsistentRepresentationPolluter.configure(metadata, df, ds_name)[0]
        # calculate quality measure and print to console
        print(f"{ds_name}: {poll.compute_quality_measure(df)}")
        qualities.append(poll.compute_quality_measure(df))
        # specific to clustering datasets, we have 5 sampled versions of each dataset, so we want to average
        # out their qualities
        if len(qualities) == 5:
            print(f'Quality average: {sum(qualities) / len(qualities)}')
            qualities = list()


if __name__ == '__main__':
    compute_org_ds_quality()
