""" Script used to do the sampling of datasets during preprocessing.
This was a required step for the clustering, as the datasets used were too big to be used.
Each dataset is sampled with each of the random seeds defined in the ``metadata.json`` file.

Prior to sampling, the number of samples which is requested through the CLI is checked and potentially changed to a
multiple of the number of classes in the input dataset.
Afterwards, sampling is done class by class, from smallest to largest class. If every class can provide enough samples
(1 / num classes samples), the dataset returned is balanced. Otherwise, sampling is done to maximize the balance of
per-class sample counts. I.e., for the smallest class, all samples are chosen, then the per-class samples are recalculated.
If each class can provide this number of samples, this is it, otherwise this process is iteratively repeated.
"""
import argparse
import json

import pandas as pd

from pathlib import Path


def relatively_equal_sampling(dataset, target_col, n_samples, seed):
    """ See sampling explanation in script description comment above.
    """
    res_df = pd.DataFrame(columns=dataset.columns)
    m = dataset[target_col].nunique()
    per_class_samples = n_samples // m
    remainder = n_samples - (per_class_samples * m)
    class_value_counts = dataset[target_col].value_counts()
    classes_sorted = class_value_counts.index.tolist()  # sorted largest to smallest
    classes_sorted.reverse()                            # sorted smallest to largest
    print(f'Per-class samples: {per_class_samples}')
    for i, cls in enumerate(classes_sorted):
        if remainder == (m - i):
            per_class_samples += 1
            remainder = 0
        if class_value_counts[cls] < per_class_samples:
            n_samples = n_samples - class_value_counts[cls]
            per_class_samples = n_samples // (m-(i+1))
            remainder = n_samples - (per_class_samples * (m - (i+1)))
            res_df = res_df.append(dataset.loc[dataset[target_col] == cls])
            print(f'Per-class samples recalculated: {per_class_samples}')
        else:
            # would, in theory, have to reduce n_samples remaining, BUT we are only getting bigger classes from now on
            # and therefore will not have to recalculate the per_class_samples again
            res_df = res_df.append(
                dataset.loc[dataset[target_col] == cls].sample(per_class_samples, replace=False, random_state=seed)
            )

    return res_df.sort_index().sample(frac=1, replace=False, random_state=seed).reset_index(drop=True)


def main(args):
    df = pd.read_csv(args.dataset, sep=',')

    with open('../metadata.json', 'r') as f:
        metadata = json.load(f)

    target_class = metadata[args.dataset.name]['target']
    random_seeds = metadata['random_seeds']

    while args.samples % df[target_class].nunique() != 0:
        args.samples = args.samples + 1

    print(f'Sample count will be {args.samples}')

    metadata[args.dataset.name]['n_samples'] = args.samples

    with open('metadata_.json', 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4)

    result_dfs = list()

    for s in random_seeds:
        result_dfs.append(relatively_equal_sampling(df, target_class, args.samples, s))

    for i, s in enumerate(random_seeds):
        result_dfs[i].to_csv(f'data/clean/{args.dataset.stem}_{s}{args.dataset.suffix}', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to perform dataset sampling.')

    parser.add_argument('-d', '--dataset', required=True, type=Path, help='Path to the dataset to sample.')
    parser.add_argument('-s', '--samples', required=True, type=int, help='Number of samples to draw from dataset.')

    main(parser.parse_args())
