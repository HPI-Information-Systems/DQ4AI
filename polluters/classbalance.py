from logging import warning as logging_warning
from numpy import inf
from pandas import DataFrame


from .interfaces import Polluter

from math import ceil, floor
from typing import List


def _calc_per_class_samples(inter_class_distance, class_order, balanced_per_class_samples):
    """
    Calculates how many samples each class should have in the polluted dataset and returns a list of class_id,
    sample_count tuples.

    :param inter_class_distance: distance (difference of sample counts) for two adjacent classes
    :param class_order: list of class_id, available_sample_count sorted (ASC) by sample count, then class id
    :param balanced_per_class_samples: samples per class in the (hypothetical) balanced baseline dataset
    :return: list of calculated sample counts to return with the polluted dataset per class
    :rtype: list([class_id, int])
    """
    per_class_samples = list()
    m = len(class_order)    # number of classes in target
    # calculate the number of samples to return for the smallest class
    if m % 2 == 1:
        # for m odd: balanced_samples - (floor(m / 2) * distance)
        per_class_samples.append([
            class_order[0],
            int(balanced_per_class_samples - (floor(m / 2) * inter_class_distance))
        ])
    else:
        # for m even: balanced_samples - (((m / 2) - 0.5) * distance)
        per_class_samples.append([
            class_order[0],
            int(balanced_per_class_samples - (((m / 2) - 0.5) * inter_class_distance))
        ])

    for i in range(1, len(class_order)):
        per_class_samples.append([class_order[i], per_class_samples[i-1][1] + inter_class_distance])
    return per_class_samples


def _sum_absolute_per_class_distances(m, per_class_distance):
    """
    Calculates and returns the sum of absolute per-class sample count distances (differences).
    This is based on the number of classes and the defined distance between per-class sample counts.
    The calculation already eliminates duplicate edges (e.g. A -> C is considered, so C -> A is not).

    :param m: number of classes in the dataset
    :param per_class_distance: difference in per-class sample count
    :return: sum of absolute per-class sample count differences
    :rtype: int
    """
    return per_class_distance * sum([(m - pcd_mult) * pcd_mult for pcd_mult in range(m)])


def _calc_sample_counts(percentages, highest_possible):
    """
    Calculates, for the given list of percentages and a maximum number the absolute values corresponding
    to these percentages in reference to the given maximum.

    :param percentages: list of floats in [0.0, 1.0]
    :param highest_possible: integer denoting the reference to apply the percentages to (i.e. the 100% value)
    :return: sorted (asc) list of integers similar to to round(percentages * highest_possible)
             the difference between two adjacent values is always the same
    """
    percentages = sorted(percentages)
    potential_counts = [round(sam_pct * highest_possible) for sam_pct in percentages]
    # ensure that the differences are REALLY always the same, just to be sure!
    diffs = [potential_counts[i+1] - potential_counts[i] for i in range(len(potential_counts) - 1)]
    equalized_diff = floor(sum(diffs) / len(diffs))
    counts = [potential_counts[0] + i * equalized_diff for i in range(len(potential_counts))]
    return counts


def _max_lvl_percentages(m):
    """
    For maximum imbalance, we fix that largest class is at 100% of samples allowed and smallest class is at
    1% of samples allowed. All other classes in between are equally spaced in terms of percentage.

    Therefore, we calculate the spacing between percentages as 0.99 (remaining percentage from 0.01 to 1.0) / m-1
    (number of spaces).

    :param m: number of classes in target
    :returns: percentage of samples allowed for each class, assuming classes sorted by sample count in original data
    :rtype: list
    """
    spacing = 0.99 / (m - 1)
    return [0.01] + [0.01 + (i * spacing) for i in range(1, m - 1)] + [1.0]


def _calc_imbalance_level(m, n, per_class_distance):
    """
    Calculates the imbalance level based on dataset metrics and absolute per-class sample count distance
    (difference).

    :param m: number of classes in the dataset
    :param n: total number of samples the polluter will return
    :param per_class_distance: difference in sample count for two adjacent classes in the polluted dataset
    :return: imbalance level
    :rtype: float
    """
    return _sum_absolute_per_class_distances(m, per_class_distance) / (((m + 1) / 3) * n)


def _calc_max_lvl_sample_counts(per_class_samples, n_samples_to_return=None):
    """
    Calculates the number of samples to return per class when assuming maximum imbalance. Ensures that this is
    possible by setting the number of samples for the largest class accordingly.

    :param per_class_samples: number of samples available in the original data per class
    :type per_class_samples: list
    :param n_samples_to_return: number of samples to return with the polluted dataset, optional
    :type n_samples_to_return: int
    :returns: sample count per class, assuming classes sorted by sample count in original data
    :rtype: list
    """
    # sorts per-class sample counts ascending
    sorted_pcs = sorted(per_class_samples)
    # gets number of distinct classes in target
    m = len(sorted_pcs)
    # returns maximum pollution percentages for per class samples in reference to largest class sample count
    percentages = _max_lvl_percentages(m)

    # can't get more samples than the most-represented class has
    highest_possible = sorted_pcs[-1]
    # iterate through smaller classes, reduce the highest possible sample count to return as needed
    for i in range(m - 1):
        highest_possible = min(highest_possible, floor(sorted_pcs[i] / percentages[i]))

    # ensure that the final per-class sample counts are created in a way that a balanced dataset can be created
    # 1. Sum of maximum imbalance per-class sample counts needs to be dividable by the number of classes
    # 2. Sum of maximum imbalance per-class sample counts divided by number of classes can't be larger than smallest class sample count
    # 3. Sum of maximum imbalance per-class sample counts can't be larger than number of samples to return (if given)
    while sum(_calc_sample_counts(percentages, highest_possible)) % m != 0 or \
            sum(_calc_sample_counts(percentages, highest_possible)) // m > sorted_pcs[0] or \
            (n_samples_to_return is not None and sum(_calc_sample_counts(percentages, highest_possible)) > n_samples_to_return):
        highest_possible -= 1

    return _calc_sample_counts(percentages, highest_possible)


def _calc_max_samples(per_class_samples):
    """
    Calculates and returns the maximum possible total sample count to return for a given list of per-class sample
    counts.

    :param per_class_samples: list of integers, sample counts for each class
    :return: maximum possible total number of samples for balancing/imbalancing
    :rtype: int
    """
    return sum(_calc_max_lvl_sample_counts(per_class_samples))


class ClassBalancePolluter(Polluter):
    """
    This class implements a class balance polluter, which changes the balance of per-class sample counts in a given
    dataset. This imbalance is created on a balanced baseline by adding / removing an equal amount of samples
    per class with more samples being removed for classes further away from the middle of a list sorted ascending
    by sample count and then class name in the original dataset. I.e. the sample counts of the resulting dataset will
    always approximate a straight line if sorted.
    """

    @staticmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        """ Generates a dictionary of static parameters to initialize the own specialized polluter instances from.
        This is done by reading the appropriate fields in the metadata dictionary section corresponding to the given
        dataset name and/or analyzing the given dataset.

        :param metadata: dataset metadata dictionary read from the metadata.json file
        :type metadata: dict
        :param dataset: raw dataset as read from disk
        :type dataset: pd.DataFrame
        :param ds_name: name of the dataset file - same as the key in the metadata dictionary
        :type ds_name: str
        :returns: parameter dictionary to use in polluter instance initialization
        :rtype: dict
        """
        spec_meta = metadata[ds_name]
        return {
            'target_column': spec_meta['target'],
            'n_samples': len(dataset) if spec_meta.get('n_samples') is None else spec_meta.get('n_samples')
        }

    @classmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        """ Configures the special polluter, setting its parameters based on the metadata, dataset, dataset name
        and random seed provided.
        In addition to fixed per-dataset parameters, each polluter may specify, in this class, ranges from which to
        pull flexible parameters (e.g. percentage of samples to pollute).
        This function returns a list of configured polluter instances.

        :param metadata: dataset metadata dictionary read from the metadata.json file
        :type metadata: dict
        :param dataset: raw dataset as read from disk
        :type dataset: pd.DataFrame
        :param ds_name: name of the dataset file - same as the key in the metadata dictionary
        :type ds_name: str
        :returns: list of configured polluter instances
        :rtype: list
        """
        configured_polluters = list()
        static_params = cls.get_static_params(metadata, dataset, ds_name)
        for rand_seed in metadata['random_seeds']:
            for imbalance_level in [i / 20 for i in range(21)]:
                configured_polluters.append(cls(imbalance_level=imbalance_level, random_seed=rand_seed, **static_params))
        return configured_polluters

    def __init__(self, imbalance_level: float, target_column: str, n_samples: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imbalance_level = imbalance_level
        self._target_col = target_column
        self._target_n_samples = inf if n_samples is None else n_samples

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame = None) -> float:
        """
        This method calculates the quality measure of the polluted dataframe.
        For the class balance, it is presumed to be worst (quality = 0) if the class sample
        counts alternate between the maximum sample count and no samples at all. This can never
        occur and, thus, constitutes a theoretical minimum. The maximum balance of 1 is achieved
        if all classes have the same sample count.

        From the theoretical minimum, we can determine the maximum possible sum of per-class
        sample count differences to be floor(m/2) * ceil(m/2) * max sample count.
        Calculating the actual sum of per-class sample count differences and using this theoretical
        difference, we can calculate the quotient, which is the measure of how imbalanced the dataset
        is, 1 being the most imbalanced. Calculating 1 - this measure, we get the quality measure.

        :param df_polluted: the polluted dataframe (required)
        :param df_clean: the original dataframe, unused for this quality measure
        :return: the quality measure of the polluted dataframe
        """
        # number of classes
        m = df_polluted[self._target_col].nunique()
        # per-class sample counts
        class_sample_counts = df_polluted[self._target_col].value_counts().to_dict()
        # list of (classname, sample count) sorted descending by sample count first and class name second
        class_order = sorted(class_sample_counts.items(), key=lambda el: (el[1], el[0]), reverse=True)

        # maximum distance between two individual classes (= empirical maximum possible sample count)
        max_ind_dist = class_order[0][1]
        # maximum distance in the whole class system, assuming pairwise distances to be taken only in one direction
        max_tot_dist = ceil(m/2) * floor(m/2) * max_ind_dist

        ind_distances = list()
        for i, (_, majority_count) in enumerate(class_order):
            for _, minority_count in class_order[i + 1:]:
                ind_distances.append(majority_count - minority_count)

        return 1.0 - (sum(ind_distances) / max_tot_dist)

    def pollute(self, df: DataFrame) -> DataFrame:
        # generate the order of classes based on their sample count first and name second
        class_sample_counts = df[self._target_col].value_counts().to_dict()
        class_order = sorted(class_sample_counts.items(), key=lambda el: (el[1], el[0]))

        # calculated how many samples would be used at maximum pollution level
        max_samp_count = _calc_max_samples([tup[1] for tup in class_order])

        # tell user if we can't return as many samples as desired
        if max_samp_count < self._target_n_samples:
            logging_warning(f'The maximum possible number of samples that can be returned ({max_samp_count}) '
                            f'is smaller than the desired number of samples to return ({self._target_n_samples}).\n'
                            f'Will return maximum possible number of samples.')
            self._target_n_samples = max_samp_count

        # ensure that, in a balanced dataset, all classes can have the same sample count (none are off by one)
        original_target_n_samples = self._target_n_samples
        while self._target_n_samples % len(class_sample_counts) != 0:
            self._target_n_samples -= 1
        if original_target_n_samples != self._target_n_samples:
            logging_warning(f'As the original target number of samples ({original_target_n_samples}) was not dividable '
                            f'by the number of classes without leaving a remainder, it was changed to '
                            f'({self._target_n_samples}).')

        # calculate, in case of maximum pollution level, how many samples per class would be returned
        max_pc_samples = _calc_max_lvl_sample_counts([tup[1] for tup in class_order], self._target_n_samples)

        # calculate, based on the imbalance level, how many samples difference there will be between
        # adjacent classes in the returned dataframe
        inter_class_dist = self._calc_absolute_per_class_distance(self._imbalance_level,
                                                                  len(class_order),        # num classes
                                                                  self._target_n_samples,  # num samples desired
                                                                  max_pc_samples[-1])      # "100%" at max poll. level

        # calculate the actual per-class sample count to extract and return
        per_class_samples = _calc_per_class_samples(inter_class_dist,
                                                    [tup[0] for tup in class_order],
                                                    self._target_n_samples / len(class_order))  # samples per class if balanced

        if sum([s_count for _, s_count in per_class_samples]) != self._target_n_samples:
            diff = self._target_n_samples - sum([s_count for _, s_count in per_class_samples])
            assert diff < len(per_class_samples), f'Something is really wrong, missed samples ({diff}) may never be ' \
                                                  f'larger than the number of classes ({len(per_class_samples)}) ' \
                                                  f'as they could\'ve been split up over the classes!'
            margin = len(per_class_samples) - diff  # how many classes to not touch on higher side of distribution
            for i in range(len(per_class_samples) - margin - diff, len(per_class_samples) - margin):
                per_class_samples[i][1] += 1

        res_df = DataFrame(columns=df.columns)
        for class_id, s_count in per_class_samples:
            res_df = res_df.append(
                df.loc[df[self._target_col] == class_id].sample(s_count, replace=False, random_state=self.random_seed)
            )

        res_df = res_df.astype(df.dtypes.to_dict())

        return res_df.sort_index().sample(frac=1, replace=False, random_state=self.random_seed).reset_index(drop=True)

    def _calc_absolute_per_class_distance(self, imb_lvl, m, n, largest_class_samples):
        """
        Calculates, for the given imbalance level, the absolute distance between per-class sample counts required to
        get the dataset to that imbalance level.
        If an imbalance level would exceed the maximum allowed imbalance (smallest class has 1% of samples of largest)
        it is auto-capped to the maximum valid level.

        :param imb_lvl: imbalance level to calculate the per-class distance for
        :param m: number of classes in the dataset
        :param n: desired total number of samples to return
        :param largest_class_samples: number of samples available for the largest class in the dataset
        :return: the distance each class' sample count should have to both its neighboring classes in polluted dataset
        :rtype: int
        """
        per_class_distance = floor((imb_lvl * ((m + 1) / 3) * n) / sum([(m - pcd_mult) * pcd_mult for pcd_mult in range(m)]))
        changed_distance = False
        while (largest_class_samples - ((m-1) * per_class_distance)) < round(0.01 * largest_class_samples):
            per_class_distance -= 1
            changed_distance = True
        if changed_distance:
            self._imbalance_level = _calc_imbalance_level(m, n, per_class_distance)
            logging_warning('Changed the pollution level as it exceeded the reasonable limit. Now at {}'.format(
                self._imbalance_level
            ))
        return int(per_class_distance)
