from pandas import DataFrame, Series, concat, read_csv, cut
from ast import literal_eval
from numpy import mean, std, float32, arange, sum
from json import loads
from util import get_majority_baseline_performance, get_ratio_baseline_performance

CONSISTENT_REPRESENTATION_POLLUTER = ['ConsistentRepresentationPolluter', 'ConsistentRepresentationPolluter_two',
                                      'ConsistentRepresentationPolluter_five']


def get_baselines():
    """
    Compute both baselines for all classification datasets for accuracy and f1-score.
    """
    with open('../metadata.json', 'r') as file:
        metadata = loads(file.read())
    datasets = ['TelcoCustomerChurn.csv', 'SouthGermanCredit.csv', 'cmc.data', 'covid_data_pre_processed.csv']
    baselines = {dataset: {'accuracy': {}, 'f1-score': {}} for dataset in datasets}
    for dataset in datasets:
        df = read_csv(f"../classification/data/clean/{dataset}", sep=',')
        accuracy, f1_score = get_majority_baseline_performance(df[metadata[dataset]['target']].values)
        baselines[dataset]['accuracy']['baseline_majority'] = accuracy
        baselines[dataset]['f1-score']['baseline_majority'] = f1_score
        accuracy, f1_score = get_ratio_baseline_performance(df[metadata[dataset]['target']].values)
        baselines[dataset]['accuracy']['baseline_ratio'] = accuracy
        baselines[dataset]['f1-score']['baseline_ratio'] = f1_score
    return baselines


def add_distribution_to_uniqueness(df):
    """
    UniquenessPolluter's results came with two different distributions. That's why they need to be evaluated separately.
    """
    return_df = df.copy()
    mask_same = (df.polluter == 'UniquenessPolluter') & df.polluter_config.apply(lambda x: 'same' in str(x))
    mask_normal = (df.polluter == 'UniquenessPolluter') & df.polluter_config.apply(lambda x: 'normal' in str(x))
    return_df.loc[mask_same, 'polluter'] = sum(mask_same) * ['UniquenessPolluter_uniform']
    return_df.loc[mask_normal, 'polluter'] = sum(mask_normal) * ['UniquenessPolluter_normal']
    return return_df


def compute_quality_feature_accuracy(df, quality):
    """
    Return the weighted average of the numerical columns' quality and the categorical columns' as quality of datasets
    polluted by FeatureAccuracyPolluter.
    """
    with open('../metadata.json', 'r') as file:
        metadata = loads(file.read())
    return_df = df.copy()
    aggregation = lambda x: number_categorical_columns / number_columns * x[0] + \
                            number_numerical_columns / number_columns * x[1]
    for dataset in df.dataset.unique():
        number_numerical_columns = len(metadata[dataset]['numerical_cols'])
        number_categorical_columns = len(metadata[dataset]['categorical_cols'])
        number_columns = number_numerical_columns + number_categorical_columns

        mask = (df.dataset == dataset) & (df.polluter == 'FeatureAccuracyPolluter')
        new_quality_values = return_df.loc[mask][quality].apply(aggregation)
        return_df.loc[mask, quality] = new_quality_values
    return return_df


def prepare_qualities(df):
    """
    The polluter provided the quality for the polluted train and test datasets. For the scenarios with only one of the
    polluted datasets, the corresponding value is taken over. For the scenario with both datasets being polluted, their
    qualities are combined according to train-test-split ratio of the dataset.
    ConsistentRepresentationPolluter had an extra seed (42) for no pollution, thus the train_clean_test_clean scenario
    contains no new information.
    """
    df.loc[df.scenario == 'train_clean_test_clean', 'quality'] = 1.0
    df.loc[df.scenario == 'train_polluted_test_clean', 'quality'] = df[
        df.scenario == 'train_polluted_test_clean'].quality_train
    df.loc[df.scenario == 'train_clean_test_polluted', 'quality'] = df[
        df.scenario == 'train_clean_test_polluted'].quality_test

    if df.quality_train.dtype == 'object':
        # Some polluters report multiple quality measures, that need to be handled

        # ConsistentRepresentationPolluter gives the whole dataset's quality first
        mask = (df.polluter.isin(CONSISTENT_REPRESENTATION_POLLUTER))
        df.loc[mask, 'quality_train'] = df[mask].quality_train.str[0]
        df.loc[mask, 'quality_test'] = df[mask].quality_test.str[0]
        df.loc[mask & ~(df.scenario == 'train_clean_test_clean'), 'quality'] = \
            df[mask & ~(df.scenario == 'train_clean_test_clean')].quality.str[0]

        # FeatureAccuracyPolluter reports one quality for categorical columns and one for numerical columns
        mask = df.polluter == 'FeatureAccuracyPolluter'
        df.loc[mask, 'quality_train'] = compute_quality_feature_accuracy(df[mask], 'quality_train')
        df.loc[mask, 'quality_test'] = compute_quality_feature_accuracy(df[mask], 'quality_test')
        mask = (df.polluter == 'FeatureAccuracyPolluter') & (df.scenario == 'train_polluted_test_clean')
        df.loc[mask, 'quality'] = df[mask].quality_train
        mask = (df.polluter == 'FeatureAccuracyPolluter') & (df.scenario == 'train_clean_test_polluted')
        df.loc[mask, 'quality'] = df[mask].quality_test

    # Combine qualities of both polluted datasets parts according to train-test-split ratio
    train_quality = df[df.scenario == 'train_polluted_test_polluted'].quality_train
    test_quality = df[df.scenario == 'train_polluted_test_polluted'].quality_test
    combined_quality = Series(train_quality) * 0.8 + Series(test_quality) * 0.2
    df.loc[df.scenario == 'train_polluted_test_polluted', 'quality'] = combined_quality

    df.drop(df[(df.scenario == 'train_clean_test_clean') &
               (df.polluter.isin(CONSISTENT_REPRESENTATION_POLLUTER))].index, inplace=True)

    return df


def results_json_to_dataframe(raw_results, datasets=None):
    """
    Convert the results json to Pandas dataframe
    :param raw_results:         json file
    :param datasets:            list of strings, default are all three classification datasets
    """

    if datasets is None:
        datasets = ['TelcoCustomerChurn.csv', 'SouthGermanCredit.csv', 'cmc.data', 'covid_data_pre_processed.csv']
        #datasets = ['TelcoCustomerChurn.csv']
        #datasets = ['covid_data_pre_processed.csv']
    keys = ['dataset', 'polluter', 'scenario', 'seed', 'algorithm', 'accuracy', 'f1-score', 'quality', 'quality_train',
            'quality_test', 'pollution_level', 'polluter_config']
    results_dict = {key: [] for key in keys}

    for dataset in datasets:
        if dataset not in raw_results.keys():
            continue
        polluters = list(raw_results[dataset].keys())
        for polluter in polluters:
            for config in list(raw_results[dataset][polluter].keys()):
                quality = raw_results[dataset][polluter][config]['quality']
                quality_train = quality['train']
                quality_test = quality['test']
                config_dict = literal_eval(config)
                pollution_level = config_dict.get('percentage_polluted_rows', -1)
                seed = config_dict['random_seed']
                scenarios = list(raw_results[dataset][polluter][config].keys())
                scenarios.remove('quality')
                for scenario in scenarios:
                    for algorithm in list(raw_results[dataset][polluter][config][scenario].keys()):
                        accuracy = raw_results[dataset][polluter][config][scenario][algorithm]['scoring']['accuracy']
                        f1 = raw_results[dataset][polluter][config][scenario][algorithm]['scoring']['macro avg'][
                            'f1-score']
                        results_dict['dataset'].append(dataset)
                        results_dict['polluter'].append(polluter)
                        results_dict['polluter_config'].append(config_dict)
                        results_dict['seed'].append(seed)
                        results_dict['pollution_level'].append(pollution_level)
                        results_dict['quality'].append(quality)
                        results_dict['quality_train'].append(quality_train)
                        results_dict['quality_test'].append(quality_test)
                        results_dict['f1-score'].append(f1)
                        results_dict['accuracy'].append(accuracy)
                        results_dict['scenario'].append(scenario)
                        results_dict['algorithm'].append(algorithm)
    df = DataFrame.from_dict(results_dict)
    df = prepare_qualities(df)
    df = concat([df[df.polluter != 'ConsistentRepresentationPolluter'],
                 separate_representation_results(df[df.polluter == 'ConsistentRepresentationPolluter'])])
    df = add_distribution_to_uniqueness(df)
    return df


def extract_performances_for_highest_qualities_consistent_representation(df):
    """
    ConsistentRepresentationPolluter had an extra seed (42) for no pollution, thus the train_clean_test_clean scenario
    contains no new information.
    """
    performances = {dataset: {} for dataset in df.dataset.unique()}
    for dataset in df.dataset.unique():
        for algorithm in df.algorithm.unique():
            filtered_df = df[(df.dataset == dataset) & (df.algorithm == algorithm) & (df.seed != '42')]
            if filtered_df.empty:
                continue
            else:
                accuracy = filtered_df.accuracy.iloc[0]
            f1_score = df[(df.dataset == dataset) &
                          (df.algorithm == algorithm) & (df.seed == '42')
                          ]['f1-score'].iloc[0]
            performances[dataset][algorithm] = {'f1-score_mean': f1_score, 'accuracy_mean': accuracy}
    return performances


def group_and_aggregate_df(df, polluter):
    """
    Group by dataset, polluter, scenario, algorithm & pollution level, aggregate mean and std.
    The seeds are dropped to average over their results.
    """
    df = df.copy()
    if polluter in CONSISTENT_REPRESENTATION_POLLUTER:
        df = df[df.polluter.isin(CONSISTENT_REPRESENTATION_POLLUTER)]
        by = ['dataset', 'polluter', 'scenario', 'algorithm', 'pollution_level']
    else:
        df = df[df.polluter == polluter]
        by = ['dataset', 'polluter', 'scenario', 'algorithm', 'quality']
    grouped_df = df.drop(columns=['seed']).groupby(by, as_index=False).agg([mean, std]).reset_index()

    # Flat the index after grouping and aggregating
    grouped_df.columns = ['_'.join(a) for a in grouped_df.columns.to_flat_index()]
    grouped_df.rename(columns={'dataset_': 'dataset', 'polluter_': 'polluter', 'scenario_': 'scenario',
                               'algorithm_': 'algorithm', 'quality_': 'quality', 'pollution_level_': 'pollution_level'},
                      inplace=True)
    if polluter == 'FeatureAccuracyPolluter':
        grouped_df = bin_feature_accuracy(grouped_df)
    return grouped_df


def separate_representation_results(df):
    """
    Splits the results dataframe into the results of the two differently configured ConsistentRepresentationPolluter
    (with two or five representations in our experiments).
    """
    two_representation_config_length = set()
    five_representation_config_length = set()
    new_df = df.copy()

    for dataset in df.dataset.unique():
        for _ in df.pollution_level.unique():
            param_lengths = df[(df.polluter.isin(CONSISTENT_REPRESENTATION_POLLUTER)) & (df.pollution_level != '0') &
                               (df.dataset == dataset)].polluter_config.apply(
                lambda x: len(x['new_representations'])).unique()

            two_representation_config_length.add(min(param_lengths))
            five_representation_config_length.add(max(param_lengths))

    mask_two = df.polluter_config.apply(lambda x: len(x['new_representations']) in two_representation_config_length)
    mask_five = df.polluter_config.apply(lambda x: len(x['new_representations']) in five_representation_config_length)

    new_df.loc[mask_two, 'polluter'] = 'ConsistentRepresentationPolluter_two'
    new_df.loc[mask_five, 'polluter'] = 'ConsistentRepresentationPolluter_five'

    return new_df


def bin_feature_accuracy(df):
    """
    As the FeatureAccuracyPolluter reports two quality metrics (one for numeric and one for categorical columns), which
    were combined in a weighted average quality, the performance results need to be binned.
    """
    return_df = df.copy()
    bins = arange(0, 1.05, 0.05)
    mask = return_df.polluter == 'FeatureAccuracyPolluter'
    return_df.loc[mask, 'binned'] = cut(return_df[mask].quality, bins=len(bins), labels=bins)
    return_df.binned = return_df.binned.astype(float32)

    return_df_feature_acc = return_df.loc[mask]
    return_df_rest = return_df.loc[~mask]
    grouped_feature_acc = return_df_feature_acc[['dataset', 'polluter', 'scenario', 'algorithm', 'binned',
                                                 'accuracy_mean', 'accuracy_std', 'f1-score_mean', 'f1-score_std']]\
        .groupby(['dataset', 'polluter', 'scenario', 'algorithm', 'binned'], as_index=False).agg([mean]).reset_index()
    grouped_feature_acc.columns = ['_'.join(a) for a in grouped_feature_acc.columns.to_flat_index()]
    grouped_feature_acc = grouped_feature_acc.rename(columns={
        'dataset_': 'dataset',
        'polluter_': 'polluter',
        'scenario_': 'scenario',
        'algorithm_': 'algorithm',
        'binned_': 'quality',
        'accuracy_mean_mean': 'accuracy_mean',
        'accuracy_std_mean': 'accuracy_std',
        'f1-score_mean_mean': 'f1-score_mean',
        'f1-score_std_mean': 'f1-score_std'})
    return_df_rest.drop(columns='binned', inplace=True)
    final_df = concat([grouped_feature_acc, return_df_rest])
    return final_df
