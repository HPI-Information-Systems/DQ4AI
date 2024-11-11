from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from sys import stdout
from logging import DEBUG, FileHandler, StreamHandler, basicConfig
from pandas import DataFrame, concat, Categorical, get_dummies
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from numpy.random import seed, choice


def one_hot_encode(df_train: DataFrame, df_test: DataFrame, categorical_columns: List[str])\
        -> Tuple[DataFrame, DataFrame]:
    """
    Apply one-hot-encoding on categorical columns on train and test dataset.
    :param df_train:            Training dataset
    :param df_test:             Test dataset
    :param categorical_columns: List of names of the categorical columns
    """

    # Make a deep copy to prevent changes to the original datasets
    df_train = df_train.copy()
    df_test = df_test.copy()
    df = concat([df_train, df_test])

    def unique_in_categorical(df, categorical_columns):
        return all(df[col].nunique() == 1 for col in categorical_columns)

    if unique_in_categorical(df, categorical_columns):
        # If all entries in the dataset are the same, skip dropping the first column in one-hot encoding
        drop_first = False
    else:
        drop_first = True
    # Converting the categorical columns to pd.Categorical with categories from the whole dataset df is necessary to not
    # miss any categories only present in one of the datasets out
    for categorical_column in categorical_columns:
        df_train[categorical_column] = Categorical(df_train[categorical_column],
                                                   categories=df[categorical_column].unique())
        df_test[categorical_column] = Categorical(df_test[categorical_column],
                                                  categories=df[categorical_column].unique())

    df_train = get_dummies(df_train, columns=categorical_columns, drop_first=True)
    df_test = get_dummies(df_test, columns=categorical_columns, drop_first=True)

    return df_train, df_test


def start_logging(log_level=DEBUG, append=False, cmd_out=False, data_dir=Path('data/')):
    """
    Configures and starts logging for the project using the logging library.

    :param log_level: logging library's level of detail for log prints
    :type log_level: int (from logging log level enum)
    :param append: determines if log file should be opened in append mode, defaults to False
    :type append: bool
    :param cmd_out: whether to also print logs to commandline
    :type cmd_out: bool
    :param data_dir: path to data directory
    :type data_dir: pathlib.Path
    """
    log_file_path = data_dir / f'logs/experiment_{datetime.now():%Y_%m_%d_%H_%M_%S}.log'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = 'a' if append else 'w'
    handlers = [FileHandler(log_file_path, mode=file_mode)]
    if cmd_out:
        handlers.append(StreamHandler(stdout))

    basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=handlers
    )


def get_majority_baseline_performance(labels):
    """This function lets you calculate the baseline performance for a classifier
    that only predicts the majority class.

    Args:
        labels (np.array or List): the real labels of the dataset for all records

    Returns:
        acc (float) : accuracy
        f1 (float) : macro averaged f1-score
    """
    majority_class = Counter(labels).most_common()[0][0]
    acc = accuracy_score(labels, [majority_class] * len(labels))
    f1 = f1_score(labels, [majority_class] * len(labels), average="macro")

    return acc, f1


def get_ratio_baseline_performance(labels):
    """This function lets you calculate the baseline performance for a classifier
    that predicts the class with the probability of their relative occurrence.

    Args:
        labels (np.array or List): the real labels of the dataset for all records

    Returns:
        acc (float) : accuracy
        f1 (float) : macro averaged f1-score
    """
    frequency_dict = {}
    for (value, count) in Counter(labels).most_common():
        frequency_dict[value] = count / len(labels)
    seed(1)
    baseline_preds = choice(list(frequency_dict.keys()), size=len(labels), p=list(frequency_dict.values()))
    acc = accuracy_score(labels, baseline_preds)
    f1 = f1_score(labels, baseline_preds, average="macro")

    return acc, f1
