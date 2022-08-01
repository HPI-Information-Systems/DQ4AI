import numpy as np
import pandas as pd
from typing import Optional, List, Union


def discretize_column(df: pd.DataFrame, column: str, discr_column, bins: Union[str, List[int], List[float]]):
    """
    Adds a column that contains the discretized version of a (continous) column for regression.
    The discretized values are written as indices for the respective bins.

    :param df: the dataframe to discretize a column of
    :param column: the column to discretize
    :param discr_column: name the discretized column should have
    :param bins: The bins for discretization. Can be a mode (e.g. "auto") for automatic computation or a number of bin edges
    (first and last should be included)
    :returns a modified data frame with the additional discretized column
    """
    # extend min and max range by 1 % if necessary to include min and max values
    if isinstance(bins, str):
        range_min = df[column].min()
        range_max = df[column].max()
        range_min -= range_min / 100
        range_max += range_max / 100

        bins = np.histogram_bin_edges(df[column], bins=bins, range=(range_min, range_max))

    else:
        if bins[0] == df[column].min():
            bins[0] -= bins[0] / 100
        if bins[-1] == df[column].max():
            bins[-1] += bins[-1] / 100

    discr_values = pd.cut(df[column], bins=bins, labels=False)
    df_with_discretized = df.copy()
    df_with_discretized[discr_column] = discr_values

    return df_with_discretized
