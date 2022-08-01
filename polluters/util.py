from typing import List

from pandas import DataFrame

from polluters import Polluter


def pollute(df: DataFrame, polluters: List[Polluter]) -> DataFrame:
    """
    This function applies a set of polluters to the specified dataframe.
    The polluters are executed in the specified order.

    :param df: the dataframe to be polluted
    :param polluters: a list of polluters
    :return: a polluted dataframe
    """
    for polluter in polluters:
        df = polluter(df)
    return df
