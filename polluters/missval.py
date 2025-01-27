import pandas as pd
from .interfaces import Polluter


class MissingValuePolluter(Polluter):
    """
    This class implements a concrete implementation of the abstract base class Polluter.
    """
    def pollute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        As an example, this polluter simply returns a random selection of the rows of the received dataframe.

        :param df: the dataframe to be polluted
        :return: the polluted dataframe
        """
        return df.sample(50)
