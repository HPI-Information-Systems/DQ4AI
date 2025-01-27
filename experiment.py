import pandas as pd

from typing import Any
from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    This class defines an abstract base class of a experiment
    """
    def __init__(self, name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, model: Any) -> None:
        """
        A predefined default constructor. Every experiment is composed of a dataframe,
        a model and the corresponding name of the experiment
        :param name: the name of the experiment
        :param train_df: data to train the experiment model on
        :param test_df: data to test the trained model
        :param model: the model that will be used for the experiment
        """
        self.name = name
        self.train = train_df
        self.test = test_df
        self.model = model

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        This method implements the actual logic of the experiment and returns a dataframe
        containing the results of the experiment.
        :return: a dataframe containing the results of the experiment.
        """
        pass
