from numpy import random
from pandas import DataFrame

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class Polluter(ABC):
    """
    This class defines an abstract base class of a polluter.
    """

    def __init__(self, random_seed: int) -> None:
        """
        This method initializes the random seed and random generator.
        :param random_seed: the random seed
        """

        self.random_seed = random_seed
        self.random_generator = random.default_rng(random_seed)

    @staticmethod
    @abstractmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        """
        Generates a dictionary of static parameters to initialize the own specialized polluter instances from.
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
        raise NotImplementedError('Please implement the get_static_params function in each specialized polluter!')

    @classmethod
    @abstractmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        """
        Configures the special polluter, setting its parameters based on the metadata, dataset, dataset name
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

        # Hint: Implement and use the get_static_params function for cleaner code!
        # Basic structure example:
        #
        # configured_polluters = list()
        # static_params = cls.get_static_params(metadata, dataset, ds_name)
        # for rand_seed in metadata['random_seeds']:
        #     for pol_level in [i / 20 for i in range(21)]:
        #         configured_polluters.append(cls(pol_level=pol_level, random_seed=rand_seed, **static_params))
        # return configured_polluters
        raise NotImplementedError('Please implement the configure function in each specialized polluter!')

    @abstractmethod
    def pollute(self, df: DataFrame) -> DataFrame:
        """
        The pollute method takes care of polluting the received dataframe.

        :param df: the dataframe that is to be polluted
        :return: the polluted dataframe
        """
        pass

    @abstractmethod
    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame) \
            -> Union[float, Tuple[float, float]]:
        """
        This method calculates the quality measure of the polluted dataframe.

        :param df_polluted: the polluted dataframe (required)
        :param df_clean: the original dataframe (optional)
        :return: the quality measure/s of the polluted dataframe
        """
        pass

    def __call__(self, df: DataFrame) -> Tuple[DataFrame, Union[float, Tuple[float, float]]]:
        """
        The __call__ method allows the instantiated objects of the Polluter class to be called like functions. This
        will return both the polluted dataset and the quality measure calculated for it.

        :param df: the dataframe that is to be polluted
        :return: the polluted dataframe and its quality measure/s
        """

        df_polluted = self.pollute(df)
        return df_polluted, self.compute_quality_measure(df_polluted, df)

    def get_pollution_params(self) -> str:
        """
        Returns the string representation of the dictionary of all member variables, excluding the random generator.
        This is hashed and used for file naming purposes later on, which is why the object location reference for the
        random generator needs to be excluded (it is not always at the same location in memory).

        :returns: string representation of the member variable dictionary, excluding random generator field
        :rtype: str
        """
        return str({k: str(v) for k, v in vars(self).items() if k != 'random_generator'})
