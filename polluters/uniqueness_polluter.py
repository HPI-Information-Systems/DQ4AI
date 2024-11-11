from typing import Optional, List, Union, Dict

from numpy import arange, floor, ceil, array, append, repeat, round
from pandas import DataFrame
from scipy.stats import zipfian
from functools import partial
from .interfaces import Polluter


class UniquenessPolluter(Polluter):
    """
    The UniquenessPolluter adds duplicates to a dataframe.
    """

    # To get linearly decreasing qualities with 10% steps, as the relationship here is uniqueness = 1 / duplicate_factor.
    # We make the cut at a duplicate_factor of 5, because for lower quality the duplicate_factor
    # increases faster than linearly, e.g., would have to be 10 for a quality of 0.1,
    # which would make the experiments much slower.
    # Rounding is necessary because otherwise 1 becomes 0.999999998
    DUPLICATE_FACTORS = round(1 / arange(1.0, 0.15, -0.05), 10)
    DISTRIBUTIONS = {"same": [{}], "normal": [{"loc": 1.0, "scale": 5.0}]}

    @staticmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        static_params = {
            'target_feature': metadata[ds_name].get('target', None),
        }
        return static_params

    @classmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        configured_polluters = list()
        static_params = cls.get_static_params(metadata, dataset, ds_name)
        for rand_seed in metadata['random_seeds']:
            for duplicate_factor in cls.DUPLICATE_FACTORS:
                for distribution_name, distribution_params_list in cls.DISTRIBUTIONS.items():
                    for distribution_params in distribution_params_list:
                        configured_polluters.append(
                            cls(duplicate_factor=duplicate_factor, distribution_function_name=distribution_name,
                                distribution_function_parameters=distribution_params, random_seed=rand_seed,
                                **static_params))
        return configured_polluters

    def __init__(self, duplicate_factor: float, distribution_function_name, distribution_function_parameters,
                 target_feature: str, *args, **kwargs) -> None:
        """
        Initialization of the Uniqueness Polluter by specifying the parameters for duplication and the target variable

        :param duplicate_factor: The scaling factor >=1 that specifies how many samples should be present in the dataset
            after duplication. E.g. for a duplicate factor of 2, the polluted dataset will be twice the size of the
            dataset without duplicates. If the factor is 1, the dataset without duplicates is returned
        :param distribution_function_name: Specifies the distribution function that is used to sample amount numbers of
            duplicates. The amount numbers are assigned to the unique samples in the dataset. If some numbers are <=1,
            they are set to 1 to prevent the possibility of non-termination. The duplicates are then created by randomly
            choosing from the assigned amount numbers. Currently supported distributions for sampling the amount of
            numbers:
                - uniform
                - normal
                - zipf
                - same (assigns an amount number of 1 to each sample)
        :param distribution_function_parameters: Dictionary that contains the parameters for the specified distribution
            function. For the following distributions it has to contain:
                - uniform: low, high
                - normal: loc (represents mean), scale (represents stddev)
                - zipf: a (alpha that specifies the function bending), n (maximum amount number)
                - same: nothing (an empty dictionary)
        :param target_feature: The target feature, as it is relevant for preserving class balance
        """

        super().__init__(*args, **kwargs)

        if distribution_function_name == 'uniform':
            assert {"low", "high"} <= distribution_function_parameters.keys()
            self.distribution_function = partial(self.random_generator.uniform,
                                                           low=distribution_function_parameters['low'],
                                                           high=distribution_function_parameters['high'])

        elif distribution_function_name == 'normal':
            assert {"loc", "scale"} <= distribution_function_parameters.keys()
            self.distribution_function = partial(self.random_generator.normal,
                                                           loc=distribution_function_parameters["loc"],
                                                           scale=distribution_function_parameters["scale"])

        elif distribution_function_name == 'zipf':
            assert {"a", "n"} <= distribution_function_parameters.keys()
            self.distribution_function = partial(zipfian.rvs, a=distribution_function_parameters["a"],
                                                           n=distribution_function_parameters["n"],
                                                           random_state=self.random_seed)

        elif distribution_function_name == 'same':
            self.distribution_function = partial(self.random_generator.uniform, low=1,
                                                           high=1)

        else:
            raise ValueError("Distribution function not supported")

        assert duplicate_factor >= 1, f"Duplicate factor should be >=1 for the pollution to make sense, but is {duplicate_factor}"

        self.distribution_function_name = distribution_function_name
        self.distribution_function_parameters = distribution_function_parameters

        self.duplicate_factor = duplicate_factor
        self.target_variable = target_feature

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame = None) -> float:
        """
        This method calculates the quality measure of the polluted dataframe.
        Formula: #unique values -1 / #values (n) - 1

        :param df_polluted: the polluted dataframe
        :param df_clean: not used in this polluter
        :return: the quality measure of the polluted dataframe
        """
        return (len(df_polluted.drop_duplicates()) - 1) / (len(df_polluted) - 1)

    def pollute(self, df: DataFrame) -> DataFrame:
        """
        The method takes a dataframe and pollutes it. It starts by removing duplicates and samples numbers of duplicates
        from the specified distribution with the specified parameters. The values of the distribution are used to assign
        an amount number of duplicates to each sample in the dataframe. Then, for each class, the polluter computes the
        number of duplicates that should be created for the class (while keeping class balance). It randomly chooses
        from the unique class samples and adds as many duplicates of the sample as assigned to the dataset, which is
        repeated until the number of duplicates to create is reached.

        For continuous target variables, the duplication does not work. Thus, a previous discretization is required.

        :param df: The dataframe to be polluted
        :return: The polluted dataframe
        """

        df_polluted = df.drop_duplicates().reset_index(drop=True)

        # to save runtime and not shuffle the dataframe if no duplicates are added
        if self.duplicate_factor == 1:
            return df_polluted

        duplicate_distr = self.distribution_function(size=len(df_polluted))

        # to avoid negative number of duplicates or zero, otherwise the polluter might not terminate
        duplicate_distr[duplicate_distr <= 1] = 1

        target_indices = array([])
        class_indices = df_polluted.groupby(
            [self.target_variable]).groups

        for c_indices in class_indices.values():
            num_duplicates_to_sample = int(
                ceil((self.duplicate_factor - 1) * len(c_indices)))
            num_sampled_duplicates_per_class = 0

            while num_sampled_duplicates_per_class < num_duplicates_to_sample:
                c_ind = self.random_generator.choice(c_indices)
                num_duplicates = floor(duplicate_distr[c_ind])

                if num_sampled_duplicates_per_class + num_duplicates > num_duplicates_to_sample:
                    num_duplicates = num_duplicates_to_sample - num_sampled_duplicates_per_class

                target_indices = append(
                    target_indices, repeat(c_ind, num_duplicates))
                num_sampled_duplicates_per_class += num_duplicates

        # fill dataframe with duplicated rows based on calculated indices
        df_polluted = df_polluted.append(df_polluted.iloc[target_indices])
        return df_polluted.sample(frac=1, replace=False, random_state=self.random_seed).reset_index(drop=True)

    def get_pollution_params(self) -> str:
        """
        Returns the string representation of the dictionary of all member variables, excluding the random generator.
        This is hashed and used for file naming purposes later on, which is why the object location reference for the
        random generator needs to be excluded (it is not always at the same location in memory).

        Remove function parameters

        :returns: string representation of the member variable dictionary, excluding random generator field
        :rtype: str
        """
        return str({k: str(v) for k, v in vars(self).items() if k != 'random_generator' and k != 'distribution_function'})
