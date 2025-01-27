import logging

from .interfaces import Polluter
from typing import Dict, List, Tuple, Union

from pandas import DataFrame


class FeatureAccuracyPolluter(Polluter):
    """
    The FeatureAccuracyPolluter implements a concrete implementation of the abstract base class Polluter and is
    designed to change the accuracy of a set of given features. This is done either by adding a certain noise to all
    samples of a feature (numerical features) or by swapping a certain amount of samples to a random different category
    (categorical features).
    """

    @staticmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        """
        Generates a dictionary of static parameters to initialize the own specialized polluter instances from. This is
        done by reading the appropriate fields in the metadata dictionary section corresponding to the given dataset
        name and/or analyzing the given dataset.

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
        if spec_meta.get('numerical_cols') is not None and spec_meta.get('categorical_cols') is not None:
            numerical_cols = spec_meta['numerical_cols']
            categorical_cols = spec_meta['categorical_cols']
        else:
            raise ValueError(f'Categorical and numerical columns both have to be defined for {ds_name} in the '
                             f'metadata file. (keys: "numerical_cols" and "categorical_cols")')

        return {
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }

    @classmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        """
        Configures the special polluter, setting its parameters based on the metadata, dataset, dataset name
        and random seed provided. In addition to fixed per-dataset parameters, each polluter may specify, in this class,
        ranges from which to pull flexible parameters (e.g. percentage of samples to pollute). This function returns a
        list of configured polluter instances.

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
            for pollution_level in [i / 20 for i in range(21)]:
                configured_polluters.append(cls(
                    pollution_levels=pollution_level,
                    random_seed=rand_seed,
                    **static_params
                ))
        return configured_polluters

    def __init__(self, pollution_levels: Union[float, Dict[str, float]], categorical_cols: List[str],
                 numerical_cols: List[str], *args, **kwargs) -> None:
        """
        This method initializes a specific FeatureAccuracyPolluter object by specifying which numerical and categorical
        features exist and which should be polluted with which pollution level.

        :param pollution_levels: the number specifying the pollution level of all features or the dictionary specifying
        a certain pollution level for each feature to be polluted separately
        :param categorical_cols: the list of all categorical features
        :param numerical_cols: the list of all numerical features
        """

        super().__init__(*args, **kwargs)

        self._categorical_cols = categorical_cols
        self._numerical_cols = numerical_cols

        if isinstance(pollution_levels, float):
            if not 0 <= pollution_levels <= 1:
                raise ValueError('Pollution level must be between 0 and 1.')
            # Specify same pollution level for all features
            self._pollution_levels = {col: pollution_levels for col in self._categorical_cols + self._numerical_cols}
        else:
            if not all(0 <= v <= 1 for v in pollution_levels.values()):
                raise ValueError('Pollution level of every feature must be between 0 and 1.')
            self._pollution_levels = pollution_levels

    def pollute(self, df: DataFrame) -> DataFrame:
        """
        This method takes care of polluting the received dataframe. It calls two different methods for the treatment of
        categorical and numerical features. Constant features must be removed for the pollution of categorical columns.

        :param df: the dataframe to be polluted
        :return: the polluted dataframe
        """

        # Ensure that the original dataframe is not changed
        df_polluted = df.copy(deep=True)

        for col in self._pollution_levels.keys():
            if col in self._categorical_cols:
                self._change_categories_of_col(df_polluted, col)
            elif col in self._numerical_cols:
                self._add_gaussian_noise_to_col(df_polluted, col)
            else:
                raise Warning(f'Feature "{col}" does not exist in the dataframe but appears in the pollution levels '
                              f'dictionary. Therefore, no pollution will be applied to it.')

        return df_polluted

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame) -> Tuple[float, float]:
        """
        This method calculates the two quality measures of all categorical and all numerical features of the polluted
        dataframe. The per-feature qualities (partial qualities) are averaged to compute the final quality measures.
        We have to return two quality measures as we pollute categorical and numerical features differently which leads
        to different calculations and thus different value ranges of the per-feature qualities for categorical and
        numerical features.

        For categorical features, the per-feature quality is defined as the percentage of how many
        samples were not changed. For numerical features, the per-feature quality is defined as the average
        value change normalized by the mean of the original values subtracted from 1.

        :param df_polluted: the polluted dataframe (required)
        :param df_clean: the original dataframe (required)
        :return the two quality measures of the categorical and numerical features of the polluted dataframe
        """

        # Ensure comparable indices for experiments applying train-test split
        df_polluted = df_polluted.reset_index(drop=True)
        df_clean = df_clean.reset_index(drop=True)

        cat_partial_qualities, num_partial_qualities = list(), list()

        for col in self._categorical_cols + self._numerical_cols:
            if col in self._categorical_cols:
                # Get number of mismatches (meaning how many samples were changed)
                n_mismatches = len(df_polluted[df_polluted[col] != df_clean[col]])
                # Calculate percentage of how many samples stayed the same
                cat_partial_qualities.append(1.0 - (n_mismatches / len(df_polluted.index)))
            else:
                abs_distances = abs(df_clean[col] - df_polluted[col])

                if max(abs_distances) != 0.0:
                    # Subtract average value change normalized by mean of original values from 1
                    num_partial_qualities.append(1.0 - (sum(abs_distances) / len(df_polluted.index) / df_clean[col].mean()))
                else:
                    # If no pollution was applied, pass per-feature quality of 1
                    num_partial_qualities.append(1.0)

        # Catch if lists contain values to avoid division by zero
        if not cat_partial_qualities:
            cat_quality = None
        else:
            cat_quality = sum(cat_partial_qualities) / len(cat_partial_qualities)

        if not num_partial_qualities:
            num_quality = None
        else:
            num_quality = sum(num_partial_qualities) / len(num_partial_qualities)

        # Return average of partial qualities for both feature types
        return cat_quality, num_quality

    def _change_categories_of_col(self, df: DataFrame, col: str) -> None:
        """
        This method swaps the current categories of certain samples in a given feature for a random different
        category. How many samples to pollute is determined by the pollution level. Constant column are skipped as no
        other value for changing is available otherwise.

        :param df: the dataframe to be polluted
        :param col: the name of the feature to be polluted
        """

        # Skip constant columns (otherwise changing of categorical values will fail)
        if df[col].nunique() == 1:
            logging.debug(f'Categorical feature {col} was not polluted by the FeatureAccuracyPolluter as it is '
                          f'constant.')
            return

        # Determine which samples to pollute
        n_to_pollute = int(self._pollution_levels[col] * len(df.index))
        indices_to_pollute = self.random_generator.choice(df.index, size=n_to_pollute, replace=False)

        # Exchange current category with random different category
        all_categories = df[col].unique()

        if len({0, 1}.symmetric_difference(all_categories)) == 0:
            # Invert values for binary features (to improve runtime)
            df.loc[indices_to_pollute, col] = (~df.loc[indices_to_pollute, col].astype(bool)).astype(int)
        else:
            df.loc[indices_to_pollute, col] = df.loc[indices_to_pollute, col]\
                .apply(lambda v: self.random_generator.choice([c for c in all_categories if c != v], size=1)[0])

    def _add_gaussian_noise_to_col(self, df: DataFrame, col: str) -> None:
        """
        This method applies a gaussian noise to all samples of a specific feature. The pollution level defined for each
        feature determines the standard deviation of the distribution sample and thus denotes how wide it can be spread.

        :param df: the dataframe to be polluted
        :param col: the name of the feature to be polluted
        """

        # Catch if no pollution should be performed (because distribution sample will only consist of zeros anyway)
        if self._pollution_levels[col] == 0.0:
            return

        gaussian_mean = 0.0
        dist_sample = [0.0]

        # Ensure that no value in distribution sample is equal to zero (to ensure that all samples are polluted)
        while 0.0 in dist_sample:
            # Generate distribution sample with mean of zero and pollution level as standard deviation
            dist_sample = self.random_generator.normal(gaussian_mean, self._pollution_levels[col], size=len(df.index))

        # Extract mean of original feature value distribution (to convert noise to feature value scale)
        col_mean = df[col].mean()

        # Compute and add noise
        noise = dist_sample * col_mean
        df[col] += noise
