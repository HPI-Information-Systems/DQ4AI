from typing import List, Union, Dict

from numpy import array, array_equal, logical_and, all
from pandas import DataFrame
from .interfaces import Polluter


class CompletenessPolluter(Polluter):
    """
    This class pollutes the dimension completeness. It can be configured with the percentages to pollute for certain
    columns or overall, and placeholder values for numeric and categorical columns. For each affected column, the polluter
    replaces the values at the specified percentage of positions with the placeholder values to simulate missing values.

    Since the polluter has to check for missing values in all feature columns (not only the ones to pollute) to compute
    the global quality measure, placeholder values for all columns are required, even if not all are polluted.
    """

    POLLUTION_LEVELS = [i / 20 for i in range(21)]

    @staticmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        spec_meta = metadata[ds_name]
        placeholder_values = spec_meta.get('placeholders', None)
        static_params = {
            'target_feature': spec_meta.get('target', None),
            'placeholder_numerical':  placeholder_values.get('numerical', None), 
            'placeholder_categorical': placeholder_values.get('categorical', None),
            'numerical_cols': spec_meta.get('numerical_cols', None),
            'categorical_cols': spec_meta.get('categorical_cols', None)
        }
        return static_params


    @classmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        configured_polluters = list()
        static_params = cls.get_static_params(metadata, dataset, ds_name)
        for rand_seed in metadata['random_seeds']:
            for pol_level in cls.POLLUTION_LEVELS:
                configured_polluters.append(cls(pollution_percentages=pol_level, random_seed=rand_seed, **static_params))
        return configured_polluters


    def __init__(self, pollution_percentages: Union[float, Dict[str, float]], target_feature: str,
                 placeholder_numerical: Union[int, Dict[str, str]], placeholder_categorical: Union[str, Dict[str, str]],
                 numerical_cols: List[str], categorical_cols: List[str], *args, **kwargs) -> None:
        """
        Initialize a CompletenessPolluter object by specifying the pollution percentages per column, the placeholder values
        and the target feature that should be excluded from pollution.

        :param pollution_percentages: The number specifying the pollution percentage of all features or the dictionary
                specifying a certain pollution percentage for each feature separately. Should not pollute the target
                feature.
        :param target feature: The target feature, since it should be excluded from pollution.
        :param placeholder_numerical: The numerical placeholder value for all numerical columns or one for each numeric feature
                column in the dataframe.
        :param placeholder_categorical: The categorical placeholder value for all categorical columns or one for each categorical
                feature column in the dataframe.
        :param categorical_cols: the list of all categorical features
        :param numerical_cols: the list of all numerical features
        """

        super().__init__(*args, **kwargs)

        if isinstance(pollution_percentages, dict):
            assert all(logical_and(array(list(pollution_percentages.values())) <= 1, array(list(
                pollution_percentages.values())) >= 0)), f"Pollution percentages should be between 0 and 1, but are " \
                                                         f"{pollution_percentages} "
            assert target_feature not in pollution_percentages, f"Target feature {target_feature} should not be " \
                                                                f"polluted"
        else:
            assert 0 <= pollution_percentages <= 1, f"Pollution percentage should be between 0 and 1, but is " \
                                                    f"{pollution_percentages}"

        assert numerical_cols is not None or categorical_cols is not None, \
            "Numeric or categorical columns have to be given"

        self.pollution_percentages = pollution_percentages
        self.target_feature = target_feature
        self.placeholder_numerical = placeholder_numerical
        self.placeholder_categorical = placeholder_categorical
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def pollute(self, df: DataFrame) -> DataFrame:
        """
        Pollutes the received dataframe. Reports existing pollution percentages in a column if there are already missing
        values in the dataframe. The existing missing values are excluded from the set of elements to pollute. If a
        column already contains the specified amount of missing values or more, no additional pollution is done for the
        column.

        :param df: the dataframe to be polluted
        :return: the polluted dataframe
        """
        all_feature_column_pollution_percentages = self._get_all_feature_column_percentages(
            df)
        self._check_all_placeholder_values_specified(df)

        # Ensure that the original dataframe is not changed
        df_polluted = df.copy()

        for column, pollution_percentage in all_feature_column_pollution_percentages.items():
            n_to_pollute = int(pollution_percentage * len(df_polluted.index))
            placeholder_value = self._get_placeholder_value(column)
            missing_indices, n_missing, existing_pollution_percentage = self._get_missing_for_column(
                df_polluted, column, placeholder_value)

            if n_missing >= n_to_pollute:
                print(f"""Column {column} already has pollution percentage of {existing_pollution_percentage}, which is larger
                    or equal to desired percentage {pollution_percentage}. No pollution to do for this column.""")
                continue

            elif n_missing > 0:
                print(f"""Column {column} already has pollution percentage of {existing_pollution_percentage}.
                This means that either there are missing values in the column or the
                chosen placeholder value actually lies within the data domain.""")

            n_to_pollute -= n_missing

            potential_indices_to_pollute = df_polluted.index.difference(missing_indices)
            indices_to_pollute = self.random_generator.choice(potential_indices_to_pollute, size=n_to_pollute,
                                                              replace=False)
            df_polluted.loc[indices_to_pollute, column] = placeholder_value

        return df_polluted

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame = None) -> float:
        """
        This method calculates the completeness of the polluted dataframe. Completeness is the percentage of
        observations that are not missing (i.e. not the placeholder value).

        :param df_polluted: the polluted dataframe (required)
        :param df_clean: the original dataframe (optional), which is not needed for completeness
        :return: the quality measure of the polluted dataframe
        """

        total_n_missing = 0
        possible_pollute_columns = df_polluted.columns.difference([self.target_feature])

        for column in possible_pollute_columns:
            placeholder_value = self._get_placeholder_value(column)
            _, n_missing, _ = self._get_missing_for_column(df_polluted, column, placeholder_value)
            total_n_missing += n_missing
        percentage_missing = total_n_missing / df_polluted[possible_pollute_columns].size

        return 1 - percentage_missing

    def _get_missing_for_column(self, df, column, placeholder_value):
        """
        Computes the indices, total number and percentage of missing values for a column.

        :param df: the dataframe to check for missing values
        :param column: the respective column of the dataframe
        :return: indices, total number and percentage of missing values
        """

        missing_indices = df.loc[df[column] == placeholder_value].index
        n_missing = len(missing_indices)
        existing_missing_percentage = n_missing / len(df.index)

        return missing_indices, n_missing, existing_missing_percentage

    def _check_all_placeholder_values_specified(self, df):
        """
        Checks whether the placeholder values for all columns are specified. This means that for numerical columns, there
        either has to be one single placeholder value for all of them or the numerical placeholder value dictionary has to
        contain a value for every column. Same holds for categorical columns.

        :param df: the dataframe to pollute
        """

        assert not isinstance(self.placeholder_categorical, dict) or array_equal(
            sorted(list(self.placeholder_categorical.keys())), sorted(self.categorical_cols)), \
            "Not all categorical placeholder values specified"

        assert not isinstance(self.placeholder_numerical, dict) or array_equal(
            sorted(list(self.placeholder_numerical.keys())), sorted(self.numerical_cols)), \
            "Not all numerical placeholder values specified"

    def _get_all_feature_column_percentages(self, df):
        """
        Gets the specified percentage for every feature column, which means every column in the dataframe except for the
        target column. If the pollution percentage is a scalar number, every feature column is assigned this percentage.
        Otherwise, the percentages are assigned according to the dictionary, where columns that are not included get a
        percentage of zero.

        :param df: the dataframe that is to be polluted
        :returns: a dictionary with percentage per feature column
        """

        feature_columns = df.columns.difference([self.target_feature])

        all_column_percentages = {}
        if isinstance(self.pollution_percentages, dict):
            all_column_percentages = {
                column: self.pollution_percentages[column] if column in self.pollution_percentages else 0
                for column in feature_columns}
        else:
            all_column_percentages = {
                column: self.pollution_percentages for column in feature_columns}

        return all_column_percentages

    def _get_placeholder_value(self, column):
        """
        Gets the placeholder value for the column based on whether the column is numerical or categorical.

        :param column: the column to get the placeholder value for
        :returns: a placeholder value for the column
        """

        if column in self.categorical_cols:
            return self._get_value_from_dict_or_scalar(self.placeholder_categorical, column)
        else:
            return self._get_value_from_dict_or_scalar(self.placeholder_numerical, column)

    @staticmethod
    def _get_value_from_dict_or_scalar(dict_or_scalar, key):
        """
        To wrap access to parameters that can be specified as a dictionary or a scalar value.

        :param dict_or_scalar: the element to retrieve a value from (if dict) or the value itself (scalar)
        :param key: the key to access dict_or_scalar if it's a dictionary
        :returns: the retrieved value for key if dictionary or the scalar value otherwise
        """

        if isinstance(dict_or_scalar, dict):
            return dict_or_scalar[key]
        else:
            return dict_or_scalar
