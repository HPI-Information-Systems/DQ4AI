from typing import Dict, Union, Tuple, List

from .interfaces import Polluter
from pandas import DataFrame, Series
from numpy import sort


class ConsistentRepresentationPolluter(Polluter):
    """
    Tabular data is consistent in its representation if no column has two or more unique values that are semantically
    equivalent but just expressed differently. This class pollutes implements a method to pollute the quality dimension
    consistent representation and a method to compute two quality measures (See compute_quality_measure). After running
    the pollution method, you can get the newly introduced representations via the attribute.
    :param percentage_polluted_rows:    float, in [0,1]
    :param number_of_representations:   2d-dict with names of to be polluted categorical column names as outer keys,
                                        existing unique column values being int or str as inner keys and the number
                                        of representations for that value as values
    :param num_pollutable_columns:      int, in [1, |columns| - 1]

    """

    def __init__(self, random_seed: int, percentage_polluted_rows: float, num_pollutable_columns: int,
                 number_of_representations: Dict[str, Dict[Union[str, int], int]]) -> None:
        super().__init__(random_seed)
        self.new_representations = {}
        self.percentage_polluted_rows = percentage_polluted_rows
        self.num_pollutable_columns = num_pollutable_columns
        self.number_of_representations = number_of_representations

    @staticmethod
    def get_static_params(metadata: dict, dataset: DataFrame, ds_name: str) -> dict:
        """
        Generates a dictionary of static parameters to initialize the ConsistentRepresentationPolluter instances
        from. This is done by reading the appropriate fields in the metadata dictionary section corresponding to the
        given dataset name and/or analyzing the given dataset. The ConsistentRepresentationPolluter needs the number of
        categorical columns and its names.
        :param metadata:    dict, dataset metadata dictionary read from the metadata.json file
        :param ds_name:     string, name of the dataset file - same as the key in the metadata dictionary
        :param dataset:     pd.DataFrame, raw dataset as read from disk, not needed for ConsistentRepresentationPolluter

        :returns: dict, parameter dictionary to use in polluter instance initialization
        """
        return {'num_pollutable_columns': len(metadata[ds_name]['categorical_cols'])}

    @classmethod
    def configure(cls, metadata: dict, dataset: DataFrame, ds_name: str) -> List['Polluter']:
        """
        Configures the ConsistentRepresentationPolluter, setting its parameters based on the metadata, dataset, dataset
        name and random seed provided.
        :param metadata:    dict, dataset metadata dictionary read from the metadata.json file
        :param dataset:     pd.DataFrame, raw dataset as read from disk
        :param ds_name:     string, name of the dataset file - same as the key in the metadata dictionary

        :returns: list of configured polluter instances
        """
        configured_polluters = list()
        static_params = cls.get_static_params(metadata, dataset, ds_name)

        # Return list of configured polluters empty if the dataset has no pollutable columns
        if static_params['num_pollutable_columns'] == 0:
            return list()

        # Add original baseline (no pollution)
        configured_polluters.append(cls(percentage_polluted_rows=0, **static_params, random_seed=42,
                                        number_of_representations={}))

        # Add polluters for all random seeds and all percentages of pollution in [0.1, 1]
        for random_seed in metadata['random_seeds']:
            for percentage_polluted_rows in [i / 20 for i in range(1,21)]:
                for count in [2, 5]:
                    num_of_repr = {column: {value: count for value in sort(Series.unique(dataset[column]))}
                                   for column in metadata[ds_name]['categorical_cols']}
                    configured_polluters.append(cls(percentage_polluted_rows=percentage_polluted_rows, **static_params,
                                                    random_seed=random_seed, number_of_representations=num_of_repr))
        return configured_polluters

    def _check_pollution_input(self, df: DataFrame):
        if self.percentage_polluted_rows and (not 0 <= self.percentage_polluted_rows <= 1):
            raise ValueError('Percentage_polluted_rows must be between 0 and 1.')
        for column in self.number_of_representations.keys():
            if column not in df.columns:
                raise ValueError(f"No column with name {column} in given dataframe.")
            for original in self.number_of_representations[column]:
                if self.number_of_representations[column][original] < 2:
                    raise ValueError(f"Number of representations for an original value must be greater than two.\n"
                                     f"If you do not want to pollute value {original} in column {column}, just omit it "
                                     f"from number_of_representations.")

    def pollute(self, df: DataFrame) -> DataFrame:
        """
        Pollutes the quality dimension consistent representation for a proportion of rows of the given dataframe df
        specified in percentage_polluted_rows during initialisation. Only categorical values can be polluted. For each
        column the to be polluted rows are picked randomly again.

        :param df:  Pandas dataframe
        """

        self._check_pollution_input(df)

        # Stop here if no pollution is wanted
        if self.percentage_polluted_rows == 0:
            return df

        polluted_df = df.copy()
        self._generate_new_representations(df)

        for column in self.number_of_representations.keys():
            # Specify to be polluted rows in current column
            number_polluted_rows = int(self.percentage_polluted_rows * len(df.index))
            indices_to_pollute = self.random_generator.choice(df.index, size=number_polluted_rows, replace=False)

            for original_value in self.number_of_representations[column].keys():

                # Only pollute a row if its original value shall be polluted
                if self.new_representations[column][original_value]:
                    for row in indices_to_pollute:
                        if df.iloc[row][column] == original_value:
                            polluted_df.at[row, column] = \
                                self.random_generator.choice(self.new_representations[column][original_value])

        return polluted_df

    def _generate_new_representations(self, df: DataFrame):
        """
        Generates new representations for categorical values being strings or integers.
        If the values are strings, new representations get a trailing incrementing number.
        If the values are integers, new representations are added after the maximum existing value.
        """

        self._check_pollution_input(df)
        representations = {}
        for column in self.number_of_representations.keys():
            representations[column] = {}
            # current_max is only needed for integer columns (see following else branch), but its scope has to be here
            current_max = None
            for original_value in self.number_of_representations[column].keys():
                if isinstance(original_value, str):
                    representations[column][original_value] = \
                        [str(original_value) + '-' + str(i) for i in
                         range(1, self.number_of_representations[column][original_value])]
                else:
                    # Initialise current_max to column's maximum
                    current_max = current_max if current_max else df[column].max()
                    representations[column][original_value] = \
                        [current_max + i for i in range(1, self.number_of_representations[column][original_value])]
                    current_max = max(representations[column][original_value])

        self.new_representations = representations

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame) -> Tuple[float, float]:
        """
        Counts the minimal number of changes needed to obtain consistent representation in df_polluted per column and
        normalise them by the number of rows. By subtracting this from one, this gives column-wise quality measures
        which then are being averaged over:
        1. all columns (without target column)
        2. all pollutable columns (without target column), specified during initialisation
        to obtain the dataframe's quality measure. Returns overall quality measure and quality measure over all
        pollutable columns. Only columns holding categorical values or integers that can be treated as categoricals are
        pollutable by this polluter method.

        :param df_polluted: Pandas dataframe
        :param df_clean:    Pandas dataframe
        """

        if (df_clean.columns != df_polluted.columns).all():
            raise ValueError('Column names of original and polluted dataframes must match.')
        # If no pollution was applied the quality is 100%
        if self.percentage_polluted_rows == 0:
            return 1.0, 1.0
        if self.num_pollutable_columns < 1 or self.num_pollutable_columns > df_clean.shape[1] - 1:
            raise ValueError('Number of pollutable columns must be between 1 and |columns| - 1 as the target column '
                             'must not be counted.')

        pollution = dict.fromkeys(df_clean.columns, 0.0)

        for column in self.new_representations.keys():
            for original, representations in self.new_representations[column].items():
                representations = representations.copy()
                representations.append(original)

                # Count occurrences for all representations of an original
                # Representations may be non-existent due to randomness and sparseness of polluted rows
                repr_num = {i: df_polluted[column].value_counts().get(i, 0) for i in representations}
                most_frequent_value = max(repr_num, key=repr_num.get)
                del repr_num[most_frequent_value]
                min_change_num = sum(repr_num.values())
                pollution[column] += min_change_num

            # Normalize by number of rows
            pollution[column] = pollution[column] / df_polluted.shape[0]

        measure_overall = 1 - (sum(pollution.values()) / (df_polluted.shape[1] - 1))
        measure_pollutable = 1 - (sum(pollution.values()) / self.num_pollutable_columns)
        return measure_overall, measure_pollutable
