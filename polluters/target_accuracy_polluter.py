from .interfaces import Polluter
from typing import List
from numpy import sum, mean, abs, reshape, delete, unique, where
from pandas import DataFrame


class TargetAccuracyPolluter(Polluter):
    """
    The TargetAccuracyPolluter changes the accuracy of the target variable.

    NOTE: If you are doing regression you have to have FLOAT TARGETS as the datatype is inferred automatically in the
    get_static_params
    """

    def __init__(self, pollution_level: float, target_col: str, is_categorical: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert pollution_level >= 0. and (pollution_level <= 1.0 or (not is_categorical and pollution_level <= 1.25)), \
            f"Expected a pollution level between 0 and 1.25, got: {pollution_level}"

        self.pollution_level = pollution_level
        self.target_col = target_col
        self.is_categorical = is_categorical

    @staticmethod
    def get_static_params(metadata: dict, df: DataFrame, ds_name: str) -> dict:
        target = metadata[ds_name]["target"]
        # Integers are not numerical but rather already encoded categories
        if df[target].dtype.kind in "fc":
            is_categorical = False
        elif df[target].dtype.kind in "mMV":
            raise ValueError(f"Target variable has unexpected datatype: {df[target].dtype}")
        else:
            is_categorical = True
        return {
            "is_categorical": is_categorical,
            "target_col": target}

    @classmethod
    def configure(cls, metadata: dict, df: DataFrame, ds_name: str) -> List[Polluter]:
        configured_polluters = list()
        static_params = cls.get_static_params(metadata, df, ds_name)
        if static_params.get("is_categorical"):
            full_pollution_list = [i / 20 for i in range(21)]  # from 0 to 1.0
            pollution_list = full_pollution_list
        else:
            full_pollution_list = [i / 20 for i in range(26)]  # from 0 to 1.25
            pollution_list = full_pollution_list
        for rand_seed in metadata['random_seeds']:
            for pollution_level in pollution_list:
                configured_polluters.append(cls(
                    pollution_level=pollution_level,
                    random_seed=rand_seed,
                    **static_params
                ))
        return configured_polluters

    def pollute(self, df: DataFrame) -> DataFrame:
        assert self.target_col in df.keys(), f"Target column {self.target_col} is not a column of the given DataFrame"

        df_polluted = df.copy()

        if self.is_categorical:
            polluted_targets = self._swap_categoricals(df_polluted)
            df_polluted[[self.target_col]] = polluted_targets
        else:
            polluted_targets = self._add_noise(df_polluted)
            df_polluted[[self.target_col]] = polluted_targets

        return df_polluted

    def compute_quality_measure(self, df_polluted: DataFrame, df_clean: DataFrame) -> float:
        if self.is_categorical:
            return float(sum(df_polluted[[self.target_col]] == df_clean[[self.target_col]]) / len(df_clean.index))
        else:
            absolute_target_difference = abs(df_clean[[self.target_col]] - df_polluted[[self.target_col]])
            return float(1 - mean(absolute_target_difference / mean(df_clean[[self.target_col]])))

    def _add_noise(self, df: DataFrame):
        mean_val = 0
        std = self.pollution_level
        noise_distribution_vector = self.random_generator.normal(mean_val, std, size=len(df.index))
        df_mean = mean(df[self.target_col])

        return df[[self.target_col]].values + reshape(noise_distribution_vector * df_mean, (-1, 1))

    def _swap_categoricals(self, df: DataFrame):
        pollution_idxs = self.random_generator.choice(df.index, size=int(len(df.index) * self.pollution_level),
                                                      replace=False)
        pollution_col_values = df[[self.target_col]].values
        uniques = unique(pollution_col_values)
        for idx in pollution_idxs:
            pollution_col_values[idx] = self.random_generator.choice(
                delete(uniques, where(uniques == pollution_col_values[idx])), size=1)
        return pollution_col_values
