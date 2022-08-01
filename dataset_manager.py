from hashlib import md5
from logging import debug
from pandas import DataFrame, read_csv
from pathlib import Path
from typing import Tuple, Union
from polluters import Polluter


class DatasetManager:
    @staticmethod
    def polluted_file(polluted_dir: Path, ds_name: str, polluter: Polluter) -> Path:
        """
        Generates and returns the file path for the given dataset name and polluter applied to it.
        The dataset directory is %base polluted dir%/%polluter class%/%random seed%/. The file name is composed of the
        original dataset's filename and an MD5 hash of the pollution parameters used in the pollution of the dataset.

        :param polluted_dir: directory of the polluted dataset
        :type polluted_dir: pathlib.Path
        :param ds_name: name of the base (raw) dataset the polluted dataset is based on
        :type ds_name: str
        :param polluter: Polluter instance that was applied to the raw data to obtain the given polluted dataset
        :type polluter: Polluter
        :returns: filepath to the polluted dataset
        :rtype: pathlib.Path
        """

        # subdirectory in the polluted data directory to save to, consists from polluter class name and used random seed
        specific_subdir = polluted_dir / f'{polluter.__class__.__name__}/{polluter.random_seed}/'
        # polluted file name, consists of the original dataset name and a hash of the pollution parameters
        parameter_hash = md5(polluter.get_pollution_params().encode('utf-8')).hexdigest()
        polluted_filename = f'{ds_name.split(".")[0]}_{parameter_hash}.csv'

        return specific_subdir / polluted_filename

    @staticmethod
    def persist_dataset(polluted_dir: Path, dataset: DataFrame, ds_name: str, polluter: Polluter):
        """
        Writes the given dataset to disk as CSV file.
        The dataset is written to the directory %polluted dir%/%polluter class%/%random seed%/ and includes an MD5 hash
        of the pollution parameters used in the filename.

        :param polluted_dir: directory of the polluted dataset
        :type polluted_dir: pathlib.Path
        :param dataset: (polluted) dataset to persist
        :type dataset: pd.DataFrame
        :param ds_name: name of the base (raw) dataset the polluted dataset is based on
        :type ds_name: str
        :param polluter: Polluter instance that was applied to the raw data to obtain the given polluted dataset
        :type polluter: Polluter
        """

        polluted_file = DatasetManager.polluted_file(polluted_dir, ds_name, polluter)

        if polluted_file.is_file():
            # this specific dataset and parameter combination was already persisted, no need to do that again
            debug(f'Polluted version of {ds_name} with parameter has was already persisted at {polluted_file}.')
            return

        debug(f'Persisting dataset {ds_name} with pollution parameters {polluter.get_pollution_params()} at '
                      f'{polluted_file}')

        if not polluted_file.parent.exists():
            polluted_file.parent.mkdir(parents=True, exist_ok=True)

        dataset.to_csv(polluted_file, index=False)

    @staticmethod
    def read_or_pollute(polluted_dir: Path, dataset: DataFrame, ds_name: str, polluter: Polluter) \
            -> Tuple[DataFrame, Union[float, Tuple[float, float]]]:
        """
        Either reads the polluted dataset if it was already persisted (check based on MD5 hash of pollution
        parameters) or applies the pollution.
        Returns both the polluted dataset and quality measure for the particular quality dimension.

        :param polluted_dir: directory of the polluted dataset
        :type polluted_dir: pathlib.Path
        :param dataset: raw dataset to be polluted
        :type dataset: pd.DataFrame
        :param ds_name: name of the base (raw) dataset
        :type ds_name: str
        :param polluter: Polluter instance to be applied to the raw dataset, initialized with the pollution parameters
        :type polluter: Polluter
        :returns: the polluted dataframe and its quality measure
        :rtype: pd.DataFrame, float or tuple
        """

        polluted_file = DatasetManager.polluted_file(polluted_dir, ds_name, polluter)

        if polluted_file.is_file():
            # if polluted dataset was already persisted, read it from disk and calculate quality measure only
            debug(f'Reading dataset {ds_name} with pollution parameters {polluter.get_pollution_params()} from '
                          f'{polluted_file}')
            polluted_df = read_csv(polluted_file)
            return polluted_df, polluter.compute_quality_measure(polluted_df, dataset)
        else:
            # if polluted dataset has not been persisted before, apply Polluter instance
            debug(f'Applying pollution with parameters {polluter.get_pollution_params()} to dataset {ds_name}')
            return polluter(dataset)
