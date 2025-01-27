# Data Quality Analysis of Regression Algorithms

This directory contains all scripts and results to make the regression experiments reproducable. We use these three datasets:
- [Cars](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes/version/3), referenced as `vw_prepared.csv`
- [Houses](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), referenced as `house_prices_prepared.csv`
- [IMDB](https://www.kaggle.com/mazenramadan/imdb-most-popular-films-and-series/version/3), referenced as `imdb_prepared.cs`
- [COVID](https://www.kaggle.com/datasets/meirnizri/covid19-dataset), referenced as `covid_data_pre_processed_regression.csv`

## Data Preprocessing

Before running the experiments, we performed some preprocessing steps, which are described in the Jupyter Notebook linked below:
[RegressionDataPreparation.ipynb](notebooks/RegressionDataPreparation.ipynb)

## Regression Algorithms

We use seven different regression algorithms to run the experiments. Each regression algorithm is defined in [experiments.py](experiments.py)

| Category           | Algorithm              |
|--------------------|------------------------|
| linear regression  | linear regression      |
| linear regression  | ridge regression       |
| tree based         | decision tree          |
| tree based         | random forest          |
| tree based         | gradient boosting      |
| deep learning      | multi layer perceptron |
| deep learning      | TabNet                 |

All algorithms are implemented using the `scikit-learn` (version 1.3.2), `torch` (version 2.1.2) and `pytorch-tabnet` (version 4.1.0).

## Execution of Experiments

- `experiments.py`: Contains all regression experiment classes.
- `run_regression_experiments.py`: Runs all regression experiments, including the pollution.
- `util.py`: Util functions to support running the regression experiments.


Please ensure to set all constants at the top of [run_regression_experiments.py](run_regression_experiments.py).

The script can be started by
``` shell script
$ python run_regression_experiments.py
```

## Result File Preprocessing

After running the experiments, one gets a `results.json` file. This file needs to be processed, before plotting it.

For this, depending on the circumstances there are several fitting scripts available:
- `concat_results.py`: When running several regression experiments, several result.json files can be generated. To make it possible to combine all results into one plot, this script can be used. It takes two files and adds the source file's properties to the target file at the respective positions
- `preprocess_results_json.py`: It contains a cli interface to split pollution metrics into several dict entries for the results file, to make them plottable separately. This is useful, when using several configurations for the uniqueness polluter or the consistent representation polluter

Each function can be run with the CLI parameters defined in the respective file.

## Result Plotting

The metrics stored in the result file can be visualized with the [plots_export.py](plots_export.py) script. The script requires the CLI parameters defined in the file.
 
## Reproducibility

To make the report fully reproducible, we provide the result file used to generate the plots for the report. 
It is named `used_results_revision_1.json`.
