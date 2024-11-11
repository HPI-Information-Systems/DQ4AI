# Data Quality Analysis of Clustering Algorithms

This directory contains all scripts that are related to the clustering experiments. We have chosen to run our experiments with the following three datasets:
- [Bank](https://archive.ics.uci.edu/ml/datasets/bank+marketing), referenced as `bank.csv`
- [Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype), referenced as `covtype.csv`
- [Letter](https://archive.ics.uci.edu/ml/datasets/letter+recognition), referenced as `letter.arff`
- [COVID](https://www.kaggle.com/datasets/meirnizri/covid19-dataset), referenced as `covid_data_pre_processed_clustering.csv`


## Data Preprocessing

Before executing our experiments, we performed several preprocessing steps such as selecting features and removing classes that are too small, which can be found in the [Clustering.ipynb](../notebooks/Clustering.ipynb) notebook. As a final step of the preprocessing, we reduced the size of the chosen datasets with the [sampling.py](sampling.py) script generating different versions for each given dataset (like `bank_2967137.csv` and `bank_6837295.csv`) based on the seeds defined in the [metadata.json](../metadata.json) file.  

The sampling script needs two arguments, the path to the dataset which should be sampled (`-d | --dataset`) and the number of samples to draw from the dataset (`-s | --samples`). An example call can look like this:

``` shell script
$ python sampling.py -d bank.csv -s 7500
```

## Clustering Algorithms

Depending on dataset properties such as dimensionality or available data types, current research recommends different clustering algorithms. In order to cover a wide range of the large variety of clustering algorithms, we have decided to implement one algorithm from five of the most common categories of clustering algorithms. The categories we chose algorithms from are distribution-based, centroid-based, hierarchical, density-based and deep learning-based clustering algorithms. An overview of the analyzed algorithms can be found in the following table:

| Category            | Algorithm               | 
|---------------------|-------------------------|
| centroid-based      | k-Means / k-Prototypes  |
| deep learning-based | Deep Autoencoder        |
| density-based       | OPTICS                  |
| distribution-based  | Gaussian Mixture        | 
| hierarchical        | Agglomerative Clustering |

All algorithms are implemented using the scikit-learn (version 1.0.1) and PyTorch (version 1.10.2 supporting NVIDIA CUDA 10.2) library. If a dataset only contains of numeric columns, we run k-Means, otherwise we apply k-Prototypes. The respective implementations are located in the [experiments.py](experiments.py) file.

## Execution of Experiments

Before the experiments can be executed with the [run_experiments.py](run_experiments.py) script, it must be ensured that paths, which are set as constants at the beginning of the that script, are set correctly. This refers to the path to the clean data (`CLEAN_DATA_DIR`), the path where the polluted datasets are to be stored (`POLLUTED_DATA_DIR`), and the name of the json file (`RESULTS_JSON_PATH`) that contains the calculated result metrics for each experiment configuration. We decided to use different variations of the *Mutual Information Score* and *Rand Score* in addition to the number of identified clusters as result metrics.

Each experiment configuration describes a dataset which is corrupted with a concrete polluter (see [polluters directory](../polluters)) and passed to one of the five mentioned algorithms. For this reason, the datasets, polluters and algorithms to be used can be adjusted in the `main()` function of the [run_experiments.py](run_experiments.py) script with the variables `datasets`, `pollution_methods` and `experiments`.

Once all these variables have been set, the script can be started with the following call:

``` shell script
$ python run_experiments.py
```

## Result File Preprocessing

Like already mentioned, the calculated result metrics for each experiment configuration will be stored in a json file. This file is called `results.json` by default and needs to be preprocessed with [preprocess_results_json.py](preprocess_results_json.py) if at least one of the following two cases applies:

1. Different timestamps have to be merged. This is necessary, if the [run_experiments.py](run_experiments.py) was executed several times because each run adds a new entry in the result json file. I.e., each top-level key in the result json file describes the start time of the run and contains all calculated metrics for each experiment configuration.

2. The datasets were sampled in the preprocessing like explained in the first section of this README. If so, there are several entries in the results json for each sampled dataset, all of which have a different number at the end of the file name (e.g, `bank_2967137.csv` and `bank_6837295.csv`). These numbers represent the different seeds with which the dataset was sampled. The different entries for each dataset can be aggregated with the [preprocess_results_json.py](preprocess_results_json.py).

The preprocessing of the result json file can be started with the example command below using an argument indicating where the result json file is to be found (`--results`) and an argument defining where the preprocessed json file should be stored (`--output`). Afterwards, the script leads through a small menu that asks which of the two cases just described applies and is to be used.

``` shell script
$ python preprocess_results_json.py --results results.json --output results_preprocessed.json
```

## Result Plotting

The metrics stored in the result file can be visualized with the [export_plots.py](export_plots.py) script. This script takes a dataset name (`--ds-to-consider` argument) or uses all available datasets, and creates a sub folder for each dataset in the base directory for the plots (`--plots-dir` argument). Afterwards, it generates plots for each available pollution method and its metrics. I.e., each plot shows the dataset quality on the x-axis and the averaged metric value on the y-axis having each selected algorithm as a single line. 

In addition, the original (pre-pollution) dataset quality is indicated by a black dotted vertical line. It can be calculated with the [dataset_quality_calculation.py](dataset_quality_calculation.py) script and needs to be manually inserted in the [export_plots.py](export_plots.py) script using the `DATASET_BASE_QUALITY` constant.

If the quality of the original dataset differs from the quality of the baseline dataset used as reference for the pollution impact, the algorithms' performance on the original dataset will be indicated by horizontal dashed lines for all plots showing the Adjusted Mutual Information (AMI) score. The AMI scores of the original datasets have to be manually adjusted with the `ORIGINAL_DATASET_AMI` constant.

After both constants are set, an example call of the plot creation script can look like this:

``` shell script
$ python export_plots.py --results results.json --ds-to-consider bank.csv --plots-dir /plots
```
