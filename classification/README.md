# Data Quality Analysis of Classification Algorithms

This directory contains all scripts and results to make the classification experiments reproducable. We use these three datasets:
- [Creditworthiness](http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf), referenced as SouthGermanCredit.csv
- [Contraceptive Method Choice](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice), referenced as cmc.data
- [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn), referenced as TelcoCustomerChurn.csv


## Classification Algorithms

We use five different  algorithms to run the experiments. Each algorithm is defined in [experiments.py](experiments.py)
| Category            | Algorithm               | 
|---------------------|-------------------------|
| linear regression   | Logistic regression    |
| linear regression   | support vector machine  |
| tree based          | decision tree           |
| tree based          | k-nearest neighbors         | 
| deep learning       | multi layer perceptron  |

All algorithms are implemented using the scikit-learn (version 1.0.1).

## Execution of Experiments

The script can be started from the top directory by
``` shell script
$ python -m classification.classification
```
Note: If you do not want to see the logging messages but decent progress bars, change start_logging(cmd_out=True) to start_logging(cmd_out=False). The log files can be found in data/logs.


## Result Plotting

Navigate to  `DQ4AI/notebooks/PlottingClassificationResults.ipynb`. The notebook is currently designed to load one json file per polluter containing its results, but if you generated only one results file with the aforementioned console call, you need to read in that instead.

The plotting and preprocessing methods can be found in `classification/plotting.py` and classification/preprocessing.py
