# The Effects of Data Quality on Machine Learning Performance

The code in this repository is the code used in our experimental study to explore the correlation between data quality and ML-models performance. We open source it to support the repeatability of our analysis.

You can find all the results in our technical report from [here](https://arxiv.org/abs/2207.14529). A version of this report is submitted as a paper and is currently under review. 

## Abstract
Modern artificial intelligence AI applications require large quantities of training and test data. 
This need creates critical challenges not only concerning the availability of such data, but also regarding its quality. For example, incomplete, erroneous, or inappropriate training data can lead to unreliable models that produce ultimately poor decisions. Trustworthy AI applications require high-quality training and test data along many quality dimensions, such as accuracy, completeness, and consistency. 

We explore empirically the relationship between six data quality dimensions and the performance of 19 popular machine learning algorithms covering the tasks of classification, regression, and clustering, with the goal of explaining their performance in terms of data quality. Our experiments distinguish three scenarios based on the AI pipeline steps that were fed with polluted data: polluted training data, test data, or both. We conclude the paper with an extensive discussion of our observations.

## Project structure
The folder structure of the project is shown below. The first three top-level directories are corresponding to the ML tasks.

In each of these three directories, there are two files:
- `experiments.py`: It contains all the experiments relevant to this task.
- `main.py`: It is where the experiments per task can be run.
 ```   .
    ├── classification      # Experiments for classification algorithms
    ├── clustering          # Experiments for clustering algorithms
    ├── regression          # Experiments for regression algorithms
    ├── notebooks           # the used notebooks for visualization and data preparation (if necessary)
    ├── polluters           # The implementation of the data polluters
    ├── experiment.py       # The interface that each experiment should follow
    └── main.py             # The single point where we create the stand of the polluted data and run all the experiments.
```

In the polluters directory, there are:
- `interfaces.py`:  It contains a class that defines an abstract base class of a polluter. 
- `util.py`: All helper functions go here.

**IMPORTANT** A detailed documentation can be found in each of the tasks directories. 

## Contributors
- Sedir Mohammed
- Lukas Budach
- Moritz Feuerpfeil
- Nina Ihde
- Andrea Nathansen
- Nele Sina Noack
- Hendrik Patzlaff
- Felix Naumann
- Hazar Harmouch 

## Contact
For questions, please contact hazar.harmouch@hpi.de
