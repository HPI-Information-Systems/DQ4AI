import pandas as pd
from sklearn import datasets, linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from typing import List

from experiment import Experiment
from util import one_hot_encode


class RegressionExperiment(Experiment):
    """
    Base class for regression experiments. Each child class defines the machine learning algorithm to be used
    """

    def __init__(self, name, train_df: pd.DataFrame, test_df: pd.DataFrame, model, target_col: str,
                 categorical_columns: List[str]) -> None:
        self.target_attribute = target_col
        self.categorical_columns = categorical_columns
        super().__init__(name, train_df.copy(deep=True), test_df.copy(deep=True), model)

    def run(self) -> pd.DataFrame:
        """
        A concrete implementation of the experiment. The resources defined in the constructor can be accessed here.
        :return: a dataframe that contains the results of the experiment
        """
        train_df, test_df = one_hot_encode(self.train, self.test, self.categorical_columns)

        x_train, y_train = train_df.drop(
            columns=self.target_attribute), train_df[self.target_attribute]
        x_test, y_test = test_df.drop(
            columns=self.target_attribute), test_df[self.target_attribute]

        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)

        result = {
            "mean_squared_error": mean_squared_error(y_test, y_pred, squared=False),
            "r2_score": r2_score(y_test, y_pred)
        }

        return result


class LinearRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = linear_model.LinearRegression(n_jobs=-1)
        super().__init__('Linear_Regression', train_df,
                         test_df, model, target_col, categorical_columns)


class RidgeRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = linear_model.Ridge(random_state=12345)
        super().__init__('Ridge_Regression', train_df,
                         test_df, model, target_col, categorical_columns)


class DecisionTreeRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = tree.DecisionTreeRegressor(random_state=12345)
        super().__init__('Decision_Tree_Regression',
                         train_df, test_df, model, target_col, categorical_columns)


class RandomForestRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = RandomForestRegressor(random_state=12345, n_jobs=-1)
        super().__init__('Random_Forest_Regression',
                         train_df, test_df, model, target_col, categorical_columns)


class MLPRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = MLPRegressor(random_state=12345, max_iter=3000)
        super().__init__('MLP_Regression', train_df,
                         test_df, model, target_col, categorical_columns)


class MLPRegression5Experiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = MLPRegressor(random_state=12345, max_iter=3000, hidden_layer_sizes=(100, 100, 100, 100, 100,))
        super().__init__('MLP_Regression_5_hidden_layers', train_df,
                         test_df, model, target_col, categorical_columns)


class MLPRegression10Experiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = MLPRegressor(random_state=12345, max_iter=3000, hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100,))
        super().__init__('MLP_Regression_10_hidden_layers', train_df,
                         test_df, model, target_col, categorical_columns)


class GradientBoostingRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = GradientBoostingRegressor(random_state=12345)
        super().__init__('GradientBoosting_Regression', train_df,
                         test_df, model, target_col, categorical_columns)
