from abc import ABC
import pandas as pd
from sklearn import datasets, linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
from experiment import Experiment
from util import one_hot_encode
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor


class RegressionDataset(Dataset):
    def __init__(self, df, target, transform=None):
        self.df = df
        self.target = target
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values.astype('float32')
        y = self.y.iloc[idx].astype('float32')
        if self.transform:
            X = self.transform(X)
        return torch.tensor(X), torch.tensor(y)

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PytorchRegressionMLPExperiment(ABC):
    def __init__(self, name, train_loader, test_loader, model, device):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.device = device

    def run(self):
        criterion = torch.nn.MSELoss()
        solver = 'adam'
        alpha = 0.0001
        learning_rate_init = 0.01
        max_iter = 10
        best_loss = float('inf')
        patience = 3
        min_delta = 0.01

        if solver == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate_init, weight_decay=alpha,
                                         betas=(0.9, 0.999), eps=1e-08)
        else:
            raise ValueError("Only 'adam' solver is implemented in this setup.")

        self.model.train()
        for epoch in range(max_iter):
            epoch_start_time = time.time()
            average_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in self.train_loader:
                data_loading_start_time = time.time()
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                data_loading_time = time.time() - data_loading_start_time

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                outputs = torch.squeeze(outputs)  # Ensure the output has the shape [batch_size]
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                average_loss += loss.item()
                num_batches += 1

            # Compute the average loss for this epoch
            average_loss /= num_batches
            print(f"Epoch {epoch} average loss: {average_loss:.6f}")
            # Check for improvement
            if best_loss - average_loss > min_delta:
                best_loss = average_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} with no improvement for {patience} consecutive epochs.")
                break
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s (Data loading: {data_loading_time:.2f}s per batch)")

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                outputs = torch.squeeze(outputs)

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        r2 = r2_score(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)

        test_result = {
            'mean_squared_error': mse,
            'r2_score': r2
        }
        return {self.name: {'scoring': test_result}}


class PytorchRegressionExperiment(ABC):
    def __init__(self, name, train_loader, test_loader, model, device):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.device = device

    def run(self):
        pass


class KNNRegressor(torch.nn.Module):
    def __init__(self, n_neighbors=5):
        super(KNNRegressor, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, X_train, X_test, y_train):
        distances = torch.cdist(X_test, X_train, p=2)
        _, indices = distances.topk(self.n_neighbors, largest=False, dim=-1)
        nearest_labels = y_train[indices]
        predictions = nearest_labels.mean(dim=-1)
        return predictions


class PytorchKNeighborsRegressionExperiment(PytorchRegressionExperiment):
    def __init__(self, df_train, df_test, target_col: str, categorical_columns: List[str]) -> None:
        set_random_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = KNNRegressor(n_neighbors=5)
        train_dataset = RegressionDataset(df_train, target_col)
        test_dataset = RegressionDataset(df_test, target_col)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch k-Nearest Neighbors Regression', train_loader, test_loader, model, device)

    def run(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        X_train = []
        y_train = []
        for X_batch, y_batch in self.train_loader:
            X_train.append(X_batch.to(self.device))
            y_train.append(y_batch.to(self.device))
        X_train = torch.cat(X_train)
        y_train = torch.cat(y_train)

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model(X_train, X_batch, y_train)

                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Calculate regression metrics
        r2 = r2_score(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)

        test_result = {
            'R2 Score': r2,
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae
        }
        return {self.name: {'scoring': test_result}}


class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', output_dim=1):
        set_random_seed(42)
        super(MLPRegressor, self).__init__()

        layers = []
        in_features = input_dim

        # Add hidden layers
        for hidden_dim in hidden_layer_sizes:
            layers.append(torch.nn.Linear(in_features, hidden_dim))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'tanh':
                layers.append(torch.nn.Tanh())
            elif activation == 'logistic':
                layers.append(torch.nn.Sigmoid())
            in_features = hidden_dim

        layers.append(torch.nn.Linear(in_features, output_dim))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PytorchMLPRegressionExperiment(PytorchRegressionMLPExperiment):
    def __init__(self, df_train, df_test, target_col: str, categorical_columns: List[str]):
        set_random_seed(42)  # Set random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_train, df_test = one_hot_encode(df_train, df_test, categorical_columns)

        input_dim = df_train.shape[1] - 1
        hidden_layer_sizes = (100,)
        output_dim = 1

        model = MLPRegressor(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                             output_dim=output_dim)

        train_dataset = RegressionDataset(df_train, target_col)
        test_dataset = RegressionDataset(df_test, target_col)

        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('pytorch_MLP_Regression', train_loader, test_loader, model, device)


class PytorchMLPRegression5Experiment(PytorchRegressionMLPExperiment):
    def __init__(self, df_train, df_test, target_col: str, categorical_columns: List[str]):
        set_random_seed(42)  # Set random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_train, df_test = one_hot_encode(df_train, df_test, categorical_columns)

        input_dim = df_train.shape[1] - 1
        hidden_layer_sizes = (100, 100, 100, 100, 100,)
        output_dim = 1

        model = MLPRegressor(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                             output_dim=output_dim)

        train_dataset = RegressionDataset(df_train, target_col)
        test_dataset = RegressionDataset(df_test, target_col)

        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch_MLP_Regression_5_hidden_layers', train_loader, test_loader, model, device)


class PytorchMLPRegression10Experiment(PytorchRegressionMLPExperiment):
    def __init__(self, df_train, df_test, target_col: str, categorical_columns: List[str]):
        set_random_seed(42)  # Set random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df_train, df_test = one_hot_encode(df_train, df_test, categorical_columns)

        input_dim = df_train.shape[1] - 1
        hidden_layer_sizes = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100,)
        output_dim = 1

        model = MLPRegressor(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                             output_dim=output_dim)

        train_dataset = RegressionDataset(df_train, target_col)
        test_dataset = RegressionDataset(df_test, target_col)

        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch_MLP_Regression_10_hidden_layers', train_loader, test_loader, model, device)


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

        if self.model.__class__.__name__ == 'TabNetRegressor':
            x_train = x_train.values
            y_train = y_train.values.reshape(-1, 1)

            x_test = x_test.values
            y_test = y_test.values.reshape(-1, 1)
            self.model.fit(x_train, y_train, patience=3, max_epochs=10)
        else:
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

class TabNetRegressionExperiment(RegressionExperiment):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, categorical_columns: List[str]):
        model = TabNetRegressor(seed=12345)
        super().__init__('TabNet_Regression', train_df,
                         test_df, model, target_col, categorical_columns)
