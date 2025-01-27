from abc import ABC
from pandas import concat
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from experiment import Experiment
from util import one_hot_encode
from sklearn.preprocessing import StandardScaler  # , MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import random
import numpy as np
import time


class ClassificationDataset(Dataset):
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
        y = self.y.iloc[idx]
        if self.transform:
            X = self.transform(X)
        return torch.tensor(X), torch.tensor(y)

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KNNClassifier(torch.nn.Module):
    def __init__(self, n_neighbors=5):
        super(KNNClassifier, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, X_train, X_test, y_train):
        # Implement the kNN algorithm
        distances = torch.cdist(X_test, X_train, p=2)
        _, indices = distances.topk(self.n_neighbors, largest=False, dim=-1)
        nearest_labels = y_train[indices]
        predictions, _ = torch.mode(nearest_labels, dim=-1)
        return predictions


class PytorchClassificationExperiment(ABC):
    def __init__(self, name, train_loader, test_loader, model, device):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.device = device

    def run(self):
        self.model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        # Retrieve the entire training dataset for kNN classification
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

        test_result = classification_report(all_labels, all_preds, output_dict=True)
        return {self.name: {'scoring': test_result}}


class PytorchClassificationMLPExperiment(ABC):
    def __init__(self, name, train_loader, test_loader, model, device):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.device = device

    def run(self):
        criterion = torch.nn.CrossEntropyLoss()
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
        epochs_no_improve = 0
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
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} with no improvement for {patience} consecutive epochs.")
                break
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s (Data loading: {data_loading_time:.2f}s per batch)")

        # Evaluation loop
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                _, y_pred = torch.max(outputs, 1)

                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        test_result = classification_report(all_labels, all_preds, output_dict=True)
        return {self.name: {'scoring': test_result}}


class PytorchKNeighborsExperiment(PytorchClassificationExperiment):
    def __init__(self, df_train, df_test, metadata) -> None:
        set_random_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = KNNClassifier(n_neighbors=5)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        train_dataset = ClassificationDataset(df_train, metadata['target'])
        test_dataset = ClassificationDataset(df_test, metadata['target'])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch k-Nearest Neighbors Classification', train_loader, test_loader, model, device)


class PytorchMLPClassifier(torch.nn.Module):
    def __init__(self, random_state, input_dim, hidden_layer_sizes=(100,), activation='relu', output_dim=2):
        set_random_seed(42)
        super(PytorchMLPClassifier, self).__init__()

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

        # Add output layer
        layers.append(torch.nn.Linear(in_features, output_dim))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PytorchMLPExperiment(PytorchClassificationMLPExperiment):
    def __init__(self, df_train, df_test, metadata):
        set_random_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        input_dim = df_train.shape[1] - 1  # Excluding the target column
        hidden_layer_sizes = (100,)
        output_dim = len(df_train[metadata['target']].unique())  # Number of classes

        model = PytorchMLPClassifier(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                              output_dim=output_dim)

        train_dataset = ClassificationDataset(df_train, metadata['target'])
        test_dataset = ClassificationDataset(df_test, metadata['target'])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch Multilayer Perceptron Classification', train_loader, test_loader, model, device)


class PytorchMLP5Experiment(PytorchClassificationMLPExperiment):
    def __init__(self, df_train, df_test, metadata):
        set_random_seed(42)  # Set random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        input_dim = df_train.shape[1] - 1  # Excluding the target column
        hidden_layer_sizes = (100, 100, 100, 100, 100,)  # You can adjust the hidden layers sizes
        output_dim = len(df_train[metadata['target']].unique())  # Number of classes

        model = PytorchMLPClassifier(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                              output_dim=output_dim)

        train_dataset = ClassificationDataset(df_train, metadata['target'])
        test_dataset = ClassificationDataset(df_test, metadata['target'])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch Multilayer Perceptron Classification (5 hidden layers)', train_loader, test_loader, model, device)


class PytorchMLP10Experiment(PytorchClassificationMLPExperiment):
    def __init__(self, df_train, df_test, metadata):
        set_random_seed(42)  # Set random seed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        input_dim = df_train.shape[1] - 1
        hidden_layer_sizes = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100,)
        output_dim = len(df_train[metadata['target']].unique())

        model = PytorchMLPClassifier(input_dim=input_dim, hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                              output_dim=output_dim)

        train_dataset = ClassificationDataset(df_train, metadata['target'])
        test_dataset = ClassificationDataset(df_test, metadata['target'])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch Multilayer Perceptron Classification (10 hidden layers)', train_loader, test_loader, model, device)


class LinearSVM(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearSVM, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        outputs = outputs.squeeze()
        targets = 2 * targets - 1  # Converts 0, 1 to -1, 1
        hinge_loss = torch.mean(torch.clamp(1 - outputs * targets, min=0))
        return hinge_loss

class PytorchSupportVectorMachineExperiment(PytorchClassificationExperiment):
    def __init__(self, df_train, df_test, metadata):
        set_random_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        input_dim = df_train.shape[1] - 1
        output_dim = len(df_train[metadata['target']].unique())

        model = LinearSVM(input_dim=input_dim, output_dim=output_dim)

        train_dataset = ClassificationDataset(df_train, metadata['target'])
        test_dataset = ClassificationDataset(df_test, metadata['target'])

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=36, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=36, pin_memory=True, prefetch_factor=2)

        super().__init__('Pytorch Support Vector Machine Classification', train_loader, test_loader, model, device)

    def run(self):
        C = 1.0
        criterion = torch.nn.CrossEntropyLoss()  # Using CrossEntropyLoss instead of HingeLoss
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for epoch in range(5):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                l2_lambda = 1.0 / (2.0 * C)
                l2_reg = sum(param.norm(2) for param in self.model.parameters())
                loss += l2_lambda * l2_reg

                loss.backward()
                optimizer.step()

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                _, predictions = torch.max(outputs, 1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        test_result = classification_report(all_labels, all_preds, output_dict=True)
        return {self.name: {'scoring': test_result}}


class ClassificationExperiment(Experiment, ABC):
    def __init__(self, name, df_train, df_test, model, target, scaler=None):
        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.target = target
        self.scaler = scaler

        df = concat([df_train, df_test])
        super().__init__(name, df_train, df_test, model)

    def run(self):
        X_columns_to_drop = [self.target]
        if "index" in self.df_train.keys():
            X_columns_to_drop.append('index')

        X_train = self.df_train.drop(columns=X_columns_to_drop)
        X_test = self.df_test.drop(columns=X_columns_to_drop)
        y_train = self.df_train[self.target]
        y_test = self.df_test[self.target]

        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        if self.model.__class__.__name__ == 'TabNetClassifier':
            X_train = X_train.values
            y_train = y_train.values

            X_test = X_test.values
            y_test = y_test.values
            self.model.fit(X_train, y_train, patience=3, max_epochs=10)
        else:
            self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        test_result = classification_report(y_test, y_pred, output_dict=True)
        return {self.name: {'scoring': test_result}}


class LogRegExperiment(ClassificationExperiment):
    def __init__(self, df_train, df_test, metadata) -> None:
        """
        This model needs the categorical columns from the metadata to one-hot-encode them.
        Min-Max-Normalisation should be used for scaling the features, if wanted.
        """
        model = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=2000, random_state=42, n_jobs=-1)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])
        
        super().__init__('Logistic Regression Classification', df_train, df_test, model, metadata['target'])


class KNeighborsExperiment(ClassificationExperiment):
    def __init__(self, df_train, df_test, metadata) -> None:
        """
        Min-Max-Normalisation should be used for scaling the features, if wanted.
        """
        model = KNeighborsClassifier(n_jobs=-1)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('k-Nearest Neighbors Classification', df_train, df_test, model, metadata['target'])


class DecisionTreeExperiment(ClassificationExperiment):
    """
    Tree-based algorithms does not need any feature scaling.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = DecisionTreeClassifier(random_state=42)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Decision Tree Classification', df_train, df_test, model, metadata['target'])


class MultilayerPerceptronExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to one-hot-encode them.
    Min-Max-Normalisation should be used for scaling the features, if wanted.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = MLPClassifier(random_state=42, max_iter=1000)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Multilayer Perceptron Classification', df_train, df_test, model, metadata['target'])


class MultilayerPerceptron5Experiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to one-hot-encode them.
    Min-Max-Normalisation should be used for scaling the features, if wanted.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 100, 100, 100, 100,))

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Multilayer Perceptron Classification (5 hidden layers)', df_train, df_test, model, metadata['target'])


class MultilayerPerceptron10Experiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to one-hot-encode them.
    Min-Max-Normalisation should be used for scaling the features, if wanted.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100,))

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Multilayer Perceptron Classification (10 hidden layers)', df_train, df_test, model, metadata['target'])


class SupportVectorMachineExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to one-hot-encode them.
    z-score standardisation should be used for scaling the features, if wanted.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = SVC(random_state=42, kernel='linear')

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Support Vector Machine Classification', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class GradientBoostingClassifierExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to one-hot-encode them.
    Min-Max-Normalisation should be used for scaling the features, if wanted.
    """
    def __init__(self, df_train, df_test, metadata) -> None:
        model = GradientBoostingClassifier(random_state=42)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('Gradient Boosting Classification', df_train, df_test, model, metadata['target'])


class TabNetExperiment(ClassificationExperiment):
    def __init__(self, df_train, df_test, metadata) -> None:
        model = TabNetClassifier(seed=42)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('TabNet Classification', df_train, df_test, model, metadata['target'])

class TabNetMultiTaskExperiment(ClassificationExperiment):
    def __init__(self, df_train, df_test, metadata) -> None:
        model = TabNetMultiTaskClassifier(seed=42)

        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'])

        super().__init__('TabNet Classification', df_train, df_test, model, metadata['target'])
