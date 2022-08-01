from abc import ABC
from pandas import concat
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from experiment import Experiment
from util import one_hot_encode
from sklearn.preprocessing import StandardScaler  # , MinMaxScaler


class ClassificationExperiment(Experiment, ABC):
    def __init__(self, name, df_train, df_test, model, target, scaler=None):
        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.target = target
        self.scaler = scaler

        df = concat([df_train, df_test])
        super().__init__(name, df_train, df_test, model)

    def run(self):
        # Split train and test data into X and y
        X_columns_to_drop = [self.target]
        if "index" in self.df_train.keys():
            X_columns_to_drop.append('index')

        # Extract target and drop unnecessary columns
        X_train = self.df_train.drop(columns=X_columns_to_drop)
        X_test = self.df_test.drop(columns=X_columns_to_drop)
        y_train = self.df_train[self.target]
        y_test = self.df_test[self.target]

        # Scale features if necessary for this ml algorithm
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

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
