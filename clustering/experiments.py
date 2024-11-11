""" All experiments run for clustering are defined here as classes, all creating their respective model upon
initialization and implementing inference in their ``run()`` method.
"""

from abc import ABC
from copy import deepcopy
from gower import gower_matrix
from tqdm import tqdm

import logging
import numpy as np
import pandas as pd
import random

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.metrics import rand_score, mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from experiment import Experiment
from util import one_hot_encode


class ClusteringBaseExperiment(Experiment, ABC):
    """ Parent class for all clustering experiments. Implements the initialization interface to the generic Experiment
    class and performs one-hot encoding of categorical columns.

    Takes only one dataframe as we are unsupervised - will be saved as test data for later use in base
    Experiment object.
    """
    def __init__(self, name, df, model, metadata, use_gower=False):
        """
        :param name: name of the experiment, usually just model name
        :param df: dataset which the clustering will run on, saved as test data
        :param model: model instance which will be used to run inference
        :param metadata: metadata dictionary corresponding to the given dataset
        :param use_gower: whether the gower distance matrix should be used at inference time instead of the raw dataset
        """
        df_exp = df.copy(deep=True)

        # encode categorical columns
        _, df_exp = one_hot_encode(pd.DataFrame(columns=df_exp.columns), df_exp, metadata['categorical_cols'])

        # set and encode target column
        self._target_col = metadata['target']
        df_exp[self._target_col] = LabelEncoder().fit_transform(df_exp[self._target_col])

        self._use_gower = use_gower
        self._cat_feats = [c for c in df_exp.columns if
                           (c not in metadata['numerical_cols']) and (c != self._target_col)]

        # train data passed is None since we have none
        Experiment.__init__(self, name, None, df_exp, model)


class ClusteringExperiment(ClusteringBaseExperiment):
    """ Parent class for those clustering experiments that use models adhering to the sklearn interface and not needing
    any special data treatment in the ``run()`` function.
    Implements this generic ``run()`` function for its children.
    """
    def run(self, verbose=True) -> dict:
        if verbose:
            logging.info(f'Running: {self.name} experiment ...')
        # split into samples and labels
        X, y = self.test[[c for c in self.test.columns if c != self._target_col]], self.test[self._target_col]

        try:
            if self._use_gower:
                dist_mat = gower_matrix(X.to_numpy(dtype=float), cat_features=[c in self._cat_feats for c in X.columns])
                y_pred = self.model.fit_predict(dist_mat)
            elif self.name == 'k-Prototypes':
                y_pred = self.model.fit_predict(X, categorical=[X.columns.get_loc(c) for c in self._cat_feats])
            else:
                y_pred = self.model.fit_predict(X)
            scores = {
                'rand': {
                    'rand score': rand_score(y, y_pred),
                    'adjusted rand score': adjusted_rand_score(y, y_pred)
                },
                'mutual information': {
                    'mutual information score': mutual_info_score(y, y_pred),
                    'adj_mut_info': adjusted_mutual_info_score(y, y_pred),
                    'norm_mut_info': normalized_mutual_info_score(y, y_pred),
                },
                'n_cluster': len(np.unique(y_pred))
            }
        except ValueError:
            scores = {
                'rand': {
                    'rand score': -1.0,
                    'adjusted rand score': -1.0
                },
                'mutual information': {
                    'mutual information score': -1.0,
                    'adj_mut_info': -1.0,
                    'norm_mut_info': -1.0,
                },
                'n_cluster': -1.0
            }

        return scores


class KMeansExperiment(ClusteringExperiment):
    """ Experiment implementing either k-Means or k-Prototypes algorithm, depending on presence of non-constant
    categorical features.
    """
    def __init__(self, train_df, test_df, metadata) -> None:
        """
        :param train_df: ignored, only there for interface consistency
        :param test_df: dataset clustering will be applied to
        :metadata: metadata dictionary corresponding to the dataset to cluster on
        """
        # use k-Prototype if at least one categorical feature is not constant and thus left after one-hot-encoding
        if (test_df[metadata['categorical_cols']].nunique() > 1).any():
            model = KPrototypes(n_clusters=test_df[metadata['target']].nunique(), n_jobs=-1, random_state=42)
            ClusteringBaseExperiment.__init__(self, 'k-Prototypes', test_df, model, metadata)
        else:
            model = KMeans(n_clusters=test_df[metadata['target']].nunique(), random_state=42)
            ClusteringBaseExperiment.__init__(self, 'k-Means', test_df, model, metadata)


class GaussianMixtureExperiment(ClusteringExperiment):
    """ Implements Gaussian Mixture Model based clustering.
    """
    def __init__(self, train_df, test_df, metadata) -> None:
        """
        :param train_df: ignored, only there for interface consistency
        :param test_df: dataset clustering will be applied to
        :metadata: metadata dictionary corresponding to the dataset to cluster on
        """
        model = GaussianMixture(n_components=test_df[metadata['target']].nunique(), random_state=42, n_init=10)
        ClusteringBaseExperiment.__init__(self, 'Gaussian Mixture', test_df, model, metadata)



class AgglomerativeExperiment(ClusteringExperiment):
    """ Implements Agglomerative Clustering, either based on the raw data or the gower distance matrix if non-constant
    categorical columns are present in the data.
    """
    def __init__(self, train_df, test_df, metadata) -> None:
        """
        :param train_df: ignored, only there for interface consistency
        :param test_df: dataset clustering will be applied to
        :metadata: metadata dictionary corresponding to the dataset to cluster on
        """
        # use gower if at least one categorical feature is not constant and thus left after one-hot-encoding
        use_gower = (test_df[metadata['categorical_cols']].nunique() > 1).any()

        if use_gower:
            model = AgglomerativeClustering(
                n_clusters=test_df[metadata['target']].nunique(),
                affinity='precomputed',
                linkage='average'
            )
        else:
            model = AgglomerativeClustering(n_clusters=test_df[metadata['target']].nunique())
        ClusteringBaseExperiment.__init__(self, 'Agglomerative', test_df, model, metadata, use_gower=use_gower)


class OPTICSExperiment(ClusteringExperiment):
    """ Implements the density-based OPTICS clustering algorithm.
    """
    def __init__(self, train_df, test_df, metadata) -> None:
        """
        :param train_df: ignored, only there for interface consistency
        :param test_df: dataset clustering will be applied to
        :metadata: metadata dictionary corresponding to the dataset to cluster on
        """
        model = OPTICS(min_cluster_size=100, n_jobs=10)
        ClusteringBaseExperiment.__init__(self, 'OPTICS', test_df, model, metadata)





class AutoEncoder(nn.Module):
    """ Autoencoder neural network used in the Autoencoder Clustering algorithm.
    The encoder and decoder are dynamically generated, with each Linear layer halving/doubling the number of features
    and the embedded space always having 2 dimensions.
    """
    def __init__(self, **kwargs):
        super().__init__()

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        in_feats = kwargs['input_shape']

        self.encoder_hidden = list()

        layer_in_out = list()
        layer_in_feats = in_feats
        layer_out_feats = layer_in_feats // 2
        # define input and hidden layers for encoder
        while layer_out_feats > 2:
            self.encoder_hidden.append(
                nn.Linear(
                    in_features=layer_in_feats, out_features=layer_out_feats
                )
            )
            layer_in_out.append((layer_in_feats, layer_out_feats))
            layer_in_feats = layer_out_feats
            layer_out_feats = layer_in_feats // 2

        self.encoder_hidden = nn.ModuleList(self.encoder_hidden)

        # define last layer of encoder
        self.encoder_output = nn.Linear(
            in_features=layer_in_feats, out_features=2
        )

        # define first layer of decoder
        self.decoder_hidden = [
            nn.Linear(
                in_features=2, out_features=layer_in_feats
            )
        ] if len(layer_in_out) > 0 else []

        # define hidden layers of decoder
        for layer_out_feats, layer_in_feats in list(reversed(layer_in_out))[:-1]:
            self.decoder_hidden.append(
                nn.Linear(
                    in_features=layer_in_feats, out_features=layer_out_feats
                )
            )

        self.decoder_hidden = nn.ModuleList(self.decoder_hidden)

        # define output layer of decoder
        self.decoder_output = nn.Linear(
            in_features=list(reversed(layer_in_out))[-1][1] if len(layer_in_out) > 0 else 2,
            out_features=kwargs['input_shape']
        )

    def forward(self, features):
        """ Network feed-forward.
        Returns both encoded and reconstructed data.
        """
        activation = features
        for layer in self.encoder_hidden:
            activation = layer(activation)
            activation = torch.relu(activation)

        encoded = self.encoder_output(activation)
        encoded = torch.relu(encoded)

        activation = encoded

        for layer in self.decoder_hidden:
            activation = layer(activation)
            activation = torch.relu(activation)

        activation = self.decoder_output(activation)
        reconstructed = torch.relu(activation)
        return encoded, reconstructed



class CustomDataset(Dataset):
    """ Dataset wrapper to be able to use PyTorch DataLoader for the batching of data during network training.
    Requires the index of the given pd.DataFrame to contain only unique values, otherwise the __getitem__() function
    can break.
    """
    def __init__(self, df: pd.DataFrame, target_column):
        self.target_col = target_column
        self.X = df[[c for c in df.columns if c != self.target_col]]
        self.y = df[self.target_col]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].astype(float).values, int(self.y[idx])]


class AutoencoderExperiment(ClusteringBaseExperiment):
    """ Implements creation and training of Autoencoder and uses Gaussian Mixture Clustering in order to extract
    clusters from the embedded space.
    Epochs (200), optimizer (Adam) and loss (MSE) are hardcoded in the ``run()`` function.
    """
    def __init__(self, train_df, test_df, metadata) -> None:
        """
        :param train_df: ignored, only there for interface consistency
        :param test_df: dataset clustering will be applied to
        :metadata: metadata dictionary corresponding to the dataset to cluster on
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata = metadata
        onehot_encoded_col_count = len(metadata['numerical_cols']) \
                                   + sum([test_df[c].nunique() for c in metadata['categorical_cols']]) \
                                   - len(metadata['categorical_cols'])
        model = AutoEncoder(input_shape=onehot_encoded_col_count).to(self.device)
        super().__init__('Autoencoder', test_df, model, metadata)

    def run(self) -> dict:
        logging.info(f'Running: {self.name} experiment ...')

        # wrap the dataframe to run clustering on into the custom dataset class to facilitate DataLoader usage
        dataset = CustomDataset(self.test, self._target_col)
        # perform train/test split at 80-20
        trainsize = int(0.8 * len(dataset))
        trainset, testset = random_split(dataset, [trainsize, len(dataset) - trainsize])

        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = DataLoader(testset, batch_size=128, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        criterion = nn.MSELoss()

        # training
        loss_per_iter = []
        loss_per_batch = []

        running_loss = 0.0
        i = 0
        pbar = tqdm(range(200))

        self.model.train()

        for epoch in pbar:
            pbar.set_description(f'Epoch {epoch:03d}: Loss: {np.sqrt(running_loss / (i + 1)):05f}')
            running_loss = 0.0

            for i, (inputs, _) in enumerate(trainloader):
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                _, outputs = self.model(inputs.float())
                loss = criterion(outputs, inputs.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_per_iter.append(loss.item())

            loss_per_batch.append(running_loss / (i + 1))

        # evaluation on test data
        self.model.eval()
        with torch.no_grad():
            testdataiter = iter(testloader)
            #inputs, labels = testdataiter.next()
            inputs, labels = next(testdataiter)
            inputs = inputs.to(self.device)
            _, outputs = self.model(inputs.float())
            logging.info(f'RMSE Training: {np.sqrt(loss_per_batch[-1])}, '
                         f'Test: {np.sqrt(criterion(outputs, inputs.float()).detach().cpu().numpy())}')

        # get embeddings
        fullloader = DataLoader(dataset, batch_size=128, shuffle=False)
        encoded_outputs = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(fullloader):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                encoded, _ = self.model(inputs.float())

                encoded_outputs.append(encoded)
                encoded_result = torch.cat(encoded_outputs, dim=0).cpu().numpy()

        encoded_data = pd.DataFrame(encoded_result)
        # attach original cluster labels to the encoded data
        encoded_data[self._target_col] = self.test[self._target_col]

        # need to pass changed dataset metadata, as the columns have changed due to encoding
        adapted_metadata = deepcopy(self.metadata)
        adapted_metadata['categorical_cols'] = []
        adapted_metadata['numerical_cols'] = encoded_data.columns.tolist()

        # there is no train data, pass None instead
        clustering_exp = GaussianMixtureExperiment(None, encoded_data, adapted_metadata)
        return clustering_exp.run(verbose=False)
