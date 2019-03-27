import os
import json
import numpy as np
import pandas as pd

class NewEmbedding(object):

    def __init__(self, data_dir):
        assert os.path.exists(data_dir), '{} does not exist!'.format(data_dir)
        self.data_dir = data_dir

        self._embedding = {}

    def save_embedding(self, save_dir, save_file):
        """
        Save the embedding
        """
        assert os.path.exists(save_dir), 'save_dir does not exist!'
        save_path = os.path.join(save_dir, save_file)
        with open(save_path, 'w+') as f:
            json.dump(self._embedding, f)

    def get_embedding(self):
        """
        Return the embedding
        """
        return self._embedding

class AtomEmbedding(NewEmbedding):
    '''
    construct a dictionary of features using the atom as a key
    '''

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def construct_embedding(self, features):
        """
        take the given features and construct a feature embedding
        the inputs for the features should be in their own folders
        with a meta.csv of infomation about the feature and a data.csv
        consisting of the key (i.e. element) and value for a given 
        feature. 

        Parameters
        ----------
        features: list of length F
            list of features to use in embedding
        """

        series_list = []

        for feature in features:
            feature_file = os.path.join(self.data_dir, feature, feature+'.csv')
            meta_file = os.path.join(self.data_dir, feature, 'meta.json')
            assert os.path.exists(feature_file), '{} does not exist!'.format(feature_file)
            assert os.path.exists(meta_file), '{} does not exist!'.format(meta_file)

            # load the feature data and its meta data
            series = pd.read_csv(feature_file, header=None, index_col=0, squeeze=True)
            with open(meta_file) as f:
                meta_data = json.load(f)

            if meta_data.get('expand') != None:
                data = series.values
                
                # ensure that we have valid inputs for expansion
                if meta_data.get('expand') == 'categorical':
                    meta_data.update({'expand': 'onehot'})
                    meta_data.update({'steps': np.max(data).astype(int)})
                if meta_data.get('log-scale') == True:
                    meta_data.update({'expand': 'onehot'})
                    data = np.log(data)
                if meta_data.get('steps') == None:
                    print('{}: number of steps not given for feature expansion, using default value of 5'.format(feature))
                    meta_data.update({'steps': 5})
                if meta_data.get('fmin') == None:
                    print('{}: fmin not specified, using minimum as default'.format(feature))
                    meta_data.update({'fmin': np.min(data)})
                if meta_data.get('fmax') == None:
                    print('{}: fmax not specified, using maximum as default'.format(feature))
                    meta_data.update({'fmax': np.max(data)})

                # carry out expansion
                if meta_data.get('expand') == 'onehot':
                    onehot = OneHotEmbedding(meta_data.get('fmin'), meta_data.get('fmax'),
                                                meta_data.get('steps'))
                    expanded_values = onehot.expand(data)
                elif meta_data.get('expand') == 'gaussian':
                    gaussian = GaussianEmbedding(meta_data.get('fmin'), meta_data.get('fmax'),
                                                 meta_data.get('steps'), meta_data.get('var'))
                    expanded_values = gaussian.expand(data)
                    expanded_values[expanded_values < 1e-3] = 0
                else:
                    raise NameError('Only \'onehot\' or \'gaussian\' feature expansions are implemented')
                
                for i in range(expanded_values.shape[1]):
                    series_list.append(pd.Series(expanded_values[:,i], index=series.index))
            else:
                series_list.append(series)

        # combine the series into a dataframe and convert to a dictionary
        df = pd.concat(series_list, axis=1)
        keys=df.index.values
        self._embedding = dict(zip(keys, df.loc[keys].values.tolist()))


class BondEmbedding(NewEmbedding):
    '''
    construct a dictionary of features using the concatenation of the 
    two relevant atoms as a key
    '''

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def construct_embedding(self):
        """
        Parameters
        ----------
        """
        feature_file = os.path.join(self.data_dir, 'Electronegativity.csv')
        assert os.path.exists(feature_file), '{} does not exist!'.format(feature_file)
        series = pd.read_csv(feature_file, header=None, index_col=0, squeeze=True).to_dict()

        for key_a, value_a in series.items():
            for key_b, value_b in series.items():
                key_ab = key_a+key_b
                # self._embedding[key_ab] = [value_a-value_b, (value_a+value_b)/2.]
                self._embedding[key_ab] = []


class GaussianEmbedding(object):
    """
    Expands a feature in a Gaussian basis.
    """
    def __init__(self, dmin, dmax, steps, var=None):
        """
        Parameters
        ----------
        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        """
        assert dmin < dmax
        self.filter = np.linspace(dmin, dmax, steps)
        if var is None:
            var = (dmax-dmin)/(2*steps)
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        TODO maybe we should normalise the rows?

        Parameters
        ----------
        distances: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class OneHotEmbedding(object):
    """
    Expands a feature using a one hot encoding.
    """
    def __init__(self, dmin, dmax, steps):
        """
        Parameters
        ----------
        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        steps: float
            number of steps for the filter
        """
        assert dmin < dmax
        self.filter = np.linspace(dmin, dmax, steps)
        self.step = self.filter[1]-self.filter[0]

    def expand(self, distances):
        """
        Apply one hot distance filter to a numpy distance array

        Parameters
        ----------
        distances: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        expand = distances[..., np.newaxis]-self.filter
        return np.where(np.logical_and(expand<self.step/2, expand>=-self.step/2), 1, 0)


def ArkelKetelaarType(A, B):
    """
    Use the idea of van Arkel-Ketelaar triangle to define the 
    expected type of bond between nodes A and B.

    TODO implement bond categorisation based on the 
        electronegativities of the two elements

    Parameters
    ----------
    A: str
        Key for the atom
    B: str
        Key for the neighbour
    """

    pass

