import numpy as np
import warnings
import pandas as pd
from scipy.special import gamma

class AtomEmbedding(object):
    '''
    construct a dictionary of features using the atom as a key.
    save the dictionary as a .JSON file. 
    '''

    def __init__(self,):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        self.data_dir = data_dir

    def construct_embedding(features):
        """
        take the given features and construct a feature embedding

        Parameters
        ----------
        features: list of length F
            list of features to use in embedding
        expand: str
            feature expansion to use on non-categorical features
        """

        for feature in features:
            feature_file = os.path.join(self.data_dir, feature+'/data.csv')
            meta_file = os.path.join(self.data_dir, feature+'/meta.csv')
            assert os.path.exists(feature_file), '{} does not exist!'.format(feature_file)
            assert os.path.exists(meta_file), '{} does not exist!'.format(meta_file)

            meta_data = pd.read_csv(meta_file, header=None, index_col=0, squeeze=True).to_dict()
            df = pd.read_csv(feature_file, delimiter=' ', index_col=0)

            if (meta_data.get('categorical') == False) & (meta_data.get('expand') != None):
                if meta_data.get('steps') == None:
                    print('number of steps not given for feature expansion, using default value of 5')
                    meta_data.update({'steps': 5})
                data = df.values

                if meta_data.get('fmin') != None:
                    meta_data.update({'fmin': np.min(data)})

                if meta_data.get('fmax') != None:
                    meta_data.update({'fmax': np.max(data)})

                if meta_data.get('expand') == 'onehot':
                    onehot = OneHotEmbedding(meta_data.get('fmin'), meta_data.get('fmax'),
                                                meta_data.get('steps'))
                    expanded_df = onehot(data)
                elif meta_data.get('expand') == 'gaussian':
                    gaussian = GaussianEmbedding(meta_data.get('fmin'), meta_data.get('fmax'),
                                                 meta_data.get('steps'), meta_data.get('var'))
                    expanded_df = gaussian(data)
                else:
                    raise NameError('Only \'onehot\' or \'gaussian\' feature expansions are implemented')



    def save():


class BondEmbedding(object):
    '''
    Use the idea of van Arkel-Ketelaar triangle and then 
    gamma distribution to define a number of electrons
    and the expected type of bond between two nodes.
    '''

    def __init__(self,):


    def evaluate():
        '''
        evaluate the weighted bond embedding idea
        '''
        eff_prob = gamma(5)/gamma(6)
        return eff_prob


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

        TODO normalise the rows

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

