import json
import numpy as np

class Featuriser(object):
    """
    Base class for featurising nodes and edges.
    """
    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, "{} is not an allowed atom type".format(key)
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])

class LoadFeaturiser(Featuriser):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Notes
    ---------
    For the specific composition net application the keys are concatenated 
    strings of the form "NaCl" where the order of concatenation matters.
    This is done because the bond "ClNa" has the opposite dipole to "NaCl" 
    so for a general representation we need to be able to asign different 
    bond features for different directions on the multigraph.

    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super(LoadFeaturiser, self).__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
