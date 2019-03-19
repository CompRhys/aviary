import json

class Featuriser(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomFeaturiser(Featuriser):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {str(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomFeaturiser, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class BondFeaturiser(Featuriser):
    """
    Initialize bond feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------
    bond_embedding_file: str
        The path to the .json file
    """
    def __init__(self, bond_embedding_file):
        with open(bond_embedding_file) as f:
            bond_embedding = json.load(f)
        bond_embedding = {str(key): value for key, value
                          in elem_embedding.items()}
        bond_types = set(elem_embedding.keys())
        super(BondFeaturiser, self).__init__(bond_types)
        for key, value in bond_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)