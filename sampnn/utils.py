import random
import sys
import math
import json
import csv
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

'''
we need a dataset class
a data loader class
a collate function to take batches of crystal idx and return atoms
'''

def input_parser():
    '''
    parse input
    '''
    parser = argparse.ArgumentParser(description='Structure Agnostic Message Passing Neural Network')
    parser.add_argument('data_options', metavar='OPTIONS', nargs='+', help='dataset options, started with the path to root dir,then other options')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression', help='complete a regression or classification task (default: regression)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N', help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train-size', default=0.6, type=float, metavar='N', help='proportion of data for training')
    parser.add_argument('--val-size', default=0.2, type=float, metavar='N', help='proportion of data for validation')
    parser.add_argument('--test-size', default=0.2, type=float, metavar='N', help='proportion of data for testing')
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=[128], nargs='+', type=int, metavar='N', help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')

    args = parser.parse_args(sys.argv[1:])

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    return args


def get_data_loaders(dataset, batch_size=64, train_size=0.6,
                    val_size=0.2, test_size=0.2,
                    num_workers=1, pin_memory=False):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    """

    assert train_size + val_size + test_size <= 1
    total = len(dataset)
    indices = list(range(total_size))
    train = math.floor(total * train_size)
    val = math.floor(total * val_size)
    test = math.floor(total * test_size)

    train_sampler = SubsetRandomSampler(indices[:train])
    val_sampler = SubsetRandomSampler(indices[train:train+val])
    test_sampler = SubsetRandomSampler(indices[-test:])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_expand, pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_expand, pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_expand, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def collate_expand():
    """
    """

    return

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_fea(self, atom_type, nbr_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomFeaturiser(AtomInitializer):
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
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomFeaturiser, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class BondFeaturiser(AtomInitializer):
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
        bond_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        bond_types = set(elem_embedding.keys())
        super(BondFeaturiser, self).__init__(bond_types)
        for key, value in bond_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """
    def __init__(self, data_dir, random_seed=123):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        self.data_dir = data_dir

        id_comp_prop_file = os.path.join(self.root_dir, 'id_comp_prop.csv')
        assert os.path.exists(id_comp_prop_file), 'id_comp_prop.csv does not exist!'

        with open(id_comp_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_fea_file = os.path.join(self.data_dir, 'atom_init.json')
        assert os.path.exists(atom_fea_file), 'atom_init.json does not exist!'
        self.atom_features = AtomFeaturiser(atom_init_file)
        self.bond_features = BondFeaturiser()

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cry_id, composition, target = self.id_prop_data[idx]
        elements, weights = parse_composition(composition)
        atom_fea = np.vstack([self.atom_features.get_fea(element) for element in elements])

        elements = set(elements)
        self_fea_idx = []
        bond_fea_idx = []
        for i, element in enumerate(elements):
            
            nbrs = list(elements.difference(set([element])))
            bond_fea = np.vstack([self.bond_features.get_fea(element+nbr) for nbr in nbrs])
            self_fea_idx += [i]*len(nbrs)

        atom_fea = torch.Tensor(atom_fea)
        bond_fea = torch.Tensor(nbr_fea)
        self_fea_idx = torch.IntTensor(self_fea_idx)
        nbr_fea_idx = torch.IntTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, bond_fea, self_fea_idx, nbr_fea_idx), target, cry_id

    def parse_composition(composition):
        """
        take an input composition string and return an array of elements
        and an array of stoichometric coefficients.
        example: La2Cu04 -> (La Cu O) and (2 1 4)
        this is done in two stages, first formatting to ensure weights
        are explicate then parsing into sections:
        example: BaCu3 -> Ba1Cu3
        example: Ba1Cu3 -> (Ba Cu) & (1 3)

        """
        regex = r"([A-Z][a-z](?![0-9]))"
        regex2 = r"([A-Z](?![0-9]|[a-z]))"
        subst = r"\g<1>1"
        composition = re.sub(regex, subst, composition.rstrip())
        composition = re.sub(regex2, subst, composition)

        elements = []
        weights = []
        regex3 = r"(\d+\.\d+)|(\d+)"
        parsed = [j for j in re.split(regex3, composition) if j]
        elements += parsed[0::2]
        weights += parsed[1::2]
        return elements, weights    