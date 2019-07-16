import os
import random
import sys
import argparse
import csv
import torch
import functools

import numpy as np
from torch.utils.data import Dataset

from sampnn.features import LoadFeaturiser
from sampnn.parse import parse


def input_parser():
    '''
    parse input
    '''
    parser = argparse.ArgumentParser(description='Structure Agnostic Message Passing Neural Network')

    # misc inputs
    parser.add_argument('--data_path', type=str, default='/home/rhys/PhD/sampnn/data/id_comp_prop.csv', metavar='PATH',
    help='dataset path')
    parser.add_argument('--fea_path', type=str, default='/home/rhys/PhD/datasets/embeddings/onehot-embedding.json', metavar='PATH',
    help='atom feature path')
    parser.add_argument('--disable-cuda', action='store_true', 
    help='Disable CUDA')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', 
    help='print frequency (default: 10)')
    
    # restart inputs
    parser.add_argument('--resume', default='', type=str, metavar='PATH', 
    help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
    help='manual epoch number (useful on restarts)')
    
    # dataloader inputs
    parser.add_argument('--workers', default=0, type=int, metavar='N', 
    help='number of data loading workers (default: 0)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', 
    help='mini-batch size (default: 256)')    
    parser.add_argument('--train-size', default=0.8, type=float, metavar='N', 
    help='proportion of data for training')
    parser.add_argument('--val-size', default=0.0, type=float, metavar='N', 
    help='proportion of training data used for validation')
    parser.add_argument('--test-size', default=0.2, type=float, metavar='N', 
    help='proportion of data for testing')
    
    # optimiser inputs
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', 
    help='choose an optimizer; SGD or Adam or RMSprop (default: SGD)')
    parser.add_argument('--loss', default='L2', type=str, metavar='L2', 
    help='choose an Loss Function; L2, L1 or Huber (default: L2)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N', 
    help='number of total epochs to run (default: 100)')
    parser.add_argument('--learning-rate', default=0.03, type=float, metavar='LR', 
    help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='W', 
    help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=0, type=float, metavar='W', 
    help='weight decay (default: 0)')
    
    # graph inputs
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', 
    help='number of hidden atom features in conv layers')
    parser.add_argument('--n-graph', default=1, type=int, metavar='N', 
    help='number of graph layers')

    args = parser.parse_args(sys.argv[1:])

    assert args.train_size + args.test_size <= 1
    args.cuda = (not args.disable_cuda) and torch.cuda.is_available()

    return args





class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """
    def __init__(self, data_path, fea_path):
        """
        """
        assert os.path.exists(data_path), '{} does not exist!'.format(data_path)

        with open(data_path) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        assert os.path.exists(fea_path), '{} does not exist!'.format(fea_path)

        self.atom_features = LoadFeaturiser(fea_path)

        self.atom_fea_dim = self.atom_features.embedding_size()

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """

        Returns
        -------
        atom_weights: torch.Tensor shape (M, 1)
            weights of atoms in the material
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the material
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indicies
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbour indicies
        target: torch.Tensor shape (1,)
            target value for material
        cry_id: torch.Tensor shape (1,)
            input id for the material
        """
        cry_id, composition, target = self.id_prop_data[idx]
        elements, weights = parse(composition)
        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elements) != 1, 'crystal {}: {}, is a pure system'.format(cry_id, composition)
        try:
            atom_fea = np.vstack([self.atom_features.get_fea(element) for element in elements])
        except AssertionError:
            print(composition)
            sys.exit()
        atom_fea = np.hstack((atom_fea,weights))
        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        for i, element in enumerate(elements):
            nbrs = elements[:i]+elements[i+1:]
            self_fea_idx += [i]*len(nbrs)
            nbr_fea_idx += env_idx[:i]+env_idx[i+1:]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        # target = torch.Tensor([np.log(float(target))])
        target = torch.Tensor([float(target)])

        return (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx), target, cry_id

def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = [] 
    batch_atom_fea = [] 
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = [] 
    batch_target = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, ((atom_weights, atom_fea, self_fea_idx, nbr_fea_idx), target, cry_id) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]  

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx+cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx+cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i]*n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i

    return (torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx)), \
            torch.stack(batch_target, dim=0), \
            batch_cry_ids


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, ):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor((0))
        self.std = torch.tensor((1))

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
