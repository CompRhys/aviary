import os
import re
import random
import sys
import math
import csv
import torch
import functools
import argparse
import numpy as np

# from scipy.special import gamma

from torch.utils.data import Dataset

from sampnn.features import LoadFeaturiser
from sampnn.parse import parse


def input_parser():
    '''
    parse input
    '''
    parser = argparse.ArgumentParser(description='Structure Agnostic Message Passing Neural Network')

    # misc inputs
    parser.add_argument('data_options', metavar='OPTIONS', nargs='+', help='dataset options, started with the path to root dir,then other options')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    
    # restart inputs
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    
    # dataloader inputs
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')    
    parser.add_argument('--train-size', default=0.6, type=float, metavar='N', help='proportion of data for training')
    parser.add_argument('--val-size', default=0.2, type=float, metavar='N', help='proportion of data for validation')
    parser.add_argument('--test-size', default=0.2, type=float, metavar='N', help='proportion of data for testing')
    
    # optimiser inputs
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 30)')
    parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N', help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float, metavar='W', help='weight decay (default: 0)')
    
    # graph inputs
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-list', default=[128,64,32], nargs='+', type=int, metavar='N', help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')

    args = parser.parse_args(sys.argv[1:])

    assert args.train_size + args.val_size + args.test_size <= 1
    args.cuda = (not args.disable_cuda) and torch.cuda.is_available()

    return args





class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """
    def __init__(self, data_dir, random_seed=123):
        assert os.path.exists(data_dir), 'data_dir does not exist!'
        self.data_dir = data_dir

        id_comp_prop_file = os.path.join(self.data_dir, 'id_comp_prop.csv')
        assert os.path.exists(id_comp_prop_file), 'id_comp_prop.csv does not exist!'

        with open(id_comp_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_fea_file = os.path.join(self.data_dir, 'atom_fea.json')
        assert os.path.exists(atom_fea_file), 'atom_fea.json does not exist!'

        bond_fea_file = os.path.join(self.data_dir, 'bond_fea.json')
        assert os.path.exists(atom_fea_file), 'bond_fea.json does not exist!'

        self.atom_features = LoadFeaturiser(atom_fea_file)
        self.bond_features = LoadFeaturiser(bond_fea_file)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        '''
        specify how to include weights into the featurisation

        TODO * include weights in more meaningful way
             * add global features like MEGnet? 
        '''
        cry_id, composition, target = self.id_prop_data[idx]
        elements, weights = parse(composition)
        weights = np.atleast_2d(weights).T
        if len(elements) == 1:
            # bad data point work out how to handle
            pass
        atom_fea = np.vstack([self.atom_features.get_fea(element) for element in elements])
        atom_fea = np.hstack((atom_fea,weights))
        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        bond_fea = []
        for i, element in enumerate(elements):
            nbrs = elements[:i]+elements[i+1:]
            bond_fea.append(torch.Tensor(np.vstack([self.bond_features.get_fea(element+nbr) for nbr in nbrs])))
            self_fea_idx += [i]*len(nbrs)
            nbr_fea_idx += env_idx[:i]+env_idx[i+1:]

        # convert all data to tensors
        atom_fea = torch.Tensor(atom_fea)
        bond_fea = torch.cat(bond_fea, dim=0)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return (atom_fea, bond_fea, self_fea_idx, nbr_fea_idx), target, cry_id


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
    batch_atom_fea = [] 
    batch_bond_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    atom_bond_idx = []
    crystal_atom_idx = [] 
    batch_target = []
    batch_cry_ids = []

    # define counters
    cry_base_idx = 0
    atom_base_idx = 0
    for (atom_fea, bond_fea, self_fea_idx, nbr_fea_idx), target, cry_id in dataset_list:
        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_bond_fea.append(bond_fea)

        batch_self_fea_idx.append(self_fea_idx+cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx+cry_base_idx)

        # TODO: use better data structure than list of tensors
        # mapping from bonds to atoms
        for _ in range(n_i):
            atom_bond_idx.append(torch.tensor([atom_base_idx,atom_base_idx+n_i-1]))
            atom_base_idx += n_i-1

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([cry_base_idx,cry_base_idx+n_i]))
        cry_base_idx += n_i

        batch_target.append(target)
        batch_cry_ids.append(cry_id)

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_bond_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.stack(atom_bond_idx, dim=0),
            torch.stack(crystal_atom_idx, dim=0)), \
            torch.stack(batch_target, dim=0), \
            batch_cry_ids


# def split_dataset(dataset, points, train_size, val_size, test_size, params):
    # """ take only a subset of the data """
    # indices = list(range(points))
    # train_idx = int(points * train_size) # note int() truncates but this same as floor for +ve 
    # val_idx = int(points * val_size)
    # test_idx = int(points * val_size)
    # train_set, val_set, test_set = indices[:train_idx], indices[train_idx:train_idx+val_idx], indices[-test_idx:]

    # train_sampler = SubsetRandomSampler(train_set)
    # train_generator = DataLoader(dataset, sampler=train_sampler, **params)

    # val_sampler = SubsetRandomSampler(val_set)
    # val_generator = DataLoader(dataset, sampler=val_sampler, **params)

    # test_sampler = SubsetRandomSampler(test_set)
    # test_generator = DataLoader(dataset, sampler=test_sampler, **params)


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
    def __init__(self, tensor, dim=0, keepdim=False):
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