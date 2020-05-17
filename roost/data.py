import os
import sys
import argparse
import functools

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from roost.features import LoadFeaturiser
from roost.parse import parse


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description=("Roost - a Structure Agnostic Message Passing "
            "Neural Network for Inorganic Materials")
    )

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/expt-non-metals.csv",
        metavar="PATH",
        help="dataset path",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="validation set path"
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="N",
        help="proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path",
        type=str,
        metavar="PATH",
        help="testing set path"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="N",
        help="proportion of data for testing",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/matscholar-embedding.json",
        metavar="PATH",
        help="atom feature path",
    )

    # dataloader inputs
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        metavar="N",
        help="seed for random number generator",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="N",
        help="sub-sample the training set for learning curves",
    )

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--classification",
        action="store_true",
        help="specifies a classification task"
    )
    task_group.add_argument(
        "--regression",
        action="store_true",
        help="specifies a regression task"
    )

    # optimiser inputs
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="str",
        help="Loss Function (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use hetroskedastic loss variant"
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="str",
        help="choose an optimizer; SGD, Adam or AdamW (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="float",
        help="initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="float [0,1]",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="float [0,1]",
        help="weight decay (default: 1e-6)",
    )

    # graph inputs
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="N",
        help="number of hidden atom features in conv layers (default: 64)",
    )
    parser.add_argument(
        "--n-graph",
        default=3,
        type=int,
        metavar="N",
        help="number of graph layers (default: 3)",
    )

    # ensemble inputs
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="name for model"
    )
    parser.add_argument(
        "--data-id",
        default="roost",
        type=str,
        metavar="N",
        help="identifier for the data/cross-val fold",
    )
    parser.add_argument(
        "--run-id",
        default=0,
        type=int,
        metavar="N",
        help="ensemble model id"
    )
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="N",
        help="number ensemble repeats"
    )

    # restart inputs
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--resume",
        action="store_true",
        help="resume from previous checkpoint"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="checkpoint path for transfer learning",
    )
    use_group.add_argument(
        "--fine-tune",
        type=str,
        metavar="PATH",
        help="checkpoint path for fine tuning"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="skip network training stages checkpoint",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="skip network training stages checkpoint"
    )

    # misc
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="log metrics to tensorboard"
    )

    args = parser.parse_args(sys.argv[1:])

    if args.model_name is None:
        args.model_name = f"{args.data_id}_s-{args.seed}_t-{args.sample}"

    if args.regression:
        args.task = "regression"
    elif args.classification:
        args.task = "classification"
    else:
        args.task = "regression"

    args.device = (
        torch.device("cuda")
        if (not args.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(self, data_path, fea_path, task, df=None):
        """
        """
        if df:
            self.df = df
        else:
            assert os.path.exists(data_path), "{} does not exist!".format(data_path)
            # make sure to use dense datasets, here do not use the default na
            # as they can clash with "NaN" which is a valid material
            self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.elem_features = LoadFeaturiser(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size()
        self.task = task
        if self.task == "regression":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "Multi-target regression currently not supported"
                )
            self.n_targets = self.df.shape[1] - 2
        elif self.task == "classification":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "One-Hot input not supported please use categorical integer"
                    " inputs for classification i.e. Dog = 0, Cat = 1, Mouse = 2"
                )
            self.n_targets = np.max(self.df[self.df.columns[2]].values) + 1

    def __len__(self):
        return len(self.df)

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
        # cry_id, composition, target = self.id_prop_data[idx]
        cry_id, composition, *targets = self.df.iloc[idx]
        elements, weights = parse(composition)
        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elements) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        if self.task == "regression":
            targets = torch.Tensor([float(t) for t in targets])
        elif self.task == "classification":
            targets = torch.LongTensor([targets[0]])

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            composition,
            cry_id,
        )


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
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for (i, ((atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            target, comp, cry_id)) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
        ),
        torch.stack(batch_target, dim=0),
        batch_comp,
        batch_cry_ids,
    )
