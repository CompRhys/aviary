import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd

import pickle
import functools

import torch
from torch.utils.data import Dataset

from roost.utils import LoadFeaturiser

from pymatgen.core.structure import Structure


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description=("cgcnn"))

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/taata-cgcnn.csv",
        metavar="PATH",
        help="Path to main data set/training set",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="Path to independent validation set",
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path",
        type=str,
        metavar="PATH",
        help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/cgcnn-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )

    # dataloader inputs
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="INT",
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        metavar="INT",
        help="Seed used when splitting data sets (default: 0)",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="INT",
        help="Sub-sample the training set for learning curves",
    )

    # optimiser inputs
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Specifies whether to use hetroskedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="STR",
        help="Optimizer used for training (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="FLOAT",
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer weight decay (default: 1e-6)",
    )

    # graph inputs
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="INT",
        help="Number of hidden features for elements (default: 64)",
    )
    parser.add_argument(
        "--h-fea-len",
        default=128,
        type=int,
        metavar="INT",
        help="Number of hidden features for output network (default: 128)",
    )
    parser.add_argument(
        "--n-graph",
        default=4,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )
    parser.add_argument(
        "--n-hidden",
        default=1,
        type=int,
        metavar="INT",
        help="Number of layers in output network (default: 1)",
    )

    # ensemble inputs
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="INT",
        help="Number models to ensemble",
    )
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )
    name_group.add_argument(
        "--data-id",
        default="cgcnn",
        type=str,
        metavar="STR",
        help="Partial identifier for sub-directory where models will be stored",
    )
    parser.add_argument(
        "--run-id",
        default=0,
        type=int,
        metavar="INT",
        help="Index for model in an ensemble of models",
    )

    # restart inputs
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--fine-tune",
        type=str,
        metavar="PATH",
        help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="Checkpoint path for transfer learning",
    )
    use_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )

    # task type
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--classification",
        action="store_true",
        help="Specifies a classification task"
    )
    task_group.add_argument(
        "--regression",
        action="store_true",
        help="Specifies a regression task"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model/ensemble",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model/ensemble"
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
        help="Log training metrics to tensorboard"
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


class GraphData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    Parameters
    ----------

    data_path: str
            The path to the dataset
    fea_path: str
            The path to the element embedding
    max_num_nbr: int
            The maximum number of neighbors while constructing the crystal graph
    radius: float
            The cutoff radius for searching neighbors
    dmin: float
            The minimum distance for constructing GaussianDistance
    step: float
            The step size for constructing GaussianDistance

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    self_fea_idx: torch.LongTensor shape (n_i, M)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    comp: str
    cif_id: str or int
    """

    def __init__(
        self,
        data_path,
        fea_path,
        task,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        use_cache=True,
    ):
        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE this naming structure might lead to clashes where the model
        # loads the wrong graph from the cache.
        self.cachedir = os.path.join(os.path.dirname(data_path), "cache/")
        if not os.path.isdir(self.cachedir):
            os.makedirs(self.cachedir)

        # make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])[:3000]

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.ari = LoadFeaturiser(fea_path)
        self.elem_fea_dim = self.ari.embedding_size()

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size

        self.task = task
        self.n_targets = 1

    def __len__(self):
        # return len(self.id_prop_data)
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, comp, target, cell, sites = self.df.iloc[idx]
        cif_id = str(cif_id)

        if os.path.exists(os.path.join(self.cachedir, cif_id + ".pkl")):
            with open(os.path.join(self.cachedir, cif_id + ".pkl"), "rb") as f:
                pkl_data = pickle.load(f)
            atom_fea = pkl_data[0]
            nbr_fea = pkl_data[1]
            self_fea_idx = pkl_data[2]
            nbr_fea_idx = pkl_data[3]

        else:
            cell, elems, coords = parse_cgcnn(cell, sites)
            # NOTE getting primative structure before constructing graph
            # significantly harms the performnace of this model.
            crystal = Structure(
                lattice=cell, species=elems, coords=coords, to_unit_cell=True
            )

            # atom features
            atom_fea = [atom.specie.symbol for atom in crystal]

            # neighbours
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            self_fea_idx, nbr_fea_idx, nbr_fea = [], [], []

            for i, nbr in enumerate(all_nbrs):
                if len(nbr) < self.max_num_nbr:
                    # warnings.warn('{} not find enough neighbors to build graph. '
                    #             'If it happens frequently, consider increase '
                    #             'radius.'.format(cif_id))

                    nbr_fea_idx.extend(list(map(lambda x: x[2], nbr)))
                    nbr_fea.extend(list(map(lambda x: x[1], nbr)))

                else:
                    nbr_fea_idx.extend(
                        list(map(lambda x: x[2], nbr[: self.max_num_nbr]))
                    )
                    nbr_fea.extend(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))

                self_fea_idx.extend([i] * min(len(nbr), self.max_num_nbr))

            nbr_fea = np.array(nbr_fea)

            with open(os.path.join(self.cachedir, cif_id + ".pkl"), "wb") as f:
                pickle.dump((atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), f)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        if self.task == "regression":
            target = torch.Tensor([float(target)])
        elif self.task == "classification":
            target = torch.LongTensor([target])

        return (atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), target, comp, cif_id


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
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
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var ** 2
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
    batch_atom_fea, batch_nbr_fea = [], []
    batch_self_fea_idx, batch_nbr_fea_idx = [], []
    crystal_atom_idx, batch_target = [], []
    batch_comps = []
    batch_cif_ids = []
    base_idx = 0
    for (
        i,
        ((atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), target, comp, cif_id),
    ) in enumerate(dataset_list):

        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_self_fea_idx.append(self_fea_idx + base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)

        crystal_atom_idx.extend([i] * n_i)
        batch_target.append(target)
        batch_comps.append(comp)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.LongTensor(crystal_atom_idx),
        ),
        torch.stack(batch_target, dim=0),
        batch_comps,
        batch_cif_ids,
    )


def parse_cgcnn(cell, sites):
    """
    """
    cell = np.array(ast.literal_eval(cell), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(sites):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords
