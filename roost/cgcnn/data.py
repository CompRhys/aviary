import ast
import functools
import os

import numpy as np
import pandas as pd

from itertools import groupby

from pymatgen.core.structure import Structure

import torch
from torch.utils.data import Dataset

from roost.core import Featurizer


class CrystalGraphData(Dataset):
    def __init__(
        self,
        data_path,
        fea_path,
        task_dict,
        inputs=["lattice", "sites"],
        identifiers=["material_id", "composition"],
        radius=5,
        max_num_nbr=12,
        dmin=0,
        step=0.2,
    ):
        """CrystalGraphData returns neighbourhood graphs

        Args:
            data_path (str): The path to the dataset
            fea_path (str): The path to the element embedding
            task_dict ({target: task}): task dict for multi-task learning
            inputs (list, optional): df columns for lattice and sites. Defaults to ["lattice", "sites"].
            identifiers (list, optional): df columns for distinguishing data points. Defaults to ["material_id", "composition"].
            radius (int, optional): cut-off radius for neighbourhood. Defaults to 5.
            max_num_nbr (int, optional): maximum number of neighbours to consider. Defaults to 12.
            dmin (int, optional): minimum distance in gaussian basis. Defaults to 0.
            step (float, optional): increment size of gaussian basis. Defaults to 0.2.
        """
        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 2, "One input column required are required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers

        assert os.path.exists(data_path), "{} does not exist!".format(data_path)

        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[], comment="#")

        self.df["Structure_obj"] = self.df[inputs].apply(get_structure, axis=1)

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.ari = Featurizer.from_json(fea_path)
        self.elem_fea_dim = self.ari.embedding_size

        self.radius = radius
        self.max_num_nbr = max_num_nbr

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size

        self.n_targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                self.n_targets.append(1)
            elif self.task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self):
        # return len(self.id_prop_data)
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # NOTE sites must be given in fractional co-ordinates
        df_idx = self.df.iloc[idx]
        crystal = df_idx["Structure_obj"]
        cif_id, comp = df_idx[self.identifiers]

        # atom features
        atom_fea = [atom.specie.symbol for atom in crystal]

        # # # neighbours
        # all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        # all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        # self_idx_old, nbr_idx_old, nbr_dist_old = [], [], []

        # for i, nbr in enumerate(all_nbrs):
        #     # NOTE due to using a geometric learning library we do not
        #     # need to set a maximum number of neighbours but do so in
        #     # order to replicate the original code.
        #     if len(nbr) < self.max_num_nbr:
        #         nbr_idx_old.extend(list(map(lambda x: x[2], nbr)))
        #         nbr_dist_old.extend(list(map(lambda x: x[1], nbr)))
        #     else:
        #         nbr_idx_old.extend(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
        #         nbr_dist_old.extend(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        #     self_idx_old.extend([i] * min(len(nbr), self.max_num_nbr))

        # nbr_dist_old = np.array(nbr_dist_old)

        self_idx, nbr_idx, _, nbr_dist = crystal.get_neighbor_list(
            self.radius,
            numerical_tol=1e-8,
        )

        if self.max_num_nbr is not None:
            _self_idx, _nbr_idx, _nbr_dist = [], [], []

            for i, g in groupby(zip(self_idx, nbr_idx, nbr_dist), key=lambda x: x[0]):
                s, n, d = zip(*sorted(g, key=lambda x: x[2]))
                _self_idx.extend(s[:self.max_num_nbr])
                _nbr_idx.extend(n[:self.max_num_nbr])
                _nbr_dist.extend(d[:self.max_num_nbr])

            self_idx = np.array(_self_idx)
            nbr_idx = np.array(_nbr_idx)
            nbr_dist = np.array(_nbr_dist)

        # assert (self_idx == self_idx_old).all()
        # assert (nbr_idx == nbr_idx_old).all()
        # assert (nbr_dist == nbr_dist_old).all()

        assert len(self_idx), f"All atoms in {cif_id} are isolated"
        assert len(nbr_idx), f"This should not be triggered but was for {cif_id}"
        assert set(self_idx) == set(range(len(atom_fea))), f"At least one atom in {cif_id} is isolated"

        nbr_dist = self.gdf.expand(nbr_dist)
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = torch.Tensor(atom_fea)
        nbr_dist = torch.Tensor(nbr_dist)
        self_idx = torch.LongTensor(self_idx)
        nbr_idx = torch.LongTensor(nbr_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(torch.Tensor([df_idx[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(torch.LongTensor([df_idx[target]]))

        return ((atom_fea, nbr_dist, self_idx, nbr_idx), targets, comp, cif_id)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): Variance of Gaussian basis. Defaults to step if not given
        """
        assert dmin < dmax
        assert dmax - dmin > step

        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)

        if var is None:
            var = step

        self.var = var

    def expand(self, distances):
        """Apply Gaussian distance filter to a numpy distance array

        Args:
            distances (ArrayLike): A distance matrix of any shape

        Returns:
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        distances = np.array(distances)

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
        (atom_fea, nbr_dist, nbr_idx, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_dist: torch.Tensor shape (n_i, M, nbr_dist_len)
        nbr_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_dist: torch.Tensor shape (N, M, nbr_dist_len)
        Bond features of each atom's M neighbors
    batch_nbr_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea = []
    batch_nbr_dist = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_comps = []
    batch_cif_ids = []
    base_idx = 0

    for i, (inputs, target, comp, cif_id) in enumerate(dataset_list):
        atom_fea, nbr_dist, self_idx, nbr_idx = inputs
        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_dist.append(nbr_dist)
        batch_self_idx.append(self_idx + base_idx)
        batch_nbr_idx.append(nbr_idx + base_idx)

        crystal_atom_idx.extend([i] * n_i)
        batch_targets.append(target)
        batch_comps.append(comp)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    atom_fea = torch.cat(batch_atom_fea, dim=0)
    nbr_dist = torch.cat(batch_nbr_dist, dim=0)
    self_idx = torch.cat(batch_self_idx, dim=0)
    nbr_idx = torch.cat(batch_nbr_idx, dim=0)
    cry_idx = torch.LongTensor(crystal_atom_idx)

    return (
        (atom_fea, nbr_dist, self_idx, nbr_idx, cry_idx),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        batch_comps,
        batch_cif_ids,
    )


def get_structure(cols):
    """ Return pymatgen structure from lattice and sites cols """
    cell, sites = cols
    cell, elems, coords = parse_cgcnn(cell, sites)
    # NOTE getting primative structure before constructing graph
    # significantly harms the performnace of this model.
    return Structure(
        lattice=cell, species=elems, coords=coords, to_unit_cell=True
    )


def parse_cgcnn(cell, sites):
    """ Parse str representation into lists """
    cell = np.array(ast.literal_eval(cell), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(sites):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords
