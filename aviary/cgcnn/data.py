import ast
import functools
import json
from itertools import groupby
from os.path import abspath, dirname, exists, join
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset


class CrystalGraphData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: Dict[str, str],
        elem_emb: str = "cgcnn92",
        inputs: Sequence[str] = ["lattice", "sites"],
        identifiers: Sequence[str] = ["material_id", "composition"],
        radius: float = 5,
        max_num_nbr: int = 12,
        dmin: float = 0,
        step: float = 0.2,
    ):
        """Data class for CGCNN models. CrystalGraphData featurises crystal structures into
        neighbourhood graphs

        Args:
            df (pd.Dataframe): Pandas dataframe holding input and target values.
            task_dict ({target: task}): task dict for multi-task learning
            elem_emb (str, optional): One of "matscholar200", "cgcnn92", "megnet16", "onehot112" or
                path to a file with custom element embeddings. Defaults to "matscholar200".
            inputs (list, optional): df columns for lattice and sites. Defaults to
                ["lattice", "sites"].
            identifiers (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
            radius (float, optional): Cut-off radius for neighbourhood. Defaults to 5.
            max_num_nbr (int, optional): maximum number of neighbours to consider. Defaults to 12.
            dmin (float, optional): minimum distance in Gaussian basis. Defaults to 0.
            step (float, optional): increment size of gaussian basis. Defaults to 0.2.
        """
        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 2, "One input column required are required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers

        self.radius = radius
        self.max_num_nbr = max_num_nbr

        if elem_emb in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_emb = join(
                dirname(abspath(__file__)), f"../embeddings/element/{elem_emb}.json"
            )
        else:
            assert exists(elem_emb), f"{elem_emb} does not exist!"

        with open(elem_emb) as f:
            self.elem_features = json.load(f)

        for key, value in self.elem_features.items():
            self.elem_features[key] = np.array(value, dtype=float)

        self.elem_emb_len = len(list(self.elem_features.values())[0])

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size

        self.df = df
        self.df["Structure_obj"] = self.df[inputs].apply(get_structure, axis=1)

        self._pre_check()

        self.n_targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                self.n_targets.append(1)
            elif self.task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self):
        return len(self.df)

    def _get_nbr_data(self, crystal):
        """get neighbours for every site

        Args:
            crystal ([Structure]): pymatgen structure to get neighbours for
        """
        self_idx, nbr_idx, _, nbr_dist = crystal.get_neighbor_list(
            self.radius, numerical_tol=1e-8
        )

        if self.max_num_nbr is not None:
            _self_idx, _nbr_idx, _nbr_dist = [], [], []

            for i, g in groupby(zip(self_idx, nbr_idx, nbr_dist), key=lambda x: x[0]):
                s, n, d = zip(*sorted(g, key=lambda x: x[2]))
                _self_idx.extend(s[: self.max_num_nbr])
                _nbr_idx.extend(n[: self.max_num_nbr])
                _nbr_dist.extend(d[: self.max_num_nbr])

            self_idx = np.array(_self_idx)
            nbr_idx = np.array(_nbr_idx)
            nbr_dist = np.array(_nbr_dist)

        return self_idx, nbr_idx, nbr_dist

    def _pre_check(self):
        """Check that none of the structures have isolated atoms."""

        print("Precheck that all structures are valid")
        all_isolated = []
        some_isolated = []

        for cif_id, crystal in zip(self.df["material_id"], self.df["Structure_obj"]):
            self_idx, nbr_idx, _ = self._get_nbr_data(crystal)

            if len(self_idx) == 0:
                all_isolated.append(cif_id)
            elif len(nbr_idx) == 0:
                all_isolated.append(cif_id)
            elif set(self_idx) != set(range(crystal.num_sites)):
                some_isolated.append(cif_id)

        if not (all_isolated == some_isolated == []):
            # drop the data points that do not give rise to dense crystal graphs
            isolated = {*all_isolated, *some_isolated}  # set union
            self.df = self.df[~self.df["material_id"].isin(isolated)]

            print(f"all atoms in these structure are isolated: {all_isolated}")
            print(f"these structure have some isolated atoms: {some_isolated}")

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # NOTE sites must be given in fractional coordinates
        df_idx = self.df.iloc[idx]
        crystal = df_idx["Structure_obj"]
        cry_ids = df_idx[self.identifiers]

        # atom features for disordered sites
        site_atoms = [atom.species.as_dict() for atom in crystal]
        atom_fea = np.vstack(
            [
                np.sum(
                    [self.elem_features[el] * amt for el, amt in site.items()], axis=0
                )
                for site in site_atoms
            ]
        )

        # # # neighbours
        self_idx, nbr_idx, nbr_dist = self._get_nbr_data(crystal)

        assert len(self_idx), f"All atoms in {cry_ids} are isolated"
        assert len(nbr_idx), f"This should not be triggered but was for {cry_ids}"
        assert set(self_idx) == set(
            range(crystal.num_sites)
        ), f"At least one atom in {cry_ids} is isolated"

        nbr_dist = self.gdf.expand(nbr_dist)

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

        return (
            (atom_fea, nbr_dist, self_idx, nbr_idx),
            targets,
            *cry_ids,
        )


class GaussianDistance:
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
    """Return pymatgen structure from lattice and sites cols"""
    cell, sites = cols
    cell, elems, coords = parse_cgcnn(cell, sites)
    # NOTE getting primitive structure before constructing graph
    # significantly harms the performance of this model.
    return Structure(lattice=cell, species=elems, coords=coords, to_unit_cell=True)


def parse_cgcnn(cell, sites):
    """Parse str representation into lists"""
    cell = np.array(ast.literal_eval(cell), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(sites):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords
