from __future__ import annotations

import ast
import functools
import json
import os
from itertools import groupby
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from aviary import PKG_DIR


class CrystalGraphData(Dataset):
    """Dataset class for the CGCNN structure model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        elem_emb: str = "cgcnn92",
        inputs: Sequence[str] = ("lattice", "sites"),
        identifiers: Sequence[str] = ("material_id", "composition"),
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
            step (float, optional): increment size of Gaussian basis. Defaults to 0.2.
        """
        if len(identifiers) != 2:
            raise AssertionError("Two identifiers are required")
        if len(inputs) != 2:
            raise AssertionError("One input column required are required")

        self.inputs = list(inputs)
        self.task_dict = task_dict
        self.identifiers = list(identifiers)

        self.radius = radius
        self.max_num_nbr = max_num_nbr

        if elem_emb in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_emb = f"{PKG_DIR}/embeddings/element/{elem_emb}.json"
        else:
            if not os.path.exists(elem_emb):
                raise AssertionError(f"{elem_emb} does not exist!")

        with open(elem_emb) as f:
            self.elem_features = json.load(f)

        for key, value in self.elem_features.items():
            self.elem_features[key] = np.array(value, dtype=float)

        self.elem_emb_len = len(list(self.elem_features.values())[0])

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size

        self.df = df
        self.df["Structure_obj"] = self.df[self.inputs].apply(get_structure, axis=1)

        self._pre_check()

        self.n_targets = []
        for target, task_type in self.task_dict.items():
            if task_type == "regression":
                self.n_targets.append(1)
            elif task_type == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        df_repr = f"cols=[{', '.join(self.df.columns)}], len={len(self.df)}"
        return f"{type(self).__name__}({df_repr}, task_dict={self.task_dict})"

    def _get_nbr_data(
        self, crystal: Structure
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get neighbours for every site.

        Args:
            crystal (Structure): pymatgen Structure to get neighbours for

        Returns:
            tuple containing:
            - np.ndarray: Site indices
            - np.ndarray: Neighbour indices
            - np.ndarray: Distances between sites and neighbours
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

    def _pre_check(self) -> None:
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

        if not all_isolated == some_isolated == []:
            # drop the data points that do not give rise to dense crystal graphs
            isolated = {*all_isolated, *some_isolated}  # set union
            self.df = self.df[~self.df["material_id"].isin(isolated)]

            print(f"all atoms in these structure are isolated: {all_isolated}")
            print(f"these structure have some isolated atoms: {some_isolated}")

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple containing:
            - tuple[Tensor, Tensor, LongTensor, LongTensor]: CGCNN model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        # NOTE sites must be given in fractional coordinates
        row = self.df.iloc[idx]
        crystal = row["Structure_obj"]
        material_ids = row[self.identifiers]

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

        if not len(self_idx):
            raise AssertionError(f"All atoms in {material_ids} are isolated")
        if not len(nbr_idx):
            raise AssertionError(
                f"This should not be triggered but was for {material_ids}"
            )
        if set(self_idx) != set(range(crystal.num_sites)):
            raise AssertionError(f"At least one atom in {material_ids} is isolated")

        nbr_dist = self.gdf.expand(nbr_dist)

        atom_fea_t = Tensor(atom_fea)
        nbr_dist_t = Tensor(nbr_dist)
        self_idx_t = LongTensor(self_idx)
        nbr_idx_t = LongTensor(nbr_idx)

        targets: list[Tensor | LongTensor] = []
        for target, task_type in self.task_dict.items():
            if task_type == "regression":
                targets.append(Tensor([row[target]]))
            elif task_type == "classification":
                targets.append(LongTensor([row[target]]))

        return (
            (atom_fea_t, nbr_dist_t, self_idx_t, nbr_idx_t),
            targets,
            *material_ids,
        )


def collate_batch(
    dataset_list: tuple[
        tuple[Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list (list[tuple]): for each data point: (atom_fea, nbr_dist, nbr_idx, target)
            tuple[
                - atom_fea (Tensor): _description_
                - nbr_dist (Tensor):
                - self_idx (LongTensor):
                - nbr_idx (LongTensor):
            ]
            - target (Tensor | LongTensor): target values containing floats for regression or
                integers as class labels for classification
            - cif_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor]: batched CGCNN model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            # TODO this last tuple is unpacked how to do type hint?
            *tuple[str | int]: Identifiers like material_id, composition
        ]
    """
    batch_atom_fea = []
    batch_nbr_dist = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []
    base_idx = 0

    # TODO: unpacking (inputs, target, comp, cif_id) doesn't appear to match the doc string
    # for dataset_list, what about nbr_idx and comp?
    for idx, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_fea, nbr_dist, self_idx, nbr_idx = inputs
        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        # batch the features together
        batch_atom_fea.append(atom_fea)
        batch_nbr_dist.append(nbr_dist)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + base_idx)
        batch_nbr_idx.append(nbr_idx + base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.extend([idx] * n_i)

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        base_idx += n_i

    atom_fea = torch.cat(batch_atom_fea, dim=0)
    nbr_dist = torch.cat(batch_nbr_dist, dim=0)
    self_idx = torch.cat(batch_self_idx, dim=0)
    nbr_idx = torch.cat(batch_nbr_idx, dim=0)
    cry_idx = LongTensor(crystal_atom_idx)

    return (
        (atom_fea, nbr_dist, self_idx, nbr_idx, cry_idx),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
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
    cell = np.array(ast.literal_eval(str(cell)), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(str(sites)):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords


class GaussianDistance:
    """Expands the distance by Gaussian basis. Unit: angstrom."""

    def __init__(
        self, dmin: float, dmax: float, step: float, var: float = None
    ) -> None:
        """_summary_

        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): Variance of Gaussian basis. Defaults to step if not given.
        """
        if dmin >= dmax:
            raise AssertionError(
                "Max radii must be larger than minimum radii for Gaussian basis expansion"
            )
        if dmax - dmin <= step:
            raise AssertionError(
                "Max radii below minimum radii + step size - please increase dmax."
            )

        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)

        if var is None:
            var = step

        self.var = var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """Apply Gaussian distance filter to a numpy distance array

        Args:
            distances (ArrayLike): A distance matrix of any shape.

        Returns:
            np.ndarray: Expanded distance matrix with the last dimension of length len(self.filter)
        """
        distances = np.array(distances)

        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )
