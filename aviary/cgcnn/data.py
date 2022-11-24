from __future__ import annotations

import ast
import functools
import itertools
import json
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch import LongTensor, Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from aviary import PKG_DIR


class CrystalGraphData(Dataset):
    """Dataset class for the CGCNN structure model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        elem_embedding: str = "cgcnn92",
        structure_col: str = "structure",
        identifiers: Sequence[str] = ("material_id",),
        radius: float = 5,
        max_num_nbr: int = 12,
        dmin: float = 0,
        step: float = 0.2,
    ):
        """Data class for CGCNN models. CrystalGraphData featurizes crystal structures into
        neighborhood graphs

        Args:
            df (pd.Dataframe): Pandas dataframe holding input and target values.
            task_dict ({target: task}): task dict for multi-task learning
            elem_embedding (str, optional): One of "matscholar200", "cgcnn92", "megnet16",
                "onehot112" or path to a file with custom element embeddings.
                Defaults to "matscholar200".
            structure_col (str, optional): df column holding pymatgen Structure objects as input.
            identifiers (list[str], optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ("material_id",).
            radius (float, optional): Cut-off radius for neighborhood. Defaults to 5.
            max_num_nbr (int, optional): maximum number of neighbors to consider. Defaults to 12.
            dmin (float, optional): minimum distance in Gaussian basis. Defaults to 0.
            step (float, optional): increment size of Gaussian basis. Defaults to 0.2.
        """
        self.task_dict = task_dict
        self.identifiers = list(identifiers)

        self.radius = radius
        self.max_num_nbr = max_num_nbr

        if elem_embedding in ("matscholar200", "cgcnn92", "megnet16", "onehot112"):
            elem_embedding = f"{PKG_DIR}/embeddings/element/{elem_embedding}.json"
        elif not os.path.isfile(elem_embedding):
            raise ValueError(f"{elem_embedding} does not exist!")

        with open(elem_embedding) as f:
            self.elem_features = json.load(f)

        for key, value in self.elem_features.items():
            self.elem_features[key] = np.array(value, dtype=float)
            if not hasattr(self, "elem_emb_len"):
                self.elem_emb_len = len(value)
            elif self.elem_emb_len != len(value):
                raise ValueError("Element embedding length mismatch!")

        self.gaussian_dist_func = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.nbr_fea_dim = self.gaussian_dist_func.embedding_size

        self.df = df
        self.structure_col = structure_col

        all_isolated = []
        some_isolated = []

        for material_id, struct in tqdm(
            zip(self.df[identifiers[0]], self.df[structure_col]),
            total=len(df),
            desc="Pre-check that all structures are valid, i.e. none have isolated atoms.",
            disable=None,
        ):
            self_idx, nbr_idx, _ = get_structure_neighbor_info(
                struct, radius, max_num_nbr
            )

            if 0 in (len(self_idx), len(nbr_idx)):
                all_isolated.append(material_id)
            if set(self_idx) != set(range(len(struct))):
                some_isolated.append(material_id)

        isolated = set(all_isolated + some_isolated)
        if len(isolated) > 0:
            # drop the data points that do not give rise to dense crystal graphs
            # TODO next line requires identifiers[0] == df.index, bad assumption
            self.df = self.df.drop(index=isolated)

            print(f"dropping {len(isolated):,} structures:")
            print(f"{len(all_isolated)} have only isolated atoms: {all_isolated}")
            print(f"{len(some_isolated)} have some isolated atoms: {some_isolated}")

        self.n_targets = []
        for target, task_type in self.task_dict.items():
            if task_type == "regression":
                self.n_targets.append(1)
            elif task_type == "classification":
                n_classes = max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        df_repr = f"cols=[{', '.join(self.df)}], len={len(self.df)}"
        return f"{type(self).__name__}({df_repr}, task_dict={self.task_dict})"

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
        # TODO try if converting to np array speeds this up due to np's faster indexing
        row = self.df.iloc[idx]
        struct = row[self.structure_col]
        material_id = row[self.identifiers[0]]

        # atom features for disordered sites
        site_atoms = [atom.species.as_dict() for atom in struct]
        atom_features = np.vstack(
            [
                np.sum(
                    [self.elem_features[el] * amt for el, amt in site.items()], axis=0
                )
                for site in site_atoms
            ]
        )

        self_idx, nbr_idx, nbr_dist = get_structure_neighbor_info(
            struct, self.radius, self.max_num_nbr
        )

        if len(self_idx) == 0:
            raise ValueError(f"All atoms in {material_id} are isolated")
        if len(nbr_idx) == 0:
            raise ValueError(
                f"Empty nbr_idx. This should not be triggered but was for {material_id}"
            )
        if set(self_idx) != set(range(len(struct))):
            raise ValueError(f"At least one atom in {material_id} is isolated")

        nbr_dist = self.gaussian_dist_func.expand(nbr_dist)

        atom_fea_t = Tensor(atom_features)
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
            *row[self.identifiers],
        )


def collate_batch(
    samples: tuple[
        tuple[Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ]
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
            - material_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor]: batched CGCNN model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            *tuple[str | int]: identifiers like material_id, composition
        ]
    """
    batch_atom_fea = []
    batch_nbr_dist = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_material_ids = []
    base_idx = 0

    # TODO: unpacking (inputs, target, comp, material_id) doesn't appear to match the doc string
    # for dataset_list, what about nbr_idx and comp?
    for idx, (inputs, target, *material_id) in enumerate(samples):
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

        # batch the targets and material_ids
        batch_targets.append(target)
        batch_material_ids.append(material_id)

        # increment the id counter
        base_idx += n_i

    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_fea = torch.cat(batch_atom_fea, dim=0).to(device)
    nbr_dist = torch.cat(batch_nbr_dist, dim=0).to(device)
    self_idx = torch.cat(batch_self_idx, dim=0).to(device)
    nbr_idx = torch.cat(batch_nbr_idx, dim=0).to(device)
    cry_idx = LongTensor(crystal_atom_idx).to(device)

    return (
        (atom_fea, nbr_dist, self_idx, nbr_idx, cry_idx),
        tuple(
            torch.stack(b_target, dim=0).to(device) for b_target in zip(*batch_targets)
        ),
        *zip(*batch_material_ids),
    )


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
        """Used by CGCNN to featurize neighbor atom distances.

        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): Variance of Gaussian basis. Defaults to step if not given.
        """
        if dmin >= dmax:
            raise ValueError(
                "Max radii must be larger than minimum radii for Gaussian basis expansion"
            )
        if dmax - dmin <= step:
            raise ValueError(
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

        return np.exp(-((distances[..., None] - self.filter) ** 2) / self.var**2)


def get_structure_neighbor_info(
    struct: Structure, radius: float = 5, max_num_nbr: int | None = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get neighbors for every site.

    Args:
        crystal (Structure): pymatgen Structure to get neighbors for

    Returns:
        tuple containing:
        - np.ndarray: Site indices
        - np.ndarray: Neighbor indices
        - np.ndarray: Distances between sites and neighbors
    """
    site_indices, neighbor_indices, _, neighbor_dists = struct.get_neighbor_list(
        radius, numerical_tol=1e-8
    )

    if max_num_nbr is not None:
        _center_indices, _neighbor_indices, _neighbor_dists = [], [], []

        for _, idx_group in itertools.groupby(  # group by site index
            zip(site_indices, neighbor_indices, neighbor_dists), key=lambda x: x[0]
        ):
            site_indices, neighbor_idx, neighbor_dist = zip(
                *sorted(idx_group, key=lambda x: x[2])  # sort by distance
            )
            _center_indices.extend(site_indices[:max_num_nbr])
            _neighbor_indices.extend(neighbor_idx[:max_num_nbr])
            _neighbor_dists.extend(neighbor_dist[:max_num_nbr])

        site_indices = np.array(_center_indices)
        neighbor_indices = np.array(_neighbor_indices)
        neighbor_dists = np.array(_neighbor_dists)

    return site_indices, neighbor_indices, neighbor_dists
