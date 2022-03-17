from __future__ import annotations

import functools
import json
from os.path import abspath, dirname, exists, join
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from torch import LongTensor, Tensor
from torch.utils.data import Dataset


class CompositionData(Dataset):
    """Dataset class for the Roost composition model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        elem_emb: str = "matscholar200",
        inputs: Sequence[str] = ("composition",),
        identifiers: Sequence[str] = ("material_id", "composition"),
    ):
        """Data class for Roost models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target names to task
                type.
            elem_emb (str, optional): One of "matscholar200", "cgcnn92", "megnet16", "onehot112" or
                path to a file with custom embeddings. Defaults to "matscholar200".
            inputs (list[str], optional): df column name holding material compositions.
                Defaults to ["composition"].
            identifiers (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
        """
        if len(identifiers) != 2:
            raise AssertionError("Two identifiers are required")
        if len(inputs) != 1:
            raise AssertionError("One input column required are required")

        self.inputs = list(inputs)
        self.task_dict = task_dict
        self.identifiers = list(identifiers)
        self.df = df

        if elem_emb in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_emb = join(
                dirname(abspath(__file__)), f"../embeddings/element/{elem_emb}.json"
            )
        else:
            if not exists(elem_emb):
                raise AssertionError(f"{elem_emb} does not exist!")

        with open(elem_emb) as f:
            self.elem_features = json.load(f)

        self.elem_emb_len = len(list(self.elem_features.values())[0])

        self.n_targets = []
        for target, task in self.task_dict.items():
            if task == "regression":
                self.n_targets.append(1)
            elif task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple: containing
            - atom_weights (Tensor): weights of atoms in the material
            - atom_fea (Tensor): features of atoms in the material
            - self_idx (Tensor): list of self indices
            - nbr_idx (Tensor): list of neighbor indices
            - target (Tensor): target value for material
            - cry_id (Tensor): input id for the material
        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[self.inputs][0]
        cry_ids = df_idx[self.identifiers].values

        comp_dict = Composition(composition).get_el_amt_dict()
        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            atom_fea = np.vstack([self.elem_features[element] for element in elements])
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )

        nele = len(elements)
        self_idx = []
        nbr_idx = []
        for i, _ in enumerate(elements):
            self_idx += [i] * nele
            nbr_idx += list(range(nele))

        # convert all data to tensors
        atom_weights = Tensor(weights)
        atom_fea = Tensor(atom_fea)
        self_idx = LongTensor(self_idx)
        nbr_idx = LongTensor(nbr_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(Tensor([df_idx[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_idx, nbr_idx),
            targets,
            *cry_ids,
        )


def collate_batch(dataset_list):
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list (list): list of tuples for each data point: (atom_fea, nbr_fea, nbr_idx, target)
            - atom_fea (Tensor):
            - nbr_fea (Tensor):
            - self_idx (LongTensor):
            - nbr_idx (LongTensor):
            - target (Tensor):
            - cif_id: str or int

    Returns:
        tuple: containing
        - batch_atom_weights (Tensor): _description_
        - batch_atom_fea (Tensor): Atom features from atom type
        - batch_self_idx (LongTensor): Indices of mapping atom to copies of itself
        - batch_nbr_idx (LongTensor): Indices of M neighbors of each atom
        - crystal_atom_idx (list[LongTensor]): Mapping from the crystal idx to atom idx
        - target (Tensor): Target value for prediction
        - batch_comps: list
        - batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_weights, atom_fea, self_idx, nbr_idx = inputs

        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_atom_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )
