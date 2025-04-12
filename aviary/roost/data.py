from collections.abc import Sequence
from functools import cache
from typing import Any

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
        inputs: str = "composition",
        identifiers: Sequence[str] = ("material_id", "composition"),
    ):
        """Data class for Roost models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target
                names to task type.
            elem_embedding (str, optional): One of "matscholar200", "cgcnn92",
                "megnet16", "onehot112" or path to a file with custom element
                embeddings. Defaults to "matscholar200".
            inputs (str, optional): df column name holding material compositions.
                Defaults to "composition".
            identifiers (list, optional): df columns for distinguishing data points.
                Will be copied over into the model's output CSV. Defaults to
                ["material_id", "composition"].
        """
        if len(identifiers) != 2:
            raise AssertionError("Two identifiers are required")

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = list(identifiers)
        self.df = df

        self.n_targets = []
        for target, task in self.task_dict.items():
            if task == "regression":
                self.n_targets.append(1)
            elif task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        df_repr = f"cols=[{', '.join(self.df.columns)}], len={len(self.df)}"
        return f"{type(self).__name__}({df_repr}, task_dict={self.task_dict})"

    # Cache data for faster training
    @cache  # noqa: B019
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset.

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple: containing
            - tuple[Tensor, Tensor, LongTensor, LongTensor]: Roost model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        row = self.df.iloc[idx]
        composition = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        comp_dict = Composition(composition).fractional_composition
        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        elem_fea = [elem.Z for elem in comp_dict]

        n_elems = len(comp_dict)
        self_idx = []
        nbr_idx = []
        for elem_idx in range(n_elems):
            self_idx += [elem_idx] * n_elems
            nbr_idx += list(range(n_elems))

        # convert all data to tensors
        elem_weights = Tensor(weights)
        elem_fea = LongTensor(elem_fea)
        self_idx = LongTensor(self_idx)
        nbr_idx = LongTensor(nbr_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(Tensor([row[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(LongTensor([row[target]]))

        return (
            (elem_weights, elem_fea, self_idx, nbr_idx),
            targets,
            *material_ids,
        )


def collate_batch(
    samples: tuple[
        tuple[Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        samples (list): list of tuples for each data point where each tuple contains:
            (elem_fea, nbr_fea, nbr_idx, target)
            - elem_fea (Tensor): Atom hidden features before convolution
            - self_idx (LongTensor): Indices of the atom's self
            - nbr_idx (LongTensor): Indices of M neighbors of each atom
            - target (Tensor | LongTensor): target values containing floats for
                regression or integers as class labels for classification
            - cif_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor]: batched Roost
                model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            # TODO this last tuple is unpacked how to do type hint?
            *tuple[str | int]: Identifiers like material_id, composition
        ]
    """
    # define the lists
    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_elem_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(samples):
        elem_weights, elem_fea, self_idx, nbr_idx = inputs

        n_sites = elem_fea.shape[0]  # number of atoms for this crystal

        # batch the features together
        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx.append(torch.tensor([idx] * n_sites))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_sites

    return (
        (
            torch.cat(batch_elem_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_elem_idx),
        ),
        tuple(
            torch.stack(b_target, dim=0) for b_target in zip(*batch_targets, strict=False)
        ),
        *zip(*batch_cry_ids, strict=False),
    )
