from __future__ import annotations

import functools
import json
import os
import re
from itertools import groupby
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from aviary import PKG_DIR
from aviary.wren.utils import relab_dict, wyckoff_multiplicity_dict


class WyckoffData(Dataset):
    """Wyckoff dataset class for the Wren model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        elem_emb: str = "matscholar200",
        sym_emb: str = "bra-alg-off",
        inputs: str = "wyckoff",
        identifiers: Sequence[str] = ("material_id", "composition", "wyckoff"),
    ):
        """Data class for Wren models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target names to task
                type for multi-task learning.
            elem_emb (str, optional): One of "matscholar200", "cgcnn92", "megnet16", "onehot112" or
                path to a file with custom element embeddings. Defaults to "matscholar200".
            sym_emb (str): Symmetry embedding. One of "bra-alg-off" (default) or "spg-alg-off".
            inputs (str, optional): df columns to be used for featurisation.
                Defaults to "wyckoff".
            identifiers (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
        """
        if len(identifiers) < 2:
            raise AssertionError("Two identifiers are required")

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = list(identifiers)
        self.df = df

        if elem_emb in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_emb = f"{PKG_DIR}/embeddings/element/{elem_emb}.json"
        elif not os.path.exists(elem_emb):
            raise AssertionError(f"{elem_emb} does not exist!")

        with open(elem_emb) as f:
            self.elem_features = json.load(f)

        self.elem_emb_len = len(list(self.elem_features.values())[0])

        if sym_emb in ["bra-alg-off", "spg-alg-off"]:
            sym_emb = f"{PKG_DIR}/embeddings/wyckoff/{sym_emb}.json"
        elif not os.path.exists(sym_emb):
            raise AssertionError(f"{sym_emb} does not exist!")

        with open(sym_emb) as f:
            self.sym_features = json.load(f)

        self.sym_emb_len = len(list(list(self.sym_features.values())[0].values())[0])

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

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple containing:
            - tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor]: Wren model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        row = self.df.iloc[idx]
        wyckoff_str = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        parsed_output = parse_aflow_wyckoff_str(wyckoff_str)
        spg_num, wyk_site_multiplcities, elements, augmented_wyks = parsed_output

        wyk_site_multiplcities = np.atleast_2d(wyk_site_multiplcities).T / np.sum(
            wyk_site_multiplcities
        )

        try:
            element_features = np.vstack([self.elem_features[el] for el in elements])
        except AssertionError:
            print(f"Failed to process elements for {material_ids}")
            raise

        try:
            symmetry_features = np.vstack(
                [
                    self.sym_features[spg_num][wyk_site]
                    for wyckoff_sites in augmented_wyks
                    for wyk_site in wyckoff_sites
                ]
            )
        except AssertionError:
            print(f"Failed to process Wyckoff positions for {material_ids}")
            raise

        n_wyks = len(elements)
        self_idx = []
        nbr_idx = []
        for i in range(n_wyks):
            self_idx += [i] * n_wyks
            nbr_idx += list(range(n_wyks))

        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(augmented_wyks)
        for i in range(n_aug):
            self_aug_fea_idx += [x + i * n_wyks for x in self_idx]
            nbr_aug_fea_idx += [x + i * n_wyks for x in nbr_idx]

        # convert all data to tensors
        wyckoff_weights = Tensor(wyk_site_multiplcities)
        element_features = Tensor(element_features)
        symmetry_features = Tensor(symmetry_features)
        self_idx = LongTensor(self_aug_fea_idx)
        nbr_idx = LongTensor(nbr_aug_fea_idx)

        targets = []
        for name in self.task_dict:
            if self.task_dict[name] == "regression":
                targets.append(Tensor([float(self.df.iloc[idx][name])]))
            elif self.task_dict[name] == "classification":
                targets.append(LongTensor([int(self.df.iloc[idx][name])]))

        return (
            (wyckoff_weights, element_features, symmetry_features, self_idx, nbr_idx),
            targets,
            *material_ids,
        )


def collate_batch(
    dataset_list: tuple[
        tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting
    crystal properties.

    Args:
        dataset_list ([tuple]): list of tuples for each data point.
            (elem_fea, nbr_fea, nbr_idx, target)

            elem_fea (Tensor): Node features from atom type and Wyckoff letter
            nbr_fea (Tensor): _description_
            nbr_idx (LongTensor):
            target (Tensor):
            cif_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor]:
                batched Wren model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            *tuple[str | int]]: Identifiers like material_id, composition
        ]
    """
    # define the lists
    batch_mult_weights = []
    batch_elem_fea = []
    batch_sym_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_wyk_idx = []
    aug_cry_idx = []
    batch_targets = []
    batch_cry_ids = []

    aug_count = 0
    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(dataset_list):
        mult_weights, elem_fea, sym_fea, self_idx, nbr_idx = inputs

        # number of atoms for this crystal
        n_el = elem_fea.shape[0]
        n_i = sym_fea.shape[0]
        n_aug = int(float(n_i) / float(n_el))

        # batch the features together
        batch_mult_weights.append(mult_weights.repeat((n_aug, 1)))
        batch_elem_fea.append(elem_fea.repeat((n_aug, 1)))
        batch_sym_fea.append(sym_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_wyk_idx.append(
            torch.tensor(range(aug_count, aug_count + n_aug)).repeat_interleave(n_el)
        )
        aug_cry_idx.append(torch.tensor([idx] * n_aug))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        aug_count += n_aug
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_mult_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_sym_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_wyk_idx),
            torch.cat(aug_cry_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )


def parse_aflow_wyckoff_str(
    aflow_label: str,
) -> tuple[str, list[float], list[str], list[tuple[str, ...]]]:
    """Parse the Wren AFLOW-like Wyckoff encoding.

    Args:
        aflow_label (str): AFLOW-style prototype string with appended chemical system

    Returns:
        tuple[str, list[float], list[str], list[str]]: spacegroup number, Wyckoff site
            multiplicities, elements symbols and equivalent wyckoff sets
    """
    proto, chemsys = aflow_label.split(":")
    elems = chemsys.split("-")
    _, _, spg_no, *wyckoff_letters = proto.split("_")

    wyckoff_site_multiplicities = []
    elements = []
    wyckoff_set = []

    for el, wyk_letters_per_elem in zip(elems, wyckoff_letters):
        # normalize Wyckoff letters to start with 1 if missing digit
        wyk_letters_per_elem = re.sub(
            r"((?<![0-9])[A-z])", r"1\g<1>", wyk_letters_per_elem
        )

        # Separate out pairs of Wyckoff letters and their number of occurrences
        sep_n_wyks = ["".join(g) for _, g in groupby(wyk_letters_per_elem, str.isalpha)]

        for n, l in zip(sep_n_wyks[0::2], sep_n_wyks[1::2]):
            m = int(n)
            elements.extend([el] * m)
            wyckoff_set.extend([l] * m)
            wyckoff_site_multiplicities.extend(
                [float(wyckoff_multiplicity_dict[spg_no][l])] * m
            )

    # NOTE This on-the-fly augmentation of equivalent Wyckoff sets could be a source of high
    # memory use. Can be turned off by commenting out the for loop and returning
    # [wyckoff_set] instead of augmented_wyckoff_set. Wren should be able to learn anyway.
    augmented_wyckoff_set = []
    for trans in relab_dict[spg_no]:
        # Apply translation dictionary of allowed relabelling operations in spacegroup
        t = str.maketrans(trans)
        augmented_wyckoff_set.append(
            tuple(",".join(wyckoff_set).translate(t).split(","))
        )

    augmented_wyckoff_set = list(set(augmented_wyckoff_set))

    return spg_no, wyckoff_site_multiplicities, elements, augmented_wyckoff_set
