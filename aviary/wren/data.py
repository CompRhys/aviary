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
from aviary.wren.utils import mult_dict, relab_dict


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

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
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
        df_idx = self.df.iloc[idx]
        swyks = df_idx[self.inputs]
        cry_ids = df_idx[self.identifiers].to_list()

        spg_no, weights, elements, aug_wyks = parse_aflow(swyks)
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            elem_fea = np.vstack([self.elem_features[el] for el in elements])
        except AssertionError:
            print(f"Failed to process elements in {cry_ids[0]}: {cry_ids[1]}-{swyks}")
            raise

        try:
            sym_fea = np.vstack(
                [self.sym_features[spg_no][wyk] for wyks in aug_wyks for wyk in wyks]
            )
        except AssertionError:
            print(
                f"Failed to process Wyckoff positions in {cry_ids[0]}: {cry_ids[1]}-{swyks}"
            )
            raise

        n_wyks = len(elements)
        self_idx = []
        nbr_idx = []
        for i in range(n_wyks):
            self_idx += [i] * n_wyks
            nbr_idx += list(range(n_wyks))

        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(aug_wyks)
        for i in range(n_aug):
            self_aug_fea_idx += [x + i * n_wyks for x in self_idx]
            nbr_aug_fea_idx += [x + i * n_wyks for x in nbr_idx]

        # convert all data to tensors
        mult_weights = Tensor(weights)
        elem_fea = Tensor(elem_fea)
        sym_fea = Tensor(sym_fea)
        self_idx = LongTensor(self_aug_fea_idx)
        nbr_idx = LongTensor(nbr_aug_fea_idx)

        targets = []
        for name in self.task_dict:
            if self.task_dict[name] == "regression":
                targets.append(Tensor([float(self.df.iloc[idx][name])]))
            elif self.task_dict[name] == "classification":
                targets.append(LongTensor([int(self.df.iloc[idx][name])]))

        return (
            (mult_weights, elem_fea, sym_fea, self_idx, nbr_idx),
            targets,
            *cry_ids,
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


def parse_aflow(
    aflow_label: str,
) -> tuple[str, list[float], list[str], list[tuple[str, ...]]]:
    """Parse the Wren AFLOW-like Wyckoff encoding.

    Args:
        aflow_label (str): AFLOW-style prototype string with appended chemical system

    Returns:
        tuple[str, list[int], list[str], list[str]]: spg_no, mult_list, ele_list, aug_wyks
    """
    proto, chemsys = aflow_label.split(":")
    elems = chemsys.split("-")
    _, _, spg_no, *wyks = proto.split("_")

    mult_list = []
    ele_list = []
    wyk_list = []

    subst = r"1\g<1>"
    for el, wyk in zip(elems, wyks):

        # Put 1's in front of all Wyckoff letters not preceded by numbers
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)

        # Separate out pairs of Wyckoff letters and their number of occurrences
        sep_n_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]

        for n, l in zip(sep_n_wyks[0::2], sep_n_wyks[1::2]):
            m = int(n)
            ele_list.extend([el] * m)
            wyk_list.extend([l] * m)
            mult_list.extend([float(mult_dict[spg_no][l])] * m)

    # NOTE This on-the-fly augmentation of equivalent Wyckoff sets is potentially a source of high
    # memory use. Can be turned off by commenting out the for loop and returning [wyk_list] instead
    # of aug_wyks. Wren should be able to learn anyway.
    aug_wyks = []
    for trans in relab_dict[spg_no]:
        # Apply translation dictionary of allowed relabelling operations in spacegroup
        t = str.maketrans(trans)
        aug_wyks.append(tuple(",".join(wyk_list).translate(t).split(",")))

    aug_wyks = list(set(aug_wyks))

    return spg_no, mult_list, ele_list, aug_wyks
