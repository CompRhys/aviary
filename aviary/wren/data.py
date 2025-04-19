import json
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from itertools import groupby
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.prototypes import (
    RE_SUBST_ONE_PREFIX,
    RE_WYCKOFF_NO_PREFIX,
    WYCKOFF_MULTIPLICITY_DICT,
    WYCKOFF_POSITION_RELAB_DICT,
)
from pymatgen.core import Element
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from aviary import PKG_DIR

with open(f"{PKG_DIR}/embeddings/wyckoff/bra-alg-off.json") as f:
    sym_embeddings = json.load(f)
WYCKOFF_SPG_LETTER_MAP: dict[str, dict[str, int]] = defaultdict(dict)
i = 0
for spg_num, embeddings in sym_embeddings.items():
    for wyckoff_letter in embeddings:
        WYCKOFF_SPG_LETTER_MAP[spg_num][wyckoff_letter] = i
        i += 1

del sym_embeddings


class WyckoffData(Dataset):
    """Wyckoff dataset class for the Wren model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        inputs: str = "protostructure",
        identifiers: Sequence[str] = ("material_id", "composition", "protostructure"),
    ):
        """Data class for Wren models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target
                names to task type for multi-task learning.
            inputs (str, optional): df columns to be used for featurization.
                Defaults to "protostructure".
            identifiers (list, optional): df columns for distinguishing data points.
                Will be copied over into the model's output CSV. Defaults to
                ["material_id", "composition", "protostructure"].
        """
        if len(identifiers) < 2:
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

    @cache  # noqa: B019
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset.

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple containing:
            - tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor]: Wren model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        row = self.df.iloc[idx]
        protostructure_label = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        parsed_output = parse_protostructure_label(protostructure_label)
        spg_num, wyk_site_multiplcities, elements, augmented_wyks = parsed_output

        wyk_site_multiplcities = np.atleast_2d(wyk_site_multiplcities).T / np.sum(
            wyk_site_multiplcities
        )

        element_features = [Element(el).Z for el in elements]

        symmetry_features = [
            WYCKOFF_SPG_LETTER_MAP[spg_num][wyk_site]
            for wyckoff_sites in augmented_wyks
            for wyk_site in wyckoff_sites
        ]

        n_wyks = len(elements)
        self_idx = []
        nbr_idx = []
        for wyk_idx in range(n_wyks):
            self_idx += [wyk_idx] * n_wyks
            nbr_idx += list(range(n_wyks))

        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(augmented_wyks)
        for aug_idx in range(n_aug):
            self_aug_fea_idx += [x + aug_idx * n_wyks for x in self_idx]
            nbr_aug_fea_idx += [x + aug_idx * n_wyks for x in nbr_idx]

        # convert all data to tensors
        wyckoff_weights = Tensor(wyk_site_multiplcities)
        element_features = LongTensor(element_features)
        symmetry_features = LongTensor(symmetry_features)
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
    samples: tuple[
        tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting
    crystal properties.
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
    for idx, (inputs, target, *cry_ids) in enumerate(samples):
        mult_weights, elem_fea, sym_fea, self_idx, nbr_idx = inputs

        n_elem = elem_fea.shape[0]
        n_sites = sym_fea.shape[0]  # number of atoms for this crystal
        n_aug = int(float(n_sites) / float(n_elem))

        # batch the features together
        batch_mult_weights.append(mult_weights.repeat((n_aug, 1)))
        batch_elem_fea.append(elem_fea.repeat(n_aug))
        batch_sym_fea.append(sym_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_wyk_idx.append(
            torch.tensor(range(aug_count, aug_count + n_aug)).repeat_interleave(n_elem)
        )
        aug_cry_idx.append(torch.tensor([idx] * n_aug))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        aug_count += n_aug
        cry_base_idx += n_sites

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
        tuple(
            torch.stack(b_target, dim=0) for b_target in zip(*batch_targets, strict=False)
        ),
        *zip(*batch_cry_ids, strict=False),
    )


def parse_protostructure_label(
    protostructure_label: str,
) -> tuple[str, list[float], list[str], list[tuple[str, ...]]]:
    """Parse the Wren AFLOW-like Wyckoff encoding.

    Args:
        protostructure_label (str): label constructed as `aflow_label:chemsys` where
            aflow_label is an AFLOW-style prototype label chemsys is the alphabetically
            sorted chemical system.

    Returns:
        tuple[str, list[float], list[str], list[str]]: spacegroup number, Wyckoff site
            multiplicities, elements symbols and equivalent wyckoff sets
    """
    aflow_label, chemsys = protostructure_label.split(":")
    elems = chemsys.split("-")
    _, _, spg_num, *wyckoff_letters = aflow_label.split("_")

    if len(elems) != len(wyckoff_letters):
        raise ValueError(
            f"Chemical system {chemsys} does not match Wyckoff letters {wyckoff_letters}"
        )

    wyckoff_site_multiplicities = []
    elements = []
    wyckoff_set = []

    for el, wyk_letters_per_elem in zip(elems, wyckoff_letters, strict=False):
        # Normalize Wyckoff letters to start with 1 if missing digit
        wyk_letters_normalized = RE_WYCKOFF_NO_PREFIX.sub(
            RE_SUBST_ONE_PREFIX, wyk_letters_per_elem
        )

        # Separate out pairs of Wyckoff letters and their number of occurrences
        sep_n_wyks = ["".join(g) for _, g in groupby(wyk_letters_normalized, str.isalpha)]

        # Process Wyckoff letters and multiplicities
        mults = map(int, sep_n_wyks[0::2])
        letters = sep_n_wyks[1::2]

        for mult, letter in zip(mults, letters, strict=False):
            elements.extend([el] * mult)
            wyckoff_set.extend([letter] * mult)
            wyckoff_site_multiplicities.extend(
                [float(WYCKOFF_MULTIPLICITY_DICT[spg_num][letter])] * mult
            )

    # Create augmented Wyckoff set
    augmented_wyckoff_set = {
        tuple(",".join(wyckoff_set).translate(str.maketrans(trans)).split(","))
        for trans in WYCKOFF_POSITION_RELAB_DICT[spg_num]
    }

    return spg_num, wyckoff_site_multiplicities, elements, list(augmented_wyckoff_set)
