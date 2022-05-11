from __future__ import annotations

import json

import numpy as np
import torch
from pymatgen.core import Composition
from torch import LongTensor, Tensor, nn

from aviary import PKG_DIR
from aviary.wren.data import parse_aflow


def collate_batch(
    features: tuple[Tensor], targets: Tensor | LongTensor, ids: list[str | int]
):
    """Zero-pad sequences of Wyckoff embeddings to the longest one in the batch and
    generate a mask to ignore padded values during self-attention.

    Args:
        features (tuple[Tensor]): Wyckoff embeddings
        targets (list[Tensor | LongTensor]): For each multi-task objective, a float tensor
            for regression or integer class labels for classification.
        ids (list[str | int]): Material identifiers. Can be anything

    Returns:
        tuple: Tuple of padded features and mask, targets and ids.
    """
    padded_features = nn.utils.rnn.pad_sequence(tuple(features), batch_first=True)
    # padded_features.shape = (batch_size, max_seq_len, n_features), so we mask sequence items that
    # are all zero across feature dimension
    mask = (padded_features == 0).all(dim=2)

    # insert outer dimension corresponding to different multi-tasking objectives
    targets = targets[None, ...]

    return (padded_features, mask), targets, ids


with open(f"{PKG_DIR}/embeddings/wyckoff/bra-alg-off.json") as file:
    sym_features = json.load(file)
with open(f"{PKG_DIR}/embeddings/element/matscholar200.json") as file:
    elem_features = json.load(file)


def wyckoff_embedding_from_aflow_str(wyckoff_str: str) -> Tensor:
    """Concatenate matscholar element and Wyckoff set embeddings while handling
    augmentation from equivalent Wyckoff sets.

    Args:
        wyckoff_str (str): Aflow-style Wyckoff string.

    Returns:
        Tensor: Shape (n_augmentations, n_features).
    """
    spg_num, elem_weights, elements, augmented_wyckoffs = parse_aflow(wyckoff_str)

    elem_weights = np.atleast_2d(elem_weights).T / np.sum(elem_weights)

    element_features = np.vstack([elem_features[el] for el in elements])

    symmetry_features = np.vstack(
        [sym_features[spg_num][wyk] for wyks in augmented_wyckoffs for wyk in wyks]
    )

    n_augments = len(augmented_wyckoffs)  # number of equivalent Wyckoff sets
    # convert all data to tensors
    element_ratios = torch.tensor(elem_weights).repeat(n_augments, 1)
    element_features = torch.tensor(element_features).repeat(n_augments, 1)
    symmetry_features = torch.tensor(symmetry_features)

    combined_features = torch.cat(
        [element_ratios, element_features, symmetry_features], dim=1
    ).float()

    return combined_features


def get_composition_embedding(formula: str) -> Tensor:
    """Concatenate matscholar element and Wyckoff set embeddings while handling
    augmentation from equivalent Wyckoff sets.

    Args:
        wyckoff_str (str): Aflow-style Wyckoff string.

    Returns:
        Tensor: Shape (n_elements, n_features). Usually (2-6, 200).
    """
    composition_dict = Composition(formula).get_el_amt_dict()
    elements, elem_weights = zip(*composition_dict.items())

    elem_weights = np.atleast_2d(elem_weights).T / sum(elem_weights)

    element_features = np.vstack([elem_features[el] for el in elements])

    # convert all data to tensors
    element_ratios = torch.tensor(elem_weights)
    element_features = torch.tensor(element_features)

    combined_features = torch.cat([element_ratios, element_features], dim=1).float()

    return combined_features
