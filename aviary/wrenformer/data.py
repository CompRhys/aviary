from __future__ import annotations

import json

import numpy as np
import torch
from pymatgen.core import Composition
from torch import LongTensor, Tensor, nn

from aviary import PKG_DIR
from aviary.wren.data import parse_aflow_wyckoff_str


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
    if features[0].ndim == 3:
        # wrenformer features are 3d with shape (n_equiv_wyksets [ragged],
        # n_wyckoff_sites_per_set [ragged], n_features [uniform])
        # we unpack the 1st dim into the batch dim and later take the mean of the
        # transformer-encoded embeddings of equivalent sets
        equivalence_counts = [len(tensor) for tensor in features]
        # unpack 3d embedding tensors along dim=0 and restack equivalent Wyckoff sets
        # along batch dim before padding
        restacked = tuple(aug for emb in features for aug in emb)
    else:
        restacked = features  # for roostformer we do nothing

    padded_features = nn.utils.rnn.pad_sequence(restacked, batch_first=True)

    # padded_features.shape = (batch_size * mean_n_equiv_wyksets, max_seq_len, n_features),
    # so we mask sequence items that are all zero across feature dimension
    mask = (padded_features == 0).all(dim=2)

    # insert outer dimension corresponding to different multi-tasking objectives
    targets = targets[None, ...]

    if features[0].ndim == 3:
        return (padded_features, mask, equivalence_counts), targets, ids

    return (padded_features, mask), targets, ids


with open(f"{PKG_DIR}/embeddings/wyckoff/bra-alg-off.json") as file:
    sym_features = json.load(file)
with open(f"{PKG_DIR}/embeddings/element/matscholar200.json") as file:
    elem_features = json.load(file)


def wyckoff_embedding_from_aflow_str(wyckoff_str: str) -> Tensor:
    """Concatenate matscholar element and Wyckoff set embeddings while handling
    augmentation of equivalent Wyckoff sets.

    Args:
        wyckoff_str (str): Aflow-style Wyckoff string.

    Returns:
        Tensor: Shape (n_equiv_wyksets, n_wyckoff_sites, n_features).
    """
    spg_num, elem_weights, elements, augmented_wyckoffs = parse_aflow_wyckoff_str(
        wyckoff_str
    )

    symmetry_features = torch.tensor(
        [
            [sym_features[spg_num][wyk] for wyk in equivalent_wyckoff_set]
            for equivalent_wyckoff_set in augmented_wyckoffs
        ]
    )

    n_augments = len(augmented_wyckoffs)  # number of equivalent Wyckoff sets
    element_features = torch.tensor([elem_features[el] for el in elements])
    element_features = element_features[None, ...].repeat(n_augments, 1, 1)

    element_ratios = torch.tensor(elem_weights)[None, :, None] / sum(elem_weights)
    element_ratios = element_ratios.repeat(n_augments, 1, 1)

    combined_features = torch.cat(
        [element_ratios, element_features, symmetry_features], dim=-1
    ).float()

    return combined_features


def get_composition_embedding(formula: str) -> Tensor:
    """Concatenate matscholar element embeddings with element ratios in compostion.

    Args:
        formula (str): Composition string.

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
