from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from pymatgen.core import Composition
from torch import LongTensor, Tensor, nn

from aviary import PKG_DIR
from aviary.data import InMemoryDataLoader
from aviary.wren.data import parse_aflow_wyckoff_str

if TYPE_CHECKING:
    import pandas as pd


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
    """Concatenate Matscholar element embeddings with Wyckoff set embeddings and handle
    augmentation of equivalent Wyckoff sets.

    Args:
        wyckoff_str (str): Aflow-style Wyckoff string.

    Returns:
        Tensor: Shape (n_equiv_wyksets, n_wyckoff_sites, n_features) where n_features =
            200 + 444 for Matscholar and Wyckoff embeddings respectively.
    """
    parsed_output = parse_aflow_wyckoff_str(wyckoff_str)
    spg_num, wyckoff_site_multiplicities, elements, augmented_wyckoffs = parsed_output

    symmetry_features = torch.tensor(
        [
            [sym_features[spg_num][wyk_pos] for wyk_pos in equivalent_wyckoff_set]
            for equivalent_wyckoff_set in augmented_wyckoffs
        ]
    )

    n_augments = len(augmented_wyckoffs)  # number of equivalent Wyckoff sets
    element_features = torch.tensor([elem_features[el] for el in elements])
    element_features = element_features[None, ...].repeat(n_augments, 1, 1)

    element_ratios = torch.tensor(wyckoff_site_multiplicities)[None, :, None] / sum(
        wyckoff_site_multiplicities
    )
    element_ratios = element_ratios.repeat(n_augments, 1, 1)

    return torch.cat(  # combined features
        [element_ratios, element_features, symmetry_features], dim=-1
    ).float()


def get_composition_embedding(formula: str) -> Tensor:
    """Concatenate matscholar element embeddings with element ratios in composition.

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

    # combined features
    return torch.cat([element_ratios, element_features], dim=1).float()


def df_to_in_mem_dataloader(
    df: pd.DataFrame,
    input_col: str = "wyckoff",
    target_col: str | None = None,
    id_col: str | None = None,
    embedding_type: Literal["wyckoff", "composition"] = "wyckoff",
    device: str | None = None,
    **kwargs: Any,
) -> InMemoryDataLoader:
    """Construct an InMemoryDataLoader with Wrenformer batch collation from a dataframe.
    Can also be used for Roostformer. Does not (currently) work with other models like CGCNN/Wren.

    Args:
        df (pd.DataFrame): Expected to have columns input_col, target_col, id_col.
        input_col (str): Column name holding the input values (Aflow Wyckoff labels or composition
            strings) from which initial embeddings will be constructed. Defaults to "wyckoff".
        target_col (str): Column name holding the target values. Defaults to None. Only leave this
            empty if making predictions since target tensor will be set to list of Nones.
        id_col (str): Column name holding sample IDs. Defaults to None. If None, IDs will be
            the dataframe index.
        embedding_type ('wyckoff' | 'composition'): Defaults to "wyckoff".
        device (str): torch.device to load tensors onto. Defaults to
            "cuda" if torch.cuda.is_available() else "cpu".
        kwargs (dict): Keyword arguments like batch_size: int and shuffle: bool
            to pass to InMemoryDataLoader. Defaults to None.

    Returns:
        InMemoryDataLoader: Ready for use in model.evaluate(data_loader) or
            [model(x) for x in data_loader]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if embedding_type not in ["wyckoff", "composition"]:
        raise ValueError(f"{embedding_type = } must be 'wyckoff' or 'composition'")

    initial_embeddings = df[input_col].map(
        wyckoff_embedding_from_aflow_str
        if embedding_type == "wyckoff"
        else get_composition_embedding
    )
    targets = (
        torch.tensor(df[target_col].to_numpy(), device=device)
        if target_col in df
        else np.empty(len(df))
    )
    if targets.dtype == torch.bool:
        targets = targets.long()  # convert binary classification targets to 0 and 1
    inputs = np.empty(len(initial_embeddings), dtype=object)
    for idx, tensor in enumerate(initial_embeddings):
        inputs[idx] = tensor.to(device)

    ids = (df[id_col] if id_col in df else df.index).to_numpy()
    return InMemoryDataLoader(
        [inputs, targets, ids], collate_fn=collate_batch, **kwargs
    )
