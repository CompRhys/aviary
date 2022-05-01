from __future__ import annotations

import functools
import json
from os.path import dirname, exists, join
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import LongTensor, Tensor, nn
from torch.utils.data import Dataset

from aviary.wren.data import parse_aflow


class WyckoffData(Dataset):
    """Wyckoff dataset class for the Wren model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        element_embedding: str = "matscholar200",
        symmetry_embedding: str = "bra-alg-off",
        input_col: str = "wyckoff",
        id_cols: Sequence[str] = ("material_id", "composition", "wyckoff"),
    ):
        """Data class for Wren models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target names to task
                type for multi-task learning.
            element_embedding (str, optional): One of "matscholar200", "cgcnn92", "megnet16",
                "onehot112" or path to a file with custom element embeddings.
                Defaults to "matscholar200".
            symmetry_embedding (str): Built-in options are "bra-alg-off" (default) or
                "spg-alg-off". Can also be path to a file with custom symmetry embeddings.
            input_col (str, optional): Name of df column containing Aflow-style Wyckoff strings.
                Defaults to "wyckoff".
            id_cols (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
        """
        self.inputs = input_col
        self.task_dict = task_dict
        self.identifiers = list(id_cols)
        self.df = df

        if element_embedding in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            element_embedding = join(
                dirname(__file__), f"../embeddings/element/{element_embedding}.json"
            )
        else:
            if not exists(element_embedding):
                raise AssertionError(f"{element_embedding} does not exist!")

        with open(element_embedding) as f:
            self.elem_features = json.load(f)

        self.elem_emb_len = len(list(self.elem_features.values())[0])

        if symmetry_embedding in ["bra-alg-off", "spg-alg-off"]:
            symmetry_embedding = join(
                dirname(__file__), f"../embeddings/wyckoff/{symmetry_embedding}.json"
            )
        else:
            if not exists(symmetry_embedding):
                raise AssertionError(f"{symmetry_embedding} does not exist!")

        with open(symmetry_embedding) as f:
            self.sym_features = json.load(f)

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
        row = self.df.iloc[idx]
        wyckoff_str = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        spg_no, weights, elements, aug_wyks = parse_aflow(wyckoff_str)
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            element_features = np.vstack([self.elem_features[el] for el in elements])
        except AssertionError:
            print(f"Failed to process elements for {material_ids}")
            raise

        try:
            sym_fea = np.vstack(
                [self.sym_features[spg_no][wyk] for wyks in aug_wyks for wyk in wyks]
            )
        except AssertionError:
            print(f"Failed to process Wyckoff positions for {material_ids}")
            raise

        # convert all data to tensors
        element_ratios = torch.tensor(weights).repeat(len(aug_wyks), 1)
        element_features = torch.tensor(element_features).repeat(len(aug_wyks), 1)
        sym_fea = torch.tensor(sym_fea)

        combined_features = torch.cat(
            [element_ratios, element_features, sym_fea], dim=1
        ).float()

        targets = []
        for name in self.task_dict:
            if self.task_dict[name] == "regression":
                targets.append(Tensor([float(self.df[name].iloc[idx])]))
            elif self.task_dict[name] == "classification":
                targets.append(LongTensor([int(self.df[name].iloc[idx])]))

        return combined_features, targets, *material_ids


def collate_batch(
    dataset_list: tuple[Tensor, list[Tensor | LongTensor], list[str | int]]
):
    """Collate a list of data and return a batch for predicting
    crystal properties.

    Args:
        dataset_list ([tuple]): list of 3-tuples for each data point.
            (elem_fea, nbr_fea, nbr_idx, target)

            elem_fea (Tensor): Node features from atom type and Wyckoff letter
            nbr_fea (Tensor): _description_
            nbr_idx (LongTensor):
            targets (Tensor | LongTensor): Ground truth floats for regression or class labels.
            material_ids: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor]:
                batched Wren model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            *tuple[str | int]]: Identifiers like material_id, composition
        ]
    """
    features, targets, *material_ids = zip(*dataset_list)
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    # padded_features.shape = (batch_size, max_seq_len, n_features), so we mask sequence items that
    # are all zero across feature dimension
    mask = (padded_features == 0).all(dim=2)

    targets = torch.tensor(targets).T

    return ((padded_features, mask), targets, *zip(*material_ids))
