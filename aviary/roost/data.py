import functools
import json
from os.path import abspath, dirname, exists, join
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from torch.utils.data import Dataset


class CompositionData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: Dict[str, str],
        elem_emb: str = "matscholar200",
        inputs: Sequence[str] = ["composition"],
        identifiers: Sequence[str] = ["material_id", "composition"],
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

        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 1, "One input column required are required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers
        self.df = df

        if elem_emb in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_emb = join(
                dirname(abspath(__file__)), f"../embeddings/element/{elem_emb}.json"
            )
        else:
            assert exists(elem_emb), f"{elem_emb} does not exist!"

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
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

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
        self_fea_idx = []
        nbr_fea_idx = []
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nele
            nbr_fea_idx += list(range(nele))

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(torch.Tensor([df_idx[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            *cry_ids,
        )


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = inputs

        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

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
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )
