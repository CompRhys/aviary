import functools
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen import Composition

from roost.core import Featurizer


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        data_path,
        fea_path,
        task_dict,
        inputs=["composition"],
        identifiers=["id", "composition"],
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            inputs (list, optional): column name for compositions.
                Defaults to ["composition"].
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers

        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size

        self.n_targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                self.n_targets.append(1)
            elif self.task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
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
        composition = df_idx[self.inputs]
        cry_id, composition = df_idx[self.identifiers]

        comp_dict = Composition(composition).get_el_amt_dict()
        elements = list(comp_dict.keys())
        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elements) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

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
            composition, cry_id
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
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, comp, cry_id) in enumerate(dataset_list):
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
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

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
        (torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        batch_comp,
        batch_cry_ids,
    )
