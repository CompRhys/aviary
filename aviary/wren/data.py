import functools
import json
import re
from itertools import groupby
from os.path import abspath, dirname, exists, join
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from aviary.wren.utils import mult_dict, relab_dict


class WyckoffData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: Dict[str, str],
        elem_emb: str = "matscholar200",
        sym_emb: str = "bra-alg-off",
        inputs: Sequence[str] = ["wyckoff"],
        identifiers: Sequence[str] = ["material_id", "composition", "wyckoff"],
    ):
        """Data class for Wren models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target names to task
                type for multi-task learning.
            elem_emb (str, optional): One of "matscholar200", "cgcnn92", "megnet16", "onehot112" or
                path to a file with custom element embeddings. Defaults to "matscholar200".
            sym_emb (str): Symmetry embedding. One of "bra-alg-off" (default) or "spg-alg-off".
            inputs (list[str], optional): df columns to be used for featurisation.
                Defaults to ["composition"].
            identifiers (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
        """
        assert len(identifiers) >= 2, "Two identifiers are required"
        assert len(inputs) == 1, "One input column required"

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

        if sym_emb in ["bra-alg-off", "spg-alg-off"]:
            sym_emb = join(
                dirname(abspath(__file__)), f"../embeddings/wyckoff/{sym_emb}.json"
            )
        else:
            assert exists(sym_emb), f"{sym_emb} does not exist!"

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

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Raises:
            AssertionError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbour indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material
        """
        df_idx = self.df.iloc[idx]
        swyks = df_idx[self.inputs][0]
        cry_ids = df_idx[self.identifiers].values

        spg_no, weights, elements, aug_wyks = parse_aflow(swyks)
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            atom_fea = np.vstack([self.elem_features[el] for el in elements])
            sym_fea = np.vstack(
                [self.sym_features[spg_no][wyk] for wyks in aug_wyks for wyk in wyks]
            )
        except AssertionError:
            print(f"failed to process {cry_ids[0]}: {cry_ids[1]}-{swyks}")
            raise

        n_wyks = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []
        for i in range(n_wyks):
            self_fea_idx += [i] * n_wyks
            nbr_fea_idx += list(range(n_wyks))

        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(aug_wyks)
        for i in range(n_aug):
            self_aug_fea_idx += [x + i * n_wyks for x in self_fea_idx]
            nbr_aug_fea_idx += [x + i * n_wyks for x in nbr_fea_idx]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        sym_fea = torch.Tensor(sym_fea)
        self_fea_idx = torch.LongTensor(self_aug_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_aug_fea_idx)

        targets = []
        for name in self.task_dict:
            if self.task_dict[name] == "regression":
                targets.append(torch.Tensor([float(self.df.iloc[idx][name])]))
            elif self.task_dict[name] == "classification":
                targets.append(torch.LongTensor([int(self.df.iloc[idx][name])]))

        return (
            (atom_weights, atom_fea, sym_fea, self_fea_idx, nbr_fea_idx),
            targets,
            *cry_ids,
        )


def collate_batch(dataset_list):
    """Collate a list of data and return a batch for predicting
    crystal properties.

    N = sum(n_i); N0 = sum(i)

    Args:
        dataset_list ([tuple]): list of tuples for each data point.
            (atom_fea, nbr_fea, nbr_fea_idx, target)

            atom_fea: torch.Tensor shape (n_i, atom_fea_len)
            nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
            nbr_fea_idx: torch.LongTensor shape (n_i, M)
            target: torch.Tensor shape (1, )
            cif_id: str or int

    Returns:
        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
            Atom features from atom type
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features of each atom"s M neighbors
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
            Target value for prediction
        batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_sym_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    aug_cry_idx = []
    batch_targets = []
    batch_cry_ids = []

    aug_count = 0
    cry_base_idx = 0
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_weights, atom_fea, sym_fea, self_fea_idx, nbr_fea_idx = inputs

        # number of atoms for this crystal
        n_el = atom_fea.shape[0]
        n_i = sym_fea.shape[0]
        n_aug = int(float(n_i) / float(n_el))

        # batch the features together
        batch_atom_weights.append(atom_weights.repeat((n_aug, 1)))
        batch_atom_fea.append(atom_fea.repeat((n_aug, 1)))
        batch_sym_fea.append(sym_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(
            torch.tensor(range(aug_count, aug_count + n_aug)).repeat_interleave(n_el)
        )
        aug_cry_idx.append(torch.tensor([i] * n_aug))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        aug_count += n_aug
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_sym_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
            torch.cat(aug_cry_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )


def parse_aflow(aflow_label):
    """parse the wyckoff format

    Args:
        swyk_list ([type]): [description]
        relab_dict ([type]): [description]

    Returns:
        mult_list, ele_list, aug_wyks
    """
    proto, chemsys = aflow_label.split(":")
    elems = chemsys.split("-")
    _, _, spg_no, *wyks = proto.split("_")

    mult_list = []
    ele_list = []
    wyk_list = []

    subst = r"1\g<1>"
    for el, wyk in zip(elems, wyks):
        wyk = re.sub(r"((?<![0-9])[A-z])", subst, wyk)
        sep_n_wyks = ["".join(g) for _, g in groupby(wyk, str.isalpha)]

        for n, l in zip(sep_n_wyks[0::2], sep_n_wyks[1::2]):
            n = int(n)
            ele_list.extend([el] * n)
            wyk_list.extend([l] * n)
            mult_list.extend([float(mult_dict[spg_no][l])] * n)

    aug_wyks = []
    for trans in relab_dict[spg_no]:
        t = str.maketrans(trans)
        aug_wyks.append(tuple(",".join(wyk_list).translate(t).split(",")))

    aug_wyks = list(set(aug_wyks))

    return spg_no, mult_list, ele_list, aug_wyks
