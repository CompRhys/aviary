import os
import ast
import json
import functools
import pathlib

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from roost.core import Featurizer


class WyckoffData(Dataset):
    """
    The WrenData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        data_path,
        sym_path,
        fea_path,
        task_dict,
        inputs=["wyckoff"],
        identifiers=["material_id", "composition"],
    ):
        """[summary]

        Args:
            data_path ([type]): [description]
            sym_path ([type]): [description]
            fea_path ([type]): [description]
            task_dict ([type]): [description]
            inputs (list, optional): [description]. Defaults to ["composition"].
            identifiers (list, optional): [description]. Defaults to ["material_id", "composition"].
        """
        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 1, "One input column required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers

        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material composition
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.atom_features = Featurizer.from_json(fea_path)

        assert os.path.exists(sym_path), "{} does not exist!".format(sym_path)
        self.sym_features = Featurizer.from_json(sym_path)

        # TODO clean this up to use package reasources
        path = pathlib.Path(__file__).parent.absolute()
        with open(os.path.join(path, "relab.json"), 'r') as f:
            self.relab_dict = json.load(f)

        for k, v in self.relab_dict.items():
            self.relab_dict[k] = [{int(sk): sv for sk, sv in l.items()} for l in v]

        self.elem_fea_dim = self.atom_features.embedding_size
        self.sym_fea_dim = self.sym_features.embedding_size

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
                list of self indicies
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbour indicies
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material
        """
        cry_id, composition, swyks = self.df.iloc[idx][self.identifiers+self.inputs]

        weights, elements, aug_wyks = parse_wren(swyks, self.relab_dict)
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            atom_fea = np.vstack(
                [self.atom_features.get_fea(el) for el in elements]
            )
            sym_fea = np.vstack(
                [self.sym_features.get_fea(wyk) for wyks in aug_wyks for wyk in wyks]
            )
        except AssertionError:
            print(f"failed to process {cry_id}: {composition}")
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
            self_aug_fea_idx += [x + i*n_wyks for x in self_fea_idx]
            nbr_aug_fea_idx += [x + i*n_wyks for x in nbr_fea_idx]

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
            composition,
            cry_id,
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
    batch_comp = []
    batch_cry_ids = []

    aug_count = 0
    cry_base_idx = 0
    for i, (inputs, target, comp, cry_id) in enumerate(dataset_list):
        atom_weights, atom_fea, sym_fea, self_fea_idx, nbr_fea_idx = inputs

        # number of atoms for this crystal
        n_el = atom_fea.shape[0]
        n_i = sym_fea.shape[0]
        n_aug = int(float(n_i)/float(n_el))

        # batch the features together
        batch_atom_weights.append(atom_weights.repeat((n_aug, 1)))
        batch_atom_fea.append(atom_fea.repeat((n_aug, 1)))
        batch_sym_fea.append(sym_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        # print(torch.tensor(range(i, i+n_aug)).size())
        crystal_atom_idx.append(torch.tensor(range(aug_count, aug_count+n_aug)).repeat_interleave(n_el))
        aug_cry_idx.append(torch.tensor([i] * n_aug))

        # batch the targets and ids
        batch_targets.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

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
        batch_comp,
        batch_cry_ids,
    )


def parse_wren(swyk_list, relab_dict):
    """parse the wyckoff format

    Args:
        swyk_list ([type]): [description]
        relab_dict ([type]): [description]

    Returns:
        mult_list, ele_list, aug_wyks
    """
    swyk_list = ast.literal_eval(swyk_list)

    mult_list = []
    ele_list = []
    wyk_list = []

    for swyk in swyk_list:
        # mult, ele, wyk = swyk.split("_")
        ele, wyk = swyk.split(" @ ")
        mult, _, spg = wyk.split("-")
        mult_list.append(float(mult))
        ele_list.append(ele)
        wyk_list.append(wyk)

    aug_wyks = []
    for trans in relab_dict[spg]:
        t = str.maketrans(trans)
        aug_wyks.append(
            tuple(",".join(wyk_list).translate(t).split(","))
        )

    aug_wyks = list(set(aug_wyks))

    return mult_list, ele_list, aug_wyks
