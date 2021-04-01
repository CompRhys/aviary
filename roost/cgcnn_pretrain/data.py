import ast
import functools
import os

import numpy as np
import pandas as pd

from random import sample
from itertools import groupby

from pymatgen.core.structure import Structure

import torch
from torch.utils.data import Dataset

from roost.core import Featurizer


class CrystalGraphData(Dataset):
    def __init__(
        self,
        data_path,
        fea_path,
        task_dict,
        inputs=["lattice", "sites"],
        identifiers=["material_id", "composition"],
        radius=5,
        max_num_nbr=12,
        dmin=0,
        step=0.2,
        p_mask=0.15,
    ):
        """CrystalGraphData returns neighbourhood graphs

        Args:
            data_path (str): The path to the dataset
            fea_path (str): The path to the element embedding
            task_dict ({target: task}): task dict for multi-task learning
            inputs (list, optional): df columns for lattice and sites.
                Defaults to ["lattice", "sites"].
            identifiers (list, optional): df columns for distinguishing data points.
                Defaults to ["material_id", "composition"].
            radius (int, optional): cut-off radius for neighbourhood. Defaults to 5.
            max_num_nbr (int, optional): maximum number of neighbours to consider. Defaults to 12.
            dmin (int, optional): minimum distance in gaussian basis. Defaults to 0.
            step (float, optional): increment size of gaussian basis. Defaults to 0.2.
        """
        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 2, "One input column required are required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers
        self.radius = radius
        self.max_num_nbr = max_num_nbr

        assert os.path.exists(data_path), "{} does not exist!".format(data_path)

        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[], comment="#")

        self.df["Structure_obj"] = self.df[inputs].apply(get_structure, axis=1)

        self._pre_check()

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.ari = Featurizer.from_json(fea_path)
        self.ohe = Featurizer.from_json("data/el-embeddings/onehot-embedding.json")
        self.elem_fea_dim = self.ari.embedding_size

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size
        self.p_mask = p_mask

        self.n_targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "mask":
                self.n_targets.append(self.ohe.embedding_size)
            elif self.task_dict[target] == "pretrain-global":
                raise NotImplementedError("bad user")

    def __len__(self):
        return len(self.df)

    def _get_nbr_data(self, crystal):
        """get neighbours for every site

        Args:
            crystal ([Structure]): pymatgen structure to get neighbours for
        """
        # # # neighbours
        self_idx, nbr_idx, _, nbr_dist = crystal.get_neighbor_list(
            self.radius,
            numerical_tol=1e-8,
        )

        if self.max_num_nbr is not None:
            _self_idx, _nbr_idx, _nbr_dist = [], [], []

            for _, g in groupby(zip(self_idx, nbr_idx, nbr_dist), key=lambda x: x[0]):
                s, n, d = zip(*sorted(g, key=lambda x: x[2]))
                _self_idx.extend(s[:self.max_num_nbr])
                _nbr_idx.extend(n[:self.max_num_nbr])
                _nbr_dist.extend(d[:self.max_num_nbr])

            self_idx = np.array(_self_idx)
            nbr_idx = np.array(_nbr_idx)
            nbr_dist = np.array(_nbr_dist)

        return self_idx, nbr_idx, nbr_dist

    def _pre_check(self):
        """Check that none of the structures have isolated atoms.

        Raises:
            ValueError: if isolated structures are present
        """
        print("checking all structures valid")
        all_iso = []
        some_iso = []

        for cif_id, crystal in zip(self.df["material_id"], self.df["Structure_obj"]):
            image = 0

            while image == 0:
                self_idx, nbr_idx, _ = self._get_nbr_data(crystal)

                if np.any(self_idx == nbr_idx):
                    # TODO only double along the shortest dimension
                    crystal.make_supercell([2, 2, 2])
                else:
                    image = 1

            if len(self_idx) == 0:
                all_iso.append(cif_id)
            elif len(nbr_idx) == 0:
                all_iso.append(cif_id)
            elif set(self_idx) != set(range(crystal.num_sites)):
                some_iso.append(cif_id)

        # TODO have the option for the pre-check to delete non-valid entries from the df?

        if (len(all_iso) > 0) or (len(some_iso) > 0):
            print(all_iso)
            print(some_iso)
            raise ValueError("isolated structures contained in dataframe")

    # NOTE do not cache the pre-training structures as we want to see new sets of
    # masked structures each epoch as this effectively expands the training set
    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # NOTE sites must be given in fractional co-ordinates
        df_idx = self.df.iloc[idx]
        crystal = df_idx["Structure_obj"]
        cif_id, comp = df_idx[self.identifiers]

        # atom features
        # TODO can this be vectorised with numpy?

        # handle disordered structures (multiple fractional elements per site)
        site_atoms = [atom.species.as_dict() for atom in crystal]
        atom_fea = np.vstack(
            [np.sum([self.ari.get_fea(el)*amt for el, amt in site.items()], axis=0) for site in site_atoms]
        )

        # select atleast one site in the crystal to mask
        mask_ids = np.sort(np.random.choice(
            np.arange(crystal.num_sites, dtype=int), max(1, int(self.p_mask * crystal.num_sites))
        ))

        # get the mask labels via use of a OHE of elements to handle disordered structures
        mask_labels = np.vstack(
            [np.sum([self.ohe.get_fea(el)*amt for el, amt in site_atoms[idx].items()], axis=0) for idx in mask_ids]
        )

        # TODO currently 20% no mask -> 10% no mask, 10% random mask
        mask_filter = np.random.rand(len(mask_ids))
        atom_fea[mask_ids[np.where(mask_filter < 0.8)], :] = 0

        # # # neighbours
        self_idx, nbr_idx, nbr_dist = self._get_nbr_data(crystal)

        assert len(self_idx), f"All atoms in {cif_id} are isolated"
        assert len(nbr_idx), f"This should not be triggered but was for {cif_id}"
        assert set(self_idx) == set(range(crystal.num_sites)), f"At least one atom in {cif_id} is isolated"

        nbr_dist = self.gdf.expand(nbr_dist)

        atom_fea = torch.Tensor(atom_fea)
        nbr_dist = torch.Tensor(nbr_dist)
        self_idx = torch.LongTensor(self_idx)
        nbr_idx = torch.LongTensor(nbr_idx)
        mask_ids = torch.LongTensor(mask_ids)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "mask":
                targets.append(torch.Tensor(mask_labels))
            elif self.task_dict[target] == "pretrain-global":
                raise NotImplementedError("bad user")

        return ((atom_fea, nbr_dist, self_idx, nbr_idx, mask_ids), targets, comp, cif_id)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): Variance of Gaussian basis. Defaults to step if not given
        """
        assert dmin < dmax
        assert dmax - dmin > step

        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)

        if var is None:
            var = step

        self.var = var

    def expand(self, distances):
        """Apply Gaussian distance filter to a numpy distance array

        Args:
            distances (ArrayLike): A distance matrix of any shape

        Returns:
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        distances = np.array(distances)

        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var ** 2
        )


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
        (atom_fea, nbr_dist, nbr_idx, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_dist: torch.Tensor shape (n_i, M, nbr_dist_len)
        nbr_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_dist: torch.Tensor shape (N, M, nbr_dist_len)
        Bond features of each atom's M neighbors
    batch_nbr_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea = []
    batch_nbr_dist = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_atom_idx = []
    batch_mask_idx = []
    batch_targets = []
    batch_comps = []
    batch_cif_ids = []
    base_idx = 0

    for i, (inputs, target, comp, cif_id) in enumerate(dataset_list):
        atom_fea, nbr_dist, self_idx, nbr_idx, mask_idx = inputs
        n_atoms = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_dist.append(nbr_dist)
        batch_self_idx.append(self_idx + base_idx)
        batch_nbr_idx.append(nbr_idx + base_idx)
        batch_mask_idx.append(mask_idx + base_idx)

        crystal_atom_idx.extend([i] * n_atoms)

        batch_targets.append(target)
        batch_comps.append(comp)
        batch_cif_ids.append(cif_id)

        base_idx += n_atoms

    atom_fea = torch.cat(batch_atom_fea, dim=0)
    nbr_dist = torch.cat(batch_nbr_dist, dim=0)
    self_idx = torch.cat(batch_self_idx, dim=0)
    nbr_idx = torch.cat(batch_nbr_idx, dim=0)
    mask_idx = torch.cat(batch_mask_idx, dim=0)

    cry_idx = torch.LongTensor(crystal_atom_idx)

    return (
        (atom_fea, nbr_dist, self_idx, nbr_idx, mask_idx, cry_idx),
        tuple(torch.cat(b_target, dim=0) for b_target in zip(*batch_targets)),
        batch_comps,
        batch_cif_ids,
    )


def get_structure(cols):
    """ Return pymatgen structure from lattice and sites cols """
    cell, sites = cols
    cell, elems, coords = parse_cgcnn(cell, sites)
    # NOTE getting primative structure before constructing graph
    # significantly harms the performnace of this model.

    crystal = Structure(
        lattice=cell, species=elems, coords=coords, to_unit_cell=True
    )

    # In place modification of structures that only contain a few sites
    # this is to allow us to mask ~15% of sites without having a
    # disproportionate impact on small unit cell structures.
    if crystal.num_sites < 7:
        crystal.make_supercell([2, 2, 2])

    return crystal


def parse_cgcnn(cell, sites):
    """ Parse str representation into lists """
    cell = np.array(ast.literal_eval(cell), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(sites):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords
