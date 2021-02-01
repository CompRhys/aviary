import ast
import functools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset

from roost.core import Featurizer


class CrystalGraphData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    Parameters
    ----------

    data_path: str
            The path to the dataset
    fea_path: str
            The path to the element embedding
    max_num_nbr: int
            The maximum number of neighbors while constructing the crystal graph
    radius: float
            The cutoff radius for searching neighbors
    dmin: float
            The minimum distance for constructing GaussianDistance
    step: float
            The step size for constructing GaussianDistance

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    self_fea_idx: torch.LongTensor shape (n_i, M)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    comp: str
    cif_id: str or int
    """

    def __init__(
        self,
        data_path,
        fea_path,
        task_dict,
        inputs=["lattice","sites"],
        identifiers=["material_id", "composition"],
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        use_cache=True,
    ):

        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 2, "One input column required are required"

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = identifiers

        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE this naming structure might lead to clashes where the model
        # loads the wrong graph from the cache.
        self.use_cache = use_cache
        if self.use_cache:
            self.cachedir = os.path.join(
                os.path.dirname(data_path),
                f"cache-n{max_num_nbr}-r{radius}-d{dmin}-s{step}/"
            )
            if not os.path.isdir(self.cachedir):
                os.makedirs(self.cachedir)

        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.ari = Featurizer.from_json(fea_path)
        self.elem_fea_dim = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_dim = self.gdf.embedding_size

        self.n_targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                self.n_targets.append(1)
            elif self.task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self):
        # return len(self.id_prop_data)
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # NOTE sites must be given in fractional co-ordinates
        df_idx = self.df.iloc[idx]
        cell, sites = df_idx[self.inputs]
        cif_id, comp = df_idx[self.identifiers]

        if self.use_cache:
            # hashdir = os.path.join(self.cachedir, str(abs(hash(cif_id) % 100)))
            # os.makedirs(hashdir, exist_ok=True)
            cache_path = os.path.join(self.cachedir, f"{cif_id}.pkl")

        if self.use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                try:
                    pkl_data = pickle.load(f)
                except EOFError:
                    raise EOFError(f"Check {f} for issue")
            atom_fea = pkl_data[0]
            nbr_fea = pkl_data[1]
            self_fea_idx = pkl_data[2]
            nbr_fea_idx = pkl_data[3]

        else:
            cell, elems, coords = parse_cgcnn(cell, sites)
            # NOTE getting primative structure before constructing graph
            # significantly harms the performnace of this model.
            crystal = Structure(
                lattice=cell, species=elems, coords=coords, to_unit_cell=True
            )

            # atom features
            atom_fea = [atom.specie.symbol for atom in crystal]

            # neighbours
            all_nbrs = crystal.get_all_neighbors(
                self.radius,
                include_index=True,
                numerical_tol=1e-8
            )
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            self_fea_idx, nbr_fea_idx, nbr_fea = [], [], []

            for i, nbr in enumerate(all_nbrs):
                # NOTE due to using a geometric learning library we do not
                # need to set a maximum number of neighbours but do so in
                # order to replicate the original code.
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.extend(list(map(lambda x: x[2], nbr)))
                    nbr_fea.extend(list(map(lambda x: x[1], nbr)))
                else:
                    nbr_fea_idx.extend(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                    nbr_fea.extend(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))

                if len(nbr) == 0:
                    raise ValueError(
                        f"Isolated atom found in {cif_id} ({comp}) - "
                        "increase maximum radius or remove structure"
                    )
                self_fea_idx.extend([i] * min(len(nbr), self.max_num_nbr))

            if self.use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump((atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), f)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(torch.Tensor([df_idx[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx),
            targets,
            comp,
            cif_id
        )


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step

        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)

        if var is None:
            var = step

        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
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
        (atom_fea, nbr_fea, nbr_fea_idx, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_comps = []
    batch_cif_ids = []
    base_idx = 0

    for i, (inputs, target, comp, cif_id) in enumerate(dataset_list):
        atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx = inputs
        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_self_fea_idx.append(self_fea_idx + base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)

        crystal_atom_idx.extend([i] * n_i)
        batch_targets.append(target)
        batch_comps.append(comp)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.LongTensor(crystal_atom_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        batch_comps,
        batch_cif_ids,
    )


def parse_cgcnn(cell, sites):
    """
    """
    cell = np.array(ast.literal_eval(cell), dtype=float)
    elems = []
    coords = []
    for site in ast.literal_eval(sites):
        ele, pos = site.split(" @ ")
        elems.append(ele)
        coords.append(pos.split(" "))

    coords = np.array(coords, dtype=float)
    return cell, elems, coords
