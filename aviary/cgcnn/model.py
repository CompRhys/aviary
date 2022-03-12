from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from aviary.core import BaseModelClass
from aviary.segments import MeanPooling, SimpleNetwork, SumPooling


class CrystalGraphConvNet(BaseModelClass):
    """Create a crystal graph convolutional neural network for predicting total
    material properties.

    This model is based on: https://github.com/txie-93/cgcnn [MIT License].
    Changes to the code were made to allow for the removal of zero-padding
    and to benefit from the BaseModelClass functionality. The architectural
    choices of the model remain unchanged.
    """

    def __init__(
        self,
        robust: bool,
        n_targets: list[int],
        elem_emb_len: int,
        nbr_fea_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 4,
        h_fea_len: int = 128,
        n_trunk: int = 1,
        n_hidden: int = 1,
        **kwargs,
    ) -> None:
        """Initialize CrystalGraphConvNet.

        Args:
            robust (bool): _description_
            n_targets (list[int]): _description_
            elem_emb_len (int): Number of atom features in the input.
            nbr_fea_len (int): Number of bond features.
            elem_fea_len (int, optional): Number of hidden atom features in the convolutional layers. Defaults to 64.
            n_graph (int, optional): Number of convolutional layers. Defaults to 4.
            h_fea_len (int, optional): Number of hidden features after pooling. Defaults to 128.
            n_trunk (int, optional): _description_. Defaults to 1.
            n_hidden (int, optional): Number of hidden layers after pooling. Defaults to 1.
        """
        super().__init__(robust=robust, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
        }

        self.node_nn = DescriptorNetwork(**desc_dict)

        self.model_params.update(
            {
                "robust": robust,
                "n_targets": n_targets,
                "h_fea_len": h_fea_len,
                "n_hidden": n_hidden,
            }
        )

        self.model_params.update(desc_dict)

        self.pooling = MeanPooling()

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        out_hidden = [h_fea_len] * n_hidden
        trunk_hidden = [h_fea_len] * n_trunk
        self.trunk_nn = SimpleNetwork(elem_fea_len, h_fea_len, trunk_hidden)

        self.output_nns = nn.ModuleList(
            SimpleNetwork(h_fea_len, n, out_hidden) for n in n_targets
        )

    def forward(
        self,
        atom_fea: Tensor,
        nbr_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        crystal_atom_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            atom_fea (Tensor): shape (N, orig_elem_fea_len) Atom features from atom type
            nbr_fea (Tensor): shape (N, M, nbr_fea_len) Bond features of each atom's M neighbors
            self_idx (LongTensor): _description_
            nbr_idx (LongTensor): shape (N, M) Indices of M neighbors of each atom
            crystal_atom_idx (LongTensor): of length N0 mapping from the crystal idx to atom idx

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Returns:
            Tensor: shape (N,) Atom hidden features after convolution
        """
        atom_fea = self.node_nn(atom_fea, nbr_fea, self_idx, nbr_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # NOTE required to match the reference implementation
        crys_fea = nn.functional.softplus(crys_fea)

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return (output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the CrystalGraphConvNet Model."""

    def __init__(
        self,
        elem_emb_len: int,
        nbr_fea_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 4,
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            [
                CGCNNConv(elem_fea_len=elem_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_graph)
            ]
        )

    def forward(
        self,
        atom_fea: Tensor,
        nbr_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            atom_fea (Tensor): shape (N, orig_elem_fea_len) Atom features from atom type
            nbr_fea (Tensor): shape (N, M, nbr_fea_len) Bond features of each atom's M neighbors
            self_fea_idx (LongTensor): of length N0 Mapping from the crystal idx to atom idx
            nbr_fea_idx (LongTensor): shape (N, M) Indices of M neighbors of each atom

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Returns:
            Tensor: shape (N, ) Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        return atom_fea


class CGCNNConv(nn.Module):
    """Convolutional operation on graphs."""

    def __init__(self, elem_fea_len: int, nbr_fea_len: int) -> None:
        """Initialize CGCNNConv.

        Args:
            elem_fea_len (int): Number of atom hidden features.
            nbr_fea_len (int): Number of bond features.
        """
        super().__init__()
        self.elem_fea_len = elem_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.elem_fea_len + self.nbr_fea_len, 2 * self.elem_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.elem_fea_len)
        self.bn2 = nn.BatchNorm1d(self.elem_fea_len)
        self.softplus2 = nn.Softplus()
        self.pooling = SumPooling()

    def forward(
        self,
        atom_in_fea: Tensor,
        nbr_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            atom_in_fea (Tensor): shape (N, elem_fea_len) Atom hidden features before convolution
            nbr_fea (Tensor): shape (N, M, nbr_fea_len) Bond features of each atom's M neighbors
            self_fea_idx (LongTensor): _description_
            nbr_fea_idx (LongTensor): shape (N, M) Indices of M neighbors of each atom

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Returns:
            Tensor: shape (N, elem_fea_len) Atom hidden features after convolution
        """
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.softplus1(core_fea)

        # take the elementwise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_sumed = self.pooling(nbr_msg, self_fea_idx)

        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)

        return out
