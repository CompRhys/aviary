from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pymatgen.util.due import Doi, due
from torch import LongTensor, Tensor, nn

from aviary.cgcnn.data import GaussianDistance
from aviary.core import BaseModelClass
from aviary.networks import SimpleNetwork
from aviary.scatter import scatter_reduce
from aviary.utils import get_element_embedding


@due.dcite(Doi("10.1103/PhysRevLett.120.145301"), description="CGCNN model")
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
        n_targets: Sequence[int],
        elem_embedding: str = "cgcnn92",
        radius_cutoff: float = 5.0,
        radius_min: float = 0.0,
        radius_step: float = 0.2,
        elem_fea_len: int = 64,
        n_graph: int = 4,
        h_fea_len: int = 128,
        n_trunk: int = 1,
        n_hidden: int = 1,
        **kwargs,
    ) -> None:
        """Initialize CrystalGraphConvNet.

        Args:
            robust (bool): If True, the number of model outputs is doubled. 2nd output
                for each target will be an estimate for the aleatoric uncertainty
                (uncertainty inherent to the sample) which can be used with a robust
                loss function to attenuate the weighting of uncertain samples.
            n_targets (list[int]): Number of targets to train on
            elem_embedding (str, optional): One of matscholar200, cgcnn92, megnet16,
                onehot112 or path to a file with custom element embeddings.
                Defaults to matscholar200.
            radius_cutoff (float, optional): Cut-off radius for neighborhood.
                Defaults to 5.
            radius_min (float, optional): minimum distance in Gaussian basis.
                Defaults to 0.
            radius_step (float, optional): increment size of Gaussian basis.
                Defaults to 0.2.
            elem_fea_len (int, optional): Number of hidden atom features in the
                convolutional layers. Defaults to 64.
            n_graph (int, optional): Number of convolutional layers. Defaults to 4.
            h_fea_len (int, optional): Number of hidden features after pooling. Defaults
                to 128.
            n_trunk (int, optional): Number of hidden layers in trunk after pooling.
                Defaults to 1.
            n_hidden (int, optional): Number of hidden layers after trunk for each task.
                Defaults to 1.
            **kwargs: Additional keyword arguments to pass to BaseModelClass.
        """
        super().__init__(robust=robust, **kwargs)

        self.elem_embedding = get_element_embedding(elem_embedding)
        elem_emb_len = self.elem_embedding.weight.shape[1]

        self.gaussian_dist_func = GaussianDistance(
            dmin=radius_min, dmax=radius_cutoff, step=radius_step
        )
        nbr_fea_len = self.gaussian_dist_func.embedding_size

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
        }

        self.node_nn = DescriptorNetwork(**desc_dict)

        model_params = {
            "robust": robust,
            "n_targets": n_targets,
            "h_fea_len": h_fea_len,
            "n_hidden": n_hidden,
            **desc_dict,
        }
        self.model_params.update(model_params)

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
    ) -> tuple[Tensor, ...]:
        """Forward pass.

        Args:
            atom_fea (Tensor): Atom features from atom type
            nbr_fea (Tensor): Bond features of each atom's neighbors
            self_idx (LongTensor): Mapping of Tensor rows to each nodes
            nbr_idx (LongTensor): Indices of the neighbors of each atom
            crystal_atom_idx (LongTensor): Mapping from the crystal idx to atom idx

        Returns:
            tuple[Tensor, ...]: tuple of predictions for all targets
        """
        nbr_fea = self.gaussian_dist_func.expand(nbr_fea)
        atom_fea = self.elem_embedding(atom_fea)

        atom_fea = self.node_nn(atom_fea, nbr_fea, self_idx, nbr_idx)

        crys_fea = scatter_reduce(atom_fea, crystal_atom_idx, dim=0, reduce="mean")

        # NOTE required to match the reference implementation
        crys_fea = nn.functional.softplus(crys_fea)

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the CrystalGraphConvNet
    Model.
    """

    def __init__(
        self,
        elem_emb_len: int,
        nbr_fea_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 4,
    ) -> None:
        """Initialize DescriptorNetwork.

        Args:
            elem_emb_len (int): Number of atom features in the input.
            nbr_fea_len (int): Number of bond features.
            elem_fea_len (int, optional): Number of hidden atom features in the graph
                convolution layers. Defaults to 64.
            n_graph (int, optional): Number of graph convolution layers. Defaults to 4.
        """
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            CGCNNConv(elem_fea_len=elem_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_graph)
        )

    def forward(
        self,
        atom_fea: Tensor,
        nbr_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            atom_fea (Tensor): Atom features from atom type
            nbr_fea (Tensor): Bond features of each atom's M neighbors
            self_idx (LongTensor): Mapping from the crystal idx to atom idx
            nbr_idx (LongTensor): Indices of M neighbors of each atom

        Returns:
            Tensor: Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_idx, nbr_idx)

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

    def forward(
        self,
        atom_in_fea: Tensor,
        nbr_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            atom_in_fea (Tensor): Atom hidden features before convolution
            nbr_fea (Tensor): Bond features of each atom's neighbors
            self_idx (LongTensor): Indices of the atom's self
            nbr_idx (LongTensor): Indices of M neighbors of each atom

        Returns:
            Tensor: Atom hidden features after convolution
        """
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_idx, :]
        atom_self_fea = atom_in_fea[self_idx, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.softplus1(core_fea)

        # take the elementwise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_summed = scatter_reduce(nbr_msg, self_idx, dim=0, reduce="sum")

        nbr_summed = self.bn2(nbr_summed)
        return self.softplus2(atom_in_fea + nbr_summed)
