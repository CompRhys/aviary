import torch.nn as nn

from roost.core import BaseModelClass
from roost.cgcnn.model import CGCNNConv

from roost.segments import MeanPooling

num_elements_in_ptable = 118


class CrystalGraphPreNet(BaseModelClass):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.

    This model is based on: https://github.com/txie-93/cgcnn [MIT License].
    Changes to the code were made to allow for the removal of zero-padding
    and to benefit from the BaseModelClass functionality. The architectural
    choices of the model remain unchanged.
    """

    def __init__(
        self,
        robust,
        n_targets,
        elem_emb_len,
        nbr_fea_len,
        elem_fea_len=64,
        n_graph=4,
        h_fea_len=128,
        n_hidden=1,
        **kwargs,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_elem_fea_len: int
            Number of atom features in the input.
        nbr_fea_len: int
            Number of bond features.
        elem_fea_len: int
            Number of hidden atom features in the convolutional layers
        n_graph: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_hidden: int
            Number of hidden layers after pooling
        """
        super().__init__(robust=robust, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
        }

        self.material_nn = NodeNetwork(**desc_dict)

        self.model_params.update(
            {
                "robust": robust,
                "n_targets": n_targets,
                "h_fea_len": h_fea_len,
                "n_hidden": n_hidden,
            }
        )

        self.node_linear = nn.Linear(elem_fea_len, num_elements_in_ptable)
        # self.global_pooling = MeanPooling()
        # print(f"{n_targets=}")
        # self.global_linear = nn.Linear(elem_fea_len, n_targets)
        self.model_params.update(desc_dict)

    def forward(self, atom_fea, nbr_fea, self_idx, nbr_idx, mask_idx, cry_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution

        """
        crys_fea = self.material_nn(atom_fea, nbr_fea, self_idx, nbr_idx)

        nodes = self.node_linear(crys_fea)

        return nodes


class NodeNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    CrystalGraphConvNet Model.
    """

    def __init__(
        self, elem_emb_len, nbr_fea_len, elem_fea_len=64, n_graph=4,
    ):
        """
        """
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            [CGCNNConv(
                elem_fea_len=elem_fea_len,
                nbr_fea_len=nbr_fea_len
            ) for _ in range(n_graph)]
        )

    def forward(self, atom_fea, nbr_fea, self_idx, nbr_idx):
        """Forward pass

        Args:
            atom_fea (Tensor): Atom features from atom type
            nbr_fea (Tensor): Bond features of each atom's M neighbors
            self_idx (LongTensor): Indices of M neighbors of each atom
            nbr_idx (LongTensor): Indices of M neighbors of each atom

        Returns:
            Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_idx, nbr_idx)

        return atom_fea
