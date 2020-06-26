import torch
import torch.nn as nn
from roost.core import BaseModelClass
from roost.segments import MeanPooling, SumPooling


class CGCNN(BaseModelClass):
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
        task,
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
        super(CGCNN, self).__init__(
            task=task, robust=robust, n_targets=n_targets, **kwargs
        )

        self.model_params = {
            "task": task,
            "robust": robust,
            "n_targets": n_targets,
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "h_fea_len": h_fea_len,
            "n_hidden": n_hidden,
        }

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            [ConvLayer(
                elem_fea_len=elem_fea_len,
                nbr_fea_len=nbr_fea_len
            ) for _ in range(n_graph)]
        )

        self.pooling = MeanPooling()
        self.conv_to_fc = nn.Linear(elem_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_hidden > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_hidden - 1)]
            )
            self.softpluses = nn.ModuleList(
                [nn.Softplus() for _ in range(n_hidden - 1)]
            )

        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets

        self.fc_out = nn.Linear(h_fea_len, output_dim)

    def forward(self, atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx):
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
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)

        return out


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, elem_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        elem_fea_len: int
                Number of atom hidden features.
        nbr_fea_len: int
                Number of bond features.
        """
        super(ConvLayer, self).__init__()
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

    def forward(self, atom_in_fea, nbr_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, elem_fea_len)
            Atom hidden features after convolution

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
