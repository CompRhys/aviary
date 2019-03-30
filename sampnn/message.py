import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class MessageLayer(nn.Module):
    """
    Class defining the message passing operation on the composition graph
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Inputs
        ----------
        atom_fea_len: int
            Number of atom hidden features.
        nbr_fea_len: int
            Number of bond features.
        """
        super(MessageLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        self.filter_msg = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, 
                                    self.atom_fea_len)

        self.core_msg = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, 
                                    self.atom_fea_len)

        self.bn_filter = nn.BatchNorm1d(self.atom_fea_len)
        self.bn_core = nn.BatchNorm1d(self.atom_fea_len)
        self.bn_output = nn.BatchNorm1d(self.atom_fea_len)

        self.filter_transform = nn.Sigmoid()
        self.core_transform = nn.Softplus()
        self.output_transform = nn.Softplus()

    def forward(self, atom_in_fea, bond_nbr_fea, 
                self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of atoms (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
            Atom hidden features before message passing
        bond_nbr_fea: Variable(torch.Tensor) shape (M, nbr_fea_len)
            Bond features of atom's neighbours
        self_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each atom
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each atom
        atom_bond_idx: list of torch.Tensor of length N
            mapping from the atom idx to bond idx

        Returns
        -------
        atom_out_fea: nn.Variable shape (N, atom_fea_len)
            Atom hidden features after message passing
        """
        # construct the total features for passing
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx,:]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, bond_nbr_fea], dim=1)

        # multiply input by weight matrix
        filter_fea = self.filter_msg(total_fea)
        core_fea = self.core_msg(total_fea)

        # apply batch-normalisation (regularise and reduce covariate shift)
        filter_fea = self.bn_filter(filter_fea)
        core_fea = self.bn_core(core_fea)

        # apply non-linear transformations
        filter_fea = self.filter_transform(filter_fea)
        core_fea = self.core_transform(core_fea)

        # take the elementwise product of the filter and core
        nbr_message = filter_fea * core_fea

        # sum selectivity over the neighbours to get atoms
        nbr_sumed = scatter_mean(nbr_message, self_fea_idx, dim=0)

        nbr_sumed = self.bn_output(nbr_sumed)

        atom_out_fea = self.output_transform(atom_in_fea + nbr_sumed)
        return atom_out_fea

      
class CompositionNet(nn.Module):
    """
    Create a neural network for predicting total material properties.

    The CompositionNet model is comprised of a fully connected network
    and message passing graph layers.

    The message passing layers are used to determine a descriptor set
    for the fully connected network. Critically the graphs are used to 
    represent (crystalline) materials in a structure agnostic manner 
    but contain trainable parameters unlike other structure agnostic
    approaches.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=48, n_graph=3, h_fea_list=[128], 
                 n_out = 1):
        """
        Initialize CompositionNet.

        Parameters
        ----------
        n_h: Number of hidden layers after pooling

        Inputs
        ----------
        orig_atom_fea_len: int
            Number of atom features in the input.
        nbr_fea_len: int
            Number of bond features.
        atom_fea_len: int
            Number of hidden atom features in the graph layers
        n_graph: int
            Number of graph layers
        h_fea_list: list int of length n_h
            Number of hidden features in each fc layer after pooling
        n_out: int
            Number of outputs for the fc network
        """
        super(CompositionNet, self).__init__()

        # apply linear transform to the input features to get a trainable embedding
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList([MessageLayer(atom_fea_len=atom_fea_len,
                                                 nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_graph)])

        self.graph_to_fc = nn.Linear(atom_fea_len, h_fea_list[0])
        self.graph_to_fc_bn = nn.BatchNorm1d(h_fea_list[0])
        self.graph_to_fc_softplus = nn.Softplus()

        if len(h_fea_list) > 1:
            # create a list of fully connected passing layers
            self.weightings = nn.ModuleList([nn.Linear(h_fea_list[i], h_fea_list[i+1])
                                        for i in range(len(h_fea_list)-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                        for i in range(len(h_fea_list)-1)])
            self.batchnorms = nn.ModuleList([nn.BatchNorm1d(h_fea_list[i+1])
                                        for i in range(len(h_fea_list)-1)])

        self.fc_out = nn.Linear(h_fea_list[-1], n_out)

    def forward(self, orig_atom_fea, nbr_fea, self_fea_idx, 
                nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of atoms (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        orig_atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
            Atom features of each of the N atoms in the batch
        nbr_fea: Variable(torch.Tensor) shape (M, nbr_fea_len)
            Bond features of each M bonds in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the atom each of the M bonds correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of of the neighbours of the M bonds connect to
        atom_bond_idx: list of torch.LongTensor of length C
            Mapping from the bond idx to atom idx
        crystal_atom_idx: list of torch.LongTensor of length C
            Mapping from the atom idx to crystal idx
        
        Returns
        -------
        out: nn.Variable shape (C,)
            Atom hidden features after message passing
        """

        # embed the original features into the graph layer description
        atom_fea = self.embedding(orig_atom_fea)

        # apply the graph message passing functions 
        for graph_func in self.graphs:
            atom_fea = graph_func(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the atomic features
        crys_fea = scatter_mean(atom_fea, crystal_atom_idx, dim=0)
        crys_fea = self.graph_to_fc_softplus(crys_fea)

        # prepate the crystal features for the full connected neural network
        crys_fea = self.graph_to_fc(crys_fea)
        crys_fea = self.graph_to_fc_bn(crys_fea)
        crys_fea = self.graph_to_fc_softplus(crys_fea)

        if hasattr(self, 'weightings') and hasattr(self, 'softpluses'):
            # join together the linear layers and non-linear activation functions
            # and apply recursively to build up the fully connected neural network
            for wm, bn, sp in zip(self.weightings, self.batchnorms, self.softpluses):
                crys_fea = sp(bn(wm(crys_fea)))

        out = self.fc_out(crys_fea)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        Parameters
        ----------
        N: Total number of atoms (nodes) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom feature vectors of the batch
        crystal_atom_idx: list of torch.Tensor of length C
            Mapping from the crystal idx to atom idx

        Return
        ----------
        pooled: torch.Tensor shape (C, 2*atom_fea_len)
            crystal feature vectors for the batch
        
        """

        # check that the sum of all the groups of atoms corresponding
        # to different crystals is equal to the total number of atoms
        assert crystal_atom_idx[-1,-1] == atom_fea.data.shape[0]
        
        # Pool to get the mean atomic features
        mean_fea = [torch.mean(atom_fea[idx[0]:idx[1]], dim=0, keepdim=True)
                    for idx in crystal_atom_idx]
        mean_fea = torch.cat(mean_fea, dim=0)

        return mean_fea