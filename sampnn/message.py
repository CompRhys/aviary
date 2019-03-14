import torch
import torch.nn as nn

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
        
        self.pass_msg = nn.Linear(  2*self.atom_fea_len+self.nbr_fea_len, 
                                    2*self.atom_fea_len)

        self.batchnorm2 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.batchnorm1 = nn.BatchNorm1d(self.atom_fea_len)

        self.filter_transform = nn.Sigmoid()
        self.core_transform = nn.Softplus()
        self.output_transform = nn.Softplus()

    def forward(self, atom_in_fea, bond_nbr_fea, self_fea_idx, nbr_fea_idx, atom_bond_idx):
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
        atom_self_fea = atom_in_fea.[self_fea_idx,:]
        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, bond_nbr_fea], dim=1)

        # define the message passing operation
        total_fea = self.pass_msg(total_fea)
        total_fea = self.batchnorm2(total_fea)
        
        # separate out into the sigmoid and softplus sets
        nbr_filter, nbr_core = total_fea.chunk(2, dim=1)

        # apply non-linear transformations
        nbr_filter = self.filter_transform(nbr_filter)
        nbr_core = self.core_transform(nbr_core)

        # take the elementwise product of the filter and core
        nrb_message = nbr_filter * nbr_core

        # sum selectivity over the neighbours to
        nbr_sumed = [torch.sum(atom_fea[idx_map], dim=0, keepdim=True)
                        for idx_map in atom_bond_idx]
        nbr_sumed = torch.cat(nbr_sumed, dim=0)
        nbr_sumed = self.batchnorm1(nbr_sumed)

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
                 atom_fea_len=64, n_graph=3, h_fea_len=[128], n_h=1):
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

        # note 2*atom_fea_len due to including the mean and std of atom features
        self.graph_to_fc = nn.Linear(2*atom_fea_len, h_fea_list[0])
        self.graph_to_fc_softplus = nn.Softplus()

        if len(h_fea_list) > 1:
            # create a list of fully connected passing layers
            self.fcs = nn.ModuleList([nn.Linear(h_fea_list[i], h_fea_list[i+1])
                                      for i in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for i in range(n_h-1)])

        self.fc_out = nn.Linear(h_fea_list[-1], n_out)

    def forward(self, orig_atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
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
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (M, nbr_fea_len)
            Bond features of each atom's M neighbours
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each atom
        crystal_atom_idx: list of torch.LongTensor of length C
            Mapping from the crystal idx to atom idx
        
        Returns
        -------
        out: nn.Variable shape (C,)
            Atom hidden features after message passing
        """

        # embed the original features into the graph layer description
        atom_fea = self.embedding(orig_atom_fea)

        # apply the graph message passing functions 
        for graph_func in self.graphs:
            atom_fea = graph_func(atom_fea, nbr_fea, nbr_fea_idx)

        # generate crystal features by pooling the atomic features
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.graph_to_fc_softplus(crys_fea)

        # prepate the crystal features for the full connected neural network
        crys_fea = self.graph_to_fc(crys_fea)
        crys_fea = self.graph_to_fc_softplus(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            # join together the linear layers and non-linear activation functions
            # and apply recursively to build up the fully connected neural network
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

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
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        
        # Pool to get the mean atomic features
        mean_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                    for idx_map in crystal_atom_idx]
        mean_fea = torch.cat(mean_fea, dim=0)

        # Pool to get the standard deviation of the atomic features
        std_fea = [torch.std(atom_fea[idx_map], dim=0, keepdim=True)
                    for idx_map in crystal_atom_idx]
        std_fea = torch.cat(std_fea, dim=0)

        # concatenate the pooled means and standard deviations.
        pooled = torch.cat((mean_fea, std_fea), dim=1)
        return pooled