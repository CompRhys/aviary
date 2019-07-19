import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, \
                            scatter_add, scatter_mul
from itertools import zip_longest

class MessageLayer(nn.Module):
    """
    Class defining the message passing operation on the composition graph
    """
    def __init__(self, atom_fea_len, atom_gate):
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
        
        # Message Passing
        self.linear_in = nn.Linear(2*self.atom_fea_len, 4*self.atom_fea_len)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(4*self.atom_fea_len, self.atom_fea_len)

        # Pooling and Output
        self.pooling = WeightedAttention(atom_gate)
        self.pool_act = nn.ReLU()

    def forward(self, atom_weights, atom_in_fea,
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
        atom_nbr_weights = atom_weights[nbr_fea_idx,:]
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx,:]
        fea = torch.cat([atom_self_fea, atom_nbr_fea], dim=1)

        # pass messages between the pairs of atoms
        fea = self.linear_in(fea)
        fea = self.act(fea)
        fea = self.linear_out(fea)

        # sum selectivity over the neighbours to get atoms
        fea = self.pooling(fea, self_fea_idx, atom_nbr_weights)
        fea = self.pool_act(fea)
        fea = atom_in_fea + fea

        return fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

      
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
    def __init__(self, orig_atom_fea_len,
                 atom_fea_len, n_graph):
        """
        Initialize CompositionNet.

        Parameters
        ----------
        n_h: Number of hidden layers after pooling

        Inputs
        ----------
        orig_atom_fea_len: int
            Number of atom features in the input.
        atom_fea_len: int
            Number of hidden atom features in the graph layers
        n_graph: int
            Number of graph layers
        """
        super(CompositionNet, self).__init__()

        # apply linear transform to the input features to get a trainable embedding
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # define the necessary neural networks for the pooling
        # create a list of Message passing layers
        hidden = [x * atom_fea_len for x in [3,1]]
        self.graphs = nn.ModuleList(
                            [MessageLayer(atom_fea_len=atom_fea_len,
                            atom_gate=PyramidNetwork(atom_fea_len, 1, hidden))
                            for _ in range(n_graph)]
                        )

        hidden = [x * atom_fea_len for x in [5,3,1]]
        self.cry_pool = WeightedAttention(
                            gate_nn = PyramidNetwork(atom_fea_len, 1, hidden)
                        )

        hidden = [x * atom_fea_len for x in [7,5,3,1]]
        self.output_nn = PyramidNetwork(atom_fea_len, 2, hidden)

        


    def forward(self, atom_weights, orig_atom_fea, self_fea_idx, 
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
            atom_fea = graph_func(atom_weights, atom_fea, 
                                    self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the atomic features
        crys_fea = self.cry_pool(atom_fea, crystal_atom_idx, 
                                atom_weights)

        crys_fea = self.output_nn(crys_fea)

        return crys_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedAttention(nn.Module):
    """  
    Weighted softmax attention layer  
    """
    def __init__(self, gate_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(WeightedAttention, self).__init__()
        self.gate_nn = gate_nn

    def forward(self, x, index, weights):
        """ forward pass """
        # x = x.unsqueeze(-1) if x.dim() == 1 else x

        gate = self.gate_nn(x)#.view(-1,1)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = weights * gate.exp() 
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-13)

        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return '{}(gate_nn={})'.format(self.__class__.__name__,
                                              self.gate_nn)


class PyramidNetwork(nn.Module):
    """
    make a pyramid network
    """
    def __init__(self, input_dim, output_dim, 
                hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(PyramidNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims+[output_dim]

        nodes = [nn.Linear(dims[i],dims[i+1]) for i in range(len(dims)-1)]
        acts = [nn.ReLU() for _ in range(len(dims)-2)]
        modules = [y for x in zip_longest(nodes,acts, fillvalue=None) 
                    for y in x if y is not None]
        modules = nn.ModuleList(modules)  
        self.network = nn.Sequential(*modules)

    def forward(self, fea):
        return self.network(fea)
