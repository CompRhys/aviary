import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_max, scatter_add, \
                          scatter_mean


class MessageLayer(nn.Module):
    """
    Class defining the message passing operation on the composition graph
    """
    def __init__(self, fea_len, num_heads=1):
        """
        Inputs
        ----------
        fea_len: int
            Number of elem hidden features.
        """
        super(MessageLayer, self).__init__()

        # Pooling and Output
        hidden_ele = [256]
        hidden_msg = [256]
        self.pooling = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(2*fea_len, 1, hidden_ele),
            message_nn=SimpleNetwork(2*fea_len, fea_len, hidden_msg),
            # message_nn=nn.Linear(2*fea_len, fea_len),
            # message_nn=nn.Identity(),
            ) for _ in range(num_heads)])

    def forward(self, elem_weights, elem_in_fea,
                self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elems (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each elem
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of M neighbours of each elem

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Atom hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(fea=fea,
                                     index=self_fea_idx,
                                     weights=elem_nbr_weights))

        # # Concatenate
        # return torch.cat(head_fea, dim=1)

        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

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
    def __init__(self, orig_elem_fea_len, elem_fea_len, n_graph):
        """
        Initialize CompositionNet.

        Parameters
        ----------
        n_h: Number of hidden layers after pooling

        Inputs
        ----------
        orig_elem_fea_len: int
            Number of elem features in the input.
        elem_fea_len: int
            Number of hidden elem features in the graph layers
        n_graph: int
            Number of graph layers
        """
        super(CompositionNet, self).__init__()

        # apply linear transform to the input to get a trainable embedding
        self.embedding = nn.Linear(orig_elem_fea_len, elem_fea_len-1, bias=False)

        # create a list of Message passing layers

        msg_heads = 3
        self.graphs = nn.ModuleList(
                        [MessageLayer(elem_fea_len, msg_heads)
                            for i in range(n_graph)])

        # # Concatenate
        # self.graphs = nn.ModuleList(
        #       [MessageLayer(elem_fea_len * (msg_heads ** i), msg_heads)
        #                     for i in range(n_graph)])
        # elem_fea_len = elem_fea_len * (msg_heads ** msg_heads)

        # define a global pooling function for materials
        mat_heads = 3
        mat_hidden = [256]
        msg_hidden = [256]
        self.cry_pool = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(elem_fea_len, 1, mat_hidden),
            message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, msg_hidden),
            # message_nn=nn.Linear(elem_fea_len, elem_fea_len),
            # message_nn=nn.Identity(),
            ) for _ in range(mat_heads)])

        # define an output neural network
        out_hidden = [1024, 512, 256, 128, 64, 32]
        # out_hidden = [x * elem_fea_len for x in [7, 5, 3, 1]]
        self.output_nn = ResidualNetwork(elem_fea_len, 2, out_hidden)

    def forward(self, elem_weights, orig_elem_fea, self_fea_idx,
                nbr_fea_idx, crystal_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elems (nodes) in the batch
        M: Total number of bonds (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        orig_elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Atom features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the elem each of the M bonds correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of of the neighbours of the M bonds connect to
        elem_bond_idx: list of torch.LongTensor of length C
            Mapping from the bond idx to elem idx
        crystal_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        out: nn.Variable shape (C,)
            Atom hidden features after message passing
        """

        # embed the original features into the graph layer description
        elem_fea = self.embedding(orig_elem_fea)

        # do this so that we can examine the embeddings without
        # influence of the weights
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the graph message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea,
                                  self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(attnhead(fea=elem_fea,
                                     index=crystal_elem_idx,
                                     weights=elem_weights))

        crys_fea = torch.mean(torch.stack(head_fea), dim=0)

        # apply neural network to map from learned features to target
        crys_fea = self.output_nn(crys_fea)

        return crys_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedMeanPooling(torch.nn.Module):
    """
    mean pooling
    """
    def __init__(self):
        super(WeightedMeanPooling, self).__init__()

    def forward(self, fea, index, weights):
        fea = weights * fea
        return scatter_mean(fea, index, dim=0)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedAttention(nn.Module):
    """
    Weighted softmax attention layer
    """
    def __init__(self, gate_nn, message_nn, num_heads=1):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(WeightedAttention, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, fea, index, weights):
        """ forward pass """

        gate = self.gate_nn(fea)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-13)

        fea = self.message_nn(fea)
        out = scatter_add(gate * fea, index, dim=0)

        return out

    def __repr__(self):
        return '{}(gate_nn={})'.format(self.__class__.__name__,
                                       self.gate_nn)


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(SimpleNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
        #                           for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
