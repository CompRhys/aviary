import torch
import torch.nn as nn
from roost.utils import BaseModelClass
from torch_scatter import scatter_max, scatter_add, scatter_mean


class Roost(BaseModelClass):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.

    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stiochiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        task,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        out_hidden=[1024, 512, 256, 128, 64],
        **kwargs
    ):
        super(Roost, self).__init__(
            task=task, robust=robust, n_targets=n_targets, **kwargs
        )

        # TODO find a more elegant way to structure this unpacking then
        # a dictionary seems like a non-optimal solution.
        self.model_params.update()

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn = DescriptorNetwork(**desc_dict)

        self.model_params.update({
            "task": task,
            "robust": robust,
            "n_targets": n_targets,
            "out_hidden": out_hidden,
        })

        self.model_params.update(desc_dict)

        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets

        self.output_nn = ResidualNetwork(elem_fea_len, output_dim, out_hidden)

    def forward(
        self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
    ):
        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        )

        # apply neural network to map from learned features to target
        return self.output_nn(crys_fea)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
        """
        super(DescriptorNetwork, self).__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttention(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                    # message_nn=nn.Linear(elem_fea_len, elem_fea_len),
                    # message_nn=nn.Identity(),
                )
                for _ in range(cry_heads)
            ]
        )

    def forward(
        self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
    ):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stiochiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(fea=elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    in the stiochiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super(MessageLayer, self).__init__()

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttention(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                    # message_nn=nn.Linear(2*elem_fea_len, elem_fea_len),
                    # message_nn=nn.Identity(),
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea=fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class WeightedMeanPooling(torch.nn.Module):
    """
    Weighted mean pooling
    """

    def __init__(self):
        super(WeightedMeanPooling, self).__init__()

    def forward(self, fea, index, weights):
        fea = weights * fea
        return scatter_mean(fea, index, dim=0)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class WeightedAttention(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(WeightedAttention, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, fea, index, weights):
        """ forward pass """

        gate = self.gate_nn(fea)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        fea = self.message_nn(fea)
        out = scatter_add(gate * fea, index, dim=0)

        return out

    def __repr__(self):
        return "{}(gate_nn={})".format(self.__class__.__name__, self.gate_nn)


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

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


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

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        # self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
        #                           for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
