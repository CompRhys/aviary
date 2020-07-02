import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add, scatter_mean


class MeanPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, index):

        mean = scatter_mean(x, index, dim=0)

        return mean

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class SumPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x, index):

        mean = scatter_add(x, index, dim=0)

        return mean

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class AttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(AttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, fea, index):
        """ forward pass """

        gate = self.gate_nn(fea)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        fea = self.message_nn(fea)
        out = scatter_add(gate * fea, index, dim=0)

        return out


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

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.LeakyReLU
    ):
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
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

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

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.ReLU
    ):
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
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

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
