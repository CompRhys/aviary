import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


class MeanPooling(nn.Module):
    """Mean pooling"""
    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_mean(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class SumPooling(nn.Module):
    """Sum pooling"""
    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_add(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Args:
            gate_nn: Variable(nn.Module)
            message_nn
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x, index):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, activation=nn.LeakyReLU,
        batchnorm=False
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.ReLU,
        batchnorm=False,
        return_features=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i+1]) for i in range(len(dims)-1)]
            )
        else:
            self.bns = nn.ModuleList(
                [nn.Identity() for i in range(len(dims)-1)]
            )

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.return_features = return_features
        if not self.return_features:
            self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns,
                                       self.res_fcs, self.acts):
            x = act(bn(fc(x)))+res_fc(x)

        if self.return_features:
            return x
        else:
            return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__
