from __future__ import annotations

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_scatter import scatter_add, scatter_max, scatter_mean


class MeanPooling(nn.Module):
    """Mean pooling"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return scatter_mean(x, index, dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__


class SumPooling(nn.Module):
    """Sum pooling"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return scatter_add(x, index, dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__


class AttentionPooling(nn.Module):
    """Softmax attention layer"""

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """Initialize softmax attention layer.

        Args:
            gate_nn (nn.Module): _description_
            message_nn (nn.Module): _description_
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self) -> str:
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor, index: Tensor, weights: Tensor) -> Tensor:
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights**self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self) -> str:
        return self.__class__.__name__


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    in the stoichiometry graph.
    """

    def __init__(
        self,
        elem_fea_len: int,
        elem_heads: int,
        elem_gate: list[int],
        elem_msg: list[int],
    ) -> None:
        super().__init__()

        self._repr = (
            f"{self._get_name()}(elem_fea_len={elem_fea_len}, "
            f"elem_heads={elem_heads}, elem_gate={elem_gate}, elem_msg={elem_msg})"
        )

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_in_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            elem_weights (Tensor): shape (N,) The fractional weights of elements in their materials
            elem_in_fea (Tensor): shape (N, elem_fea_len) Element hidden features before message
                passing
            self_fea_idx (LongTensor): shape (M,) Indices of the 1st element in each of the M pairs
            nbr_fea_idx (LongTensor): shape (M,) Indices of the 2nd element in each of the M pairs

        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Returns:
            Tensor: shape (N, elem_fea_len) Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elements
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights))

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self) -> str:
        return self._repr


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        batchnorm: bool = False,
    ) -> None:
        """_summary_

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_layer_dims (list[int]): _description_
            activation (type[nn.Module], optional): _description_. Defaults to nn.LeakyReLU.
            batchnorm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def reset_parameters(self) -> None:
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()


class ResidualNetwork(nn.Module):
    """Feed forward Residual Neural Network"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list[int],
        activation: type[nn.Module] = nn.ReLU,
        batchnorm: bool = False,
        return_features: bool = False,
    ) -> None:
        """_summary_

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_layer_dims (list[int]): _description_
            activation (type[nn.Module], optional): _description_. Defaults to nn.ReLU.
            batchnorm (bool, optional): _description_. Defaults to False.
            return_features (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

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

    def forward(self, x: Tensor) -> Tensor:
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        if self.return_features:
            return x
        else:
            return self.fc_out(x)

    def __repr__(self) -> str:
        return self.__class__.__name__
