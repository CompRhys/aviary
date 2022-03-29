from __future__ import annotations

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_scatter import scatter_add, scatter_max


class AttentionPooling(nn.Module):
    """Softmax attention layer"""

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """Initialize softmax attention layer.

        Args:
            gate_nn (nn.Module): Neural network to calculate attention scalars
            message_nn (nn.Module): Neural network to evaluate message updates
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input features for nodes
            index (Tensor): The indices for scatter operation over nodes

        Returns:
            Tensor: Output features for nodes
        """
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self) -> str:
        gate_nn, message_nn = self.gate_nn, self.message_nn
        return f"{type(self).__name__}(gate_nn={gate_nn}, message_nn={message_nn})"


class WeightedAttentionPooling(nn.Module):
    """Weighted softmax attention layer"""

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """Initialize softmax attention layer

        Args:
            gate_nn (nn.Module): Neural network to calculate attention scalars
            message_nn (nn.Module): Neural network to evaluate message updates
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor, index: Tensor, weights: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input features for nodes
            index (Tensor): The indices for scatter operation over nodes
            weights (Tensor): The weights to assign to nodes

        Returns:
            Tensor: Output features for nodes
        """
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights**self.pow) * gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self) -> str:
        pow, gate_nn, message_nn = float(self.pow), self.gate_nn, self.message_nn
        return f"{type(self).__name__}(pow={pow:.3}, gate_nn={gate_nn}, message_nn={message_nn})"


class MessageLayer(nn.Module):
    """MessageLayer to propagate information between nodes in graph"""

    def __init__(
        self,
        msg_fea_len: int,
        num_msg_heads: int,
        msg_gate_layers: list[int],
        msg_net_layers: list[int],
    ) -> None:
        """Initialise MessageLayer

        Args:
            msg_fea_len (int): Number of input features
            num_msg_heads (int): Number of attention heads
            msg_gate_layers (list[int]): List of hidden layer sizes for gate network
            msg_net_layers (list[int]): List of hidden layer sizes for message network
        """
        super().__init__()

        self._repr = (
            f"{self._get_name()}(msg_fea_len={msg_fea_len}, "
            f"num_msg_heads={num_msg_heads}, msg_gate_layers={msg_gate_layers}, msg_net_layers={msg_net_layers})"
        )

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * msg_fea_len, 1, msg_gate_layers),
                    message_nn=SimpleNetwork(
                        2 * msg_fea_len, msg_fea_len, msg_net_layers
                    ),
                )
                for _ in range(num_msg_heads)
            ]
        )

    def forward(
        self,
        node_weights: Tensor,
        msg_in_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            node_weights (Tensor): The fractional weights of elements in their materials
            msg_in_fea (Tensor): Node hidden features before message passing
            self_idx (LongTensor): Indices of the 1st element in each of the node pairs
            nbr_idx (LongTensor): Indices of the 2nd element in each of the node pairs

        Returns:
            Tensor: node hidden features after message passing
        """
        # construct the total features for passing
        node_nbr_weights = node_weights[nbr_idx, :]
        msg_nbr_fea = msg_in_fea[nbr_idx, :]
        msg_self_fea = msg_in_fea[self_idx, :]
        fea = torch.cat([msg_self_fea, msg_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get node updates
        head_fea = []
        for attn_head in self.pooling:
            head_fea.append(attn_head(fea, index=self_idx, weights=node_nbr_weights))

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + msg_in_fea

    def __repr__(self) -> str:
        return self._repr


class SimpleNetwork(nn.Module):
    """Simple Feed Forward Neural Network"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        batchnorm: bool = False,
    ) -> None:
        """Create a simple feed forward neural network

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use. Defaults to nn.LeakyReLU.
            batchnorm (bool, optional): Whether to use batchnorm. Defaults to False.
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
        """Forward pass through network"""
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def reset_parameters(self) -> None:
        """Reinitialise network weights using PyTorch defaults"""
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(input_dim={self.fcs[0].in_features}, "
            f"output_dim={self.fc_out.out_features}, activation={type(self.acts[0]).__name__})"
        )


class ResidualNetwork(nn.Module):
    """Feed forward Residual Neural Network"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list[int],
        activation: type[nn.Module] = nn.ReLU,
        batchnorm: bool = False,
    ) -> None:
        """Create a feed forward neural network with skip connections

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use. Defaults to nn.LeakyReLU.
            batchnorm (bool, optional): Whether to use batchnorm. Defaults to False.
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

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through network"""
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(input_dim={self.fcs[0].in_features}, "
            f"output_dim={self.fc_out.out_features}, activation={type(self.acts[0]).__name__})"
        )
