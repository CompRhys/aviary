from collections.abc import Sequence

import torch
from torch import LongTensor, Tensor, nn

from aviary.networks import SimpleNetwork
from aviary.scatter import scatter_reduce


class AttentionPooling(nn.Module):
    """Softmax attention layer. Currently unused."""

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
        """Forward pass.

        Args:
            x (Tensor): Input features for nodes
            index (Tensor): The indices for scatter operation over nodes

        Returns:
            Tensor: Output features for nodes
        """
        gate = self.gate_nn(x)

        gate -= scatter_reduce(gate, index, dim=0, reduce="amax")[index]
        gate = gate.exp()
        gate /= scatter_reduce(gate, index, dim=0, reduce="sum")[index] + 1e-10

        x = self.message_nn(x)
        return scatter_reduce(gate * x, index, dim=0, reduce="sum")

    def __repr__(self) -> str:
        gate_nn, message_nn = self.gate_nn, self.message_nn
        return f"{type(self).__name__}({gate_nn=}, {message_nn=})"


class WeightedAttentionPooling(nn.Module):
    """Weighted softmax attention layer."""

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        """Initialize softmax attention layer.

        Args:
            gate_nn (nn.Module): Neural network to calculate attention scalars
            message_nn (nn.Module): Neural network to evaluate message updates
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor, index: Tensor, weights: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input features for nodes
            index (Tensor): The indices for scatter operation over nodes
            weights (Tensor): The weights to assign to nodes

        Returns:
            Tensor: Output features for nodes
        """
        gate = self.gate_nn(x)

        gate -= scatter_reduce(gate, index, dim=0, reduce="amax")[index]
        gate = (weights**self.pow) * gate.exp()
        gate /= scatter_reduce(gate, index, dim=0, reduce="sum")[index] + 1e-10

        x = self.message_nn(x)
        return scatter_reduce(gate * x, index, dim=0, reduce="sum")

    def __repr__(self) -> str:
        pow, gate_nn, message_nn = float(self.pow), self.gate_nn, self.message_nn
        return f"{type(self).__name__}({pow=:.3}, {gate_nn=}, {message_nn=})"


class MessageLayer(nn.Module):
    """MessageLayer to propagate information between nodes in graph."""

    def __init__(
        self,
        msg_fea_len: int,
        num_msg_heads: int,
        msg_gate_layers: Sequence[int],
        msg_net_layers: Sequence[int],
    ) -> None:
        """Initialise MessageLayer.

        Args:
            msg_fea_len (int): Number of input features
            num_msg_heads (int): Number of attention heads
            msg_gate_layers (list[int]): List of hidden layer sizes for gate network
            msg_net_layers (list[int]): List of hidden layer sizes for message network
        """
        super().__init__()

        self._repr = (
            f"{self._get_name()}({msg_fea_len=}, {num_msg_heads=}, {msg_gate_layers=}, "
            f"{msg_net_layers=})"
        )

        # Pooling and Output
        self.pooling = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork(2 * msg_fea_len, 1, msg_gate_layers),
                message_nn=SimpleNetwork(2 * msg_fea_len, msg_fea_len, msg_net_layers),
            )
            for _ in range(num_msg_heads)
        )

    def forward(
        self,
        node_weights: Tensor,
        node_prev_features: Tensor,
        self_idx: LongTensor,
        neighbor_idx: LongTensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            node_weights (Tensor): The fractional weights of elements in their materials
            node_prev_features (Tensor): Node hidden features before message passing
            self_idx (LongTensor): Indices of the 1st element in each of the node pairs
            neighbor_idx (LongTensor): Indices of the 2nd element in each of the node
                pairs

        Returns:
            Tensor: node hidden features after message passing
        """
        # construct the total features for passing
        node_nbr_weights = node_weights[neighbor_idx, :]
        msg_nbr_fea = node_prev_features[neighbor_idx, :]
        msg_self_fea = node_prev_features[self_idx, :]
        message = torch.cat([msg_self_fea, msg_nbr_fea], dim=1)

        # sum selectivity over the neighbors to get node updates
        head_features = []
        for attn_head in self.pooling:
            out_msg = attn_head(message, index=self_idx, weights=node_nbr_weights)
            head_features.append(out_msg)

        # average the attention heads
        node_update = torch.stack(head_features).mean(dim=0)

        return node_update + node_prev_features

    def __repr__(self) -> str:
        return self._repr
