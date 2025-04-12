from collections.abc import Sequence

from torch import Tensor, nn


class SimpleNetwork(nn.Module):
    """Simple Feed Forward Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = False,
    ) -> None:
        """Create a simple feed forward neural network.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(
                nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
            )
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through network."""
        for fc, bn, act in zip(self.fcs, self.bns, self.acts, strict=False):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def reset_parameters(self) -> None:
        """Reinitialize network weights using PyTorch defaults."""
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dim = self.fc_out.out_features
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dim=}, {activation=})"


class ResidualNetwork(nn.Module):
    """Feed forward Residual Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,
    ) -> None:
        """Create a feed forward neural network with skip connections.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            hidden_layer_dims (list[int]): List of hidden layer sizes
            activation (type[nn.Module], optional): Which activation function to use.
                Defaults to nn.LeakyReLU.
            batch_norm (bool, optional): Whether to use batch_norm. Defaults to False.
        """
        super().__init__()

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(
                nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1)
            )
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.res_fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=False)
            if (dims[idx] != dims[idx + 1])
            else nn.Identity()
            for idx in range(len(dims) - 1)
        )
        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through network."""
        for fc, bn, res_fc, act in zip(
            self.fcs, self.bns, self.res_fcs, self.acts, strict=False
        ):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)

    def __repr__(self) -> str:
        input_dim = self.fcs[0].in_features
        output_dim = self.fc_out.out_features
        activation = type(self.acts[0]).__name__
        return f"{type(self).__name__}({input_dim=}, {output_dim=}, {activation=})"
