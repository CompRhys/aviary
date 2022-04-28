from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import BoolTensor, Tensor

from aviary.core import BaseModelClass
from aviary.segments import ResidualNetwork


class Wren(BaseModelClass):
    """The Roost model is comprised of a fully connected network
    and message passing graph layers.

    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stoichiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        robust: bool,
        n_targets: list[int],
        n_features: int,
        n_transformer_layers: int = 3,
        n_attention_heads: int = 5,
        trunk_hidden: list[int] = [1024, 512],
        out_hidden: list[int] = [256, 128, 64],
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            robust (bool): Whether to estimate standard deviation for use in a robust loss function
            n_targets (list[int]): Number of targets to train on
            trunk_hidden (list[int], optional): _description_. Defaults to [1024, 512].
            out_hidden (list[int], optional): _description_. Defaults to [256, 128, 64].
        """
        super().__init__(robust=robust, **kwargs)

        transformerLayer = nn.TransformerEncoderLayer(
            d_model=n_features, nhead=n_attention_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformerLayer, num_layers=n_transformer_layers
        )

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(n_features, out_hidden[0], trunk_hidden)

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(self, features: Tensor, mask: BoolTensor) -> tuple[Tensor, ...]:
        """Forward pass through the whole Wyckoff model.

        Args:
            features (Tensor): _description_
            mask (BoolTensor): _description_

        Returns:
            tuple[Tensor, ...]: Predictions for each batch of multitask targets.
        """
        embedding = self.transformer_encoder(features, src_key_padding_mask=mask)

        # aggregate all node representations into a single vector Wyckoff embedding
        # careful to ignore padded values when taking the mean
        embedding_masked = embedding * ~mask[..., None]
        aggregated_embedding = torch.sum(embedding_masked, dim=1) / torch.sum(
            ~mask, dim=1, keepdim=True
        )

        # main body of the FNN jointly used by all multitask objectives
        predictions = F.relu(self.trunk_nn(aggregated_embedding))

        return tuple(output_nn(predictions) for output_nn in self.output_nns)
