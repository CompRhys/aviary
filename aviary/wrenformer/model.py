from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import BoolTensor, Tensor

from aviary.core import BaseModelClass
from aviary.segments import ResidualNetwork


class Wrenformer(BaseModelClass):
    """Crabnet-inspired re-implementation of Wren as a transformer.
    https://github.com/anthony-wang/CrabNet

    Wrenformer consists of a transformer encoder who's job it is to generate an informative
    embedding given a material's composition and Wyckoff positions (think crystal symmetries).
    Since the embedding is trainable, it is systematically improvable with more data.
    Using this embedding, the residual output network regresses or classifies the targets.

    Can also be used as Roostformer by generating the input features with
    get_composition_embedding() instead of wyckoff_embedding_from_aflow_str(). Model class,
    collate_batch function and DataLoader stay the same.

    See https://nature.com/articles/s41524-021-00545-1/tables/2 for default CrabNet hyperparams.
    """

    def __init__(
        self,
        n_targets: list[int],
        n_features: int,
        d_model: int = 128,
        n_transformer_layers: int = 6,
        n_attention_heads: int = 4,
        trunk_hidden: list[int] = [1024, 512],
        out_hidden: list[int] = [256, 128, 64],
        robust: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Wrenformer model.

        Args:
            n_targets (list[int]): Number of targets to train on. 1 for regression or number of
                classes for classification.
            n_features (int): Number of features in the input data (aka embedding size).
            d_model (int): Dimension of the transformer layers. Determines size of the learned
                embedding passed to the output NN. d_model should be increased for large datasets.
                Defaults to 256.
            n_transformer_layers (int): Number of transformer layers to use. Defaults to 3.
            n_attention_heads (int): Number of attention heads to use in the transformer. d_model
                needs to be divisible by this number. Defaults to 4.
            trunk_hidden (list[int], optional): Number of hidden units in the trunk network which
                is shared across tasks when multitasking. Defaults to [1024, 512].
            out_hidden (list[int], optional): Number of hidden units in the output networks which
                are task-specific. Defaults to [256, 128, 64].
            robust (bool): Whether to estimate standard deviation of a prediction alongside the
                prediction itself for use in a robust loss function. Defaults to False.
        """
        super().__init__(robust=robust, **kwargs)

        # up or down size embedding dimension to chosen model dimension
        self.resize_embedding = nn.Linear(n_features, d_model)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_attention_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=n_transformer_layers
        )

        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(
            input_dim=d_model,
            output_dim=out_hidden[0],
            hidden_layer_dims=trunk_hidden,
        )

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(  # type: ignore
        self, features: Tensor, mask: BoolTensor, *args
    ) -> tuple[Tensor, ...]:
        """Forward pass through the Wrenformer.

        Args:
            features (Tensor): Padded sequences of Wyckoff embeddings.
            mask (BoolTensor): Indicates which tensor entries are padding.

        Returns:
            tuple[Tensor, ...]: Predictions for each batch of multitask targets.
        """
        # project input embedding onto d_model dimensions
        features = self.resize_embedding(features)
        # run self-attention
        embeddings = self.transformer_encoder(features, src_key_padding_mask=mask)

        if len(args) == 1:
            # running as wrenformer, not roostformer
            equivalence_counts: list[int] = args[0]
            # average over equivalent Wyckoff sets in a given material (brings len of dim 0
            # back to batch_size)
            equiv_embeddings = embeddings.split(equivalence_counts, dim=0)
            augmented_embeddings = [tensor.mean(dim=0) for tensor in equiv_embeddings]
            embeddings = torch.stack(augmented_embeddings)
            # do the same for mask
            mask = torch.stack([t[0] for t in mask.split(equivalence_counts, dim=0)])

        # aggregate all node representations into a single vector Wyckoff embedding
        # careful to ignore padded values when taking the mean
        embedding_masked = embeddings * ~mask[..., None]
        seq_lens = torch.sum(~mask, dim=1, keepdim=True)
        aggregated_embedding = torch.sum(embedding_masked, dim=1) / seq_lens

        # main body of the feed-forward NN jointly used by all multitask objectives
        predictions = F.relu(self.trunk_nn(aggregated_embedding))

        return tuple(output_nn(predictions) for output_nn in self.output_nns)
