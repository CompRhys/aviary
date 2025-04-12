from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pymatgen.util.due import Doi, due
from torch import BoolTensor, Tensor, nn

from aviary.core import AGGREGATORS, BaseModelClass
from aviary.networks import ResidualNetwork


@due.dcite(Doi("10.48550/arXiv.2308.14920"), description="Wrenformer model")
@due.dcite(Doi("10.1038/s41524-021-00545-1"), description="Crabnet model")
class Wrenformer(BaseModelClass):
    """Crabnet-inspired re-implementation of Wren as a transformer.
    https://github.com/anthony-wang/CrabNet.

    Wrenformer consists of a transformer encoder who's job it is to generate an
    informative embedding given a material's composition and Wyckoff positions (think
    crystal symmetries). Since the embedding is trainable, it is systematically
    improvable with more data. Using this embedding, the residual output network
    regresses or classifies the targets.

    Can also be used as Roostformer by generating the input features with
    get_composition_embedding() instead of wyckoff_embedding_from_aflow_str(). Model
    class, collate_batch function and DataLoader stay the same.

    See https://nature.com/articles/s41524-021-00545-1/tables/2 for default CrabNet
    hyperparams.
    """

    def __init__(
        self,
        robust: bool,
        n_targets: Sequence[int],
        n_features: int,
        d_model: int = 128,
        n_attn_layers: int = 6,
        n_attn_heads: int = 4,
        dropout: float = 0.0,
        trunk_hidden: Sequence[int] = (1024, 512),
        out_hidden: Sequence[int] = (256, 128, 64),
        embedding_aggregations: Sequence[str] = ("mean",),
        **kwargs,
    ) -> None:
        """Initialize the Wrenformer model.

        Args:
            n_targets (list[int]): Number of targets to train on. 1 for regression or
                number of classes for classification.
            n_features (int): Number of features in the input data (aka embedding size).
            d_model (int): Dimension of the transformer layers. Determines size of the
                learned embedding passed to the output NN. d_model should be increased
                for large datasets. Defaults to 256.
            n_attn_layers (int): Number of transformer encoder layers to use. Defaults
                to 3.
            n_attn_heads (int): Number of attention heads to use in the transformer.
                d_model needs to be divisible by this number. Defaults to 4.
            dropout (float, optional): Dropout rate for the transformer encoder. Defaults
                to 0.
            trunk_hidden (list[int], optional): Number of hidden units in the trunk
                network which is shared across tasks when multitasking. Defaults to
                [1024, 512].
            out_hidden (list[int], optional): Number of hidden units in the output
                networks which are task-specific. Defaults to [256, 128, 64].
            robust (bool): If True, the number of model outputs is doubled. 2nd output
                for each target will be an estimate for the aleatoric uncertainty
                (uncertainty inherent to the sample) which can be used with a robust
                loss function to attenuate the weighting of uncertain samples.
            embedding_aggregations (list[str]): Aggregations to apply to the learned
                embedding returned by the transformer encoder before passing into the
                ResidualNetwork. One or more of ['mean', 'std', 'sum', 'min', 'max'].
                Defaults to ['mean'].
            **kwargs: Additional keyword arguments passed to BaseModelClass.
        """
        super().__init__(robust=robust, **kwargs)

        model_params = {
            "robust": robust,
            "n_targets": n_targets,
            "n_features": n_features,
            "d_model": d_model,
            "n_attn_layers": n_attn_layers,
            "n_attn_heads": n_attn_heads,
            "dropout": dropout,
            "trunk_hidden": trunk_hidden,
            "out_hidden": out_hidden,
            "embedding_aggregations": embedding_aggregations,
        }
        self.model_params.update(model_params)

        # up- or down-size embedding dimension (n_features) to model dimension (d_model)
        self.resize_embedding = nn.Linear(n_features, d_model)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_attn_heads,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=n_attn_layers, enable_nested_tensor=False
        )

        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.embedding_aggregations = embedding_aggregations
        self.trunk_nn = ResidualNetwork(
            # len(embedding_aggregations) = number of catted tensors in
            # aggregated_embeddings below
            input_dim=len(embedding_aggregations) * d_model,
            output_dim=out_hidden[0],
            hidden_layer_dims=trunk_hidden,
        )

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(self, features: Tensor, mask: BoolTensor, *args) -> tuple[Tensor, ...]:
        """Forward pass through the Wrenformer.

        Args:
            features (Tensor): Padded sequences of Wyckoff embeddings.
            mask (BoolTensor): Indicates which tensor entries are sequence padding.
                mask[i,j] = True means batch index i, sequence index j is not allowed to
                attend, False means it participates in self-attention.
            *args: Additional arguments are only needed for Wrenformer,
                not Roostformer. So if not present, we're running as Roostformer.
                Else only first item in args is used as equivalence_counts (list[int])
                which determine the length of slices in the batch dimension originating
                from equivalent Wyckoff sets. Features for equivalent Wyckoff sets are
                averaged to remove ambiguity in assigning Wyckoff letters to Wyckoff
                positions. This averaging reduces dim=0 of features back to batch_size.

        Returns:
            tuple[Tensor, ...]: Predictions for each batch of multitask targets.
        """
        # project input embedding onto d_model dimensions
        features = self.resize_embedding(features)
        # run self-attention
        embeddings = self.transformer_encoder(features, src_key_padding_mask=mask)

        if len(args) == 1:
            # if forward() got a 3rd arg, we're running as Wrenformer, not Roostformer
            equivalence_counts: Sequence[int] = args[0]
            # average over equivalent Wyckoff sets in a given material (brings dim 0 of
            # features back to batch_size)
            equiv_embeddings = embeddings.split(equivalence_counts, dim=0)
            augmented_embeddings = [tensor.mean(dim=0) for tensor in equiv_embeddings]
            embeddings = torch.stack(augmented_embeddings)
            # all equivalent Wyckoff sets have the same mask so we pick the 1st one from
            # each split
            mask = torch.stack([t[0] for t in mask.split(equivalence_counts, dim=0)])

        # aggregate all embedding sequences of a material corresponding to Wyckoff
        # positions into a single vector Wyckoff embedding
        # careful to ignore padded values when taking the mean
        inv_mask: torch.BoolTensor = ~mask[..., None]

        aggregation_funcs = [AGGREGATORS[key] for key in self.embedding_aggregations]
        aggregated_embeddings = torch.cat(
            [func(embeddings, inv_mask, 1) for func in aggregation_funcs], dim=1
        )

        # main body of the feed-forward NN jointly used by all multitask objectives
        predictions = F.relu(self.trunk_nn(aggregated_embeddings))

        return tuple(output_nn(predictions) for output_nn in self.output_nns)
