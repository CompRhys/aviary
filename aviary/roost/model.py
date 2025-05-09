from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pymatgen.util.due import Doi, due
from torch import LongTensor, Tensor, nn

from aviary.core import BaseModelClass
from aviary.networks import ResidualNetwork, SimpleNetwork
from aviary.segments import MessageLayer, WeightedAttentionPooling
from aviary.utils import get_element_embedding


@due.dcite(Doi("10.1038/s41467-020-19964-7"), description="Roost model")
class Roost(BaseModelClass):
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
        n_targets: Sequence[int],
        elem_embedding: str = "matscholar200",
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 3,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
        trunk_hidden: Sequence[int] = (1024, 512),
        out_hidden: Sequence[int] = (256, 128, 64),
        **kwargs,
    ) -> None:
        """Composition-only model."""
        super().__init__(robust=robust, **kwargs)

        self.elem_embedding = get_element_embedding(elem_embedding)
        elem_emb_len = self.elem_embedding.weight.shape[1]
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

        self.material_nn = DescriptorNetwork(**desc_dict)  # type: ignore[arg-type]

        model_params = {
            "robust": robust,
            "n_targets": n_targets,
            "out_hidden": out_hidden,
            "trunk_hidden": trunk_hidden,
            "elem_embedding": elem_embedding,
            **desc_dict,
        }
        self.model_params.update(model_params)

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(elem_fea_len, out_hidden[0], trunk_hidden)

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
    ) -> tuple[Tensor, ...]:
        """Forward pass through the material_nn and output_nn."""
        elem_fea = self.elem_embedding(elem_fea)

        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the Roost Model."""

    def __init__(
        self,
        elem_emb_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 3,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
    ) -> None:
        """Bundles n_graph message passing layers followed by cry_heads weighted
        attention pooling layers.
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            MessageLayer(
                msg_fea_len=elem_fea_len,
                num_msg_heads=elem_heads,
                msg_gate_layers=elem_gate,
                msg_net_layers=elem_msg,
            )
            for _ in range(n_graph)
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
            )
            for _ in range(cry_heads)
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
    ) -> Tensor:
        """Forward pass through the DescriptorNetwork.

        Args:
            elem_weights (Tensor): Fractional weight of each Element in its
                stoichiometry
            elem_fea (Tensor): Element features of each of the elements in the batch
            self_idx (LongTensor): Indices of the 1st element in each of the pairs
            nbr_idx (LongTensor): Indices of the 2nd element in each of the pairs
            cry_elem_idx (list[LongTensor]): Mapping from the elem idx to crystal idx

        Returns:
            Tensor: Composition representation/features after message passing
        """
        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_idx, nbr_idx)

        # generate crystal features by pooling the elemental features
        head_fea = [
            attn_head(elem_fea, index=cry_elem_idx, weights=elem_weights)
            for attn_head in self.cry_pool
        ]

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_graph={len(self.graphs)}, cry_heads="
            f"{len(self.cry_pool)}, elem_emb_len={self.embedding.in_features}, "
            f"elem_fea_len={self.embedding.out_features})"
        )
