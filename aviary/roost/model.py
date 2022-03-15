from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from aviary.core import BaseModelClass
from aviary.segments import (
    MessageLayer,
    ResidualNetwork,
    SimpleNetwork,
    WeightedAttentionPooling,
)


class Roost(BaseModelClass):
    """
    The Roost model is comprised of a fully connected network
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
        elem_emb_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: list[int] = [256],
        elem_msg: list[int] = [256],
        cry_heads: int = 3,
        cry_gate: list[int] = [256],
        cry_msg: list[int] = [256],
        trunk_hidden: list[int] = [1024, 512],
        out_hidden: list[int] = [256, 128, 64],
        **kwargs,
    ) -> None:
        if isinstance(out_hidden[0], list):
            raise ValueError("boo hiss bad user")
            # assert all([isinstance(x, list) for x in out_hidden]),
            #   'all elements of out_hidden must be ints or all lists'
            # assert len(out_hidden) == len(n_targets),
            #   'out_hidden-n_targets length mismatch'

        super().__init__(robust=robust, **kwargs)

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

        self.material_nn = DescriptorNetwork(**desc_dict)  # type: ignore

        self.model_params.update(
            {
                "robust": robust,
                "n_targets": n_targets,
                "out_hidden": out_hidden,
                "trunk_hidden": trunk_hidden,
            }
        )

        self.model_params.update(desc_dict)

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(elem_fea_len, out_hidden[0], trunk_hidden)

        self.output_nns = nn.ModuleList(
            [ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets]
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
        cry_elem_idx: LongTensor,
    ) -> tuple[Tensor, ...]:
        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: list[int] = [256],
        elem_msg: list[int] = [256],
        cry_heads: int = 3,
        cry_gate: list[int] = [256],
        cry_msg: list[int] = [256],
    ) -> None:
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
        cry_elem_idx: LongTensor,
    ) -> Tensor:
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """
        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self) -> str:
        return self.__class__.__name__
