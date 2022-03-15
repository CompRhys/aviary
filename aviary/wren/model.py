from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_scatter import scatter_mean

from aviary.core import BaseModelClass
from aviary.segments import (
    MessageLayer,
    ResidualNetwork,
    SimpleNetwork,
    WeightedAttentionPooling,
)


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
        n_targets: Sequence[int],
        elem_emb_len: int,
        sym_emb_len: int,
        elem_fea_len: int = 32,
        sym_fea_len: int = 32,
        n_graph: int = 3,
        elem_heads: int = 1,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 1,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
        trunk_hidden: Sequence[int] = (
            1024,
            512,
        ),
        out_hidden: Sequence[int] = (
            256,
            128,
            64,
        ),
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            robust (bool): _description_
            n_targets (list[int]): _description_
            elem_emb_len (int): _description_
            sym_emb_len (int): _description_
            elem_fea_len (int, optional): _description_. Defaults to 32.
            sym_fea_len (int, optional): _description_. Defaults to 32.
            n_graph (int, optional): _description_. Defaults to 3.
            elem_heads (int, optional): _description_. Defaults to 1.
            elem_gate (list[int], optional): _description_. Defaults to [256].
            elem_msg (list[int], optional): _description_. Defaults to [256].
            cry_heads (int, optional): _description_. Defaults to 1.
            cry_gate (list[int], optional): _description_. Defaults to [256].
            cry_msg (list[int], optional): _description_. Defaults to [256].
            trunk_hidden (list[int], optional): _description_. Defaults to [1024, 512].
            out_hidden (list[int], optional): _description_. Defaults to [256, 128, 64].
        """
        super().__init__(robust=robust, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "sym_emb_len": sym_emb_len,
            "sym_fea_len": sym_fea_len,
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

        self.trunk_nn = ResidualNetwork(
            elem_fea_len + sym_fea_len, out_hidden[0], trunk_hidden
        )

        self.output_nns = nn.ModuleList(
            [ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets]
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> tuple[Tensor, ...]:
        """Forward pass through the material_nn and output_nn.

        Args:
            elem_weights (Tensor): _description_
            elem_fea (Tensor): _description_
            sym_fea (Tensor): _description_
            self_fea_idx (LongTensor): _description_
            nbr_fea_idx (LongTensor): _description_
            cry_elem_idx (LongTensor): _description_
            aug_cry_idx (LongTensor): _description_

        Returns:
            tuple[Tensor, ...]: _description_
        """
        crys_fea = self.material_nn(
            elem_weights,
            elem_fea,
            sym_fea,
            self_fea_idx,
            nbr_fea_idx,
            cry_elem_idx,
            aug_cry_idx,
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the Roost Model."""

    def __init__(
        self,
        elem_emb_len: int,
        sym_emb_len: int,
        elem_fea_len: int = 32,
        sym_fea_len: int = 32,
        n_graph: int = 3,
        elem_heads: int = 1,
        elem_gate: list[int] = [256],
        elem_msg: list[int] = [256],
        cry_heads: int = 1,
        cry_gate: list[int] = [256],
        cry_msg: list[int] = [256],
    ):
        """_summary_

        Args:
            elem_emb_len (int): _description_
            sym_emb_len (int): _description_
            elem_fea_len (int, optional): _description_. Defaults to 32.
            sym_fea_len (int, optional): _description_. Defaults to 32.
            n_graph (int, optional): _description_. Defaults to 3.
            elem_heads (int, optional): _description_. Defaults to 1.
            elem_gate (list[int], optional): _description_. Defaults to [256].
            elem_msg (list[int], optional): _description_. Defaults to [256].
            cry_heads (int, optional): _description_. Defaults to 1.
            cry_gate (list[int], optional): _description_. Defaults to [256].
            cry_msg (list[int], optional): _description_. Defaults to [256].
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.elem_embed = nn.Linear(elem_emb_len, elem_fea_len)
        self.sym_embed = nn.Linear(sym_emb_len + 1, sym_fea_len)

        # create a list of Message passing layers
        fea_len = elem_fea_len + sym_fea_len

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=fea_len,
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
                    gate_nn=SimpleNetwork(fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(fea_len, fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_fea_idx: LongTensor,
        nbr_fea_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            elem_weights (Tensor): Fractional weight of each Element in its stoichiometry
            elem_fea (Tensor): Element features of each of the N elements in the batch
            sym_fea (Tensor): Wyckoff Position features of each of the N elements in the batch
            self_fea_idx (Tensor): Indices of the first element in each of the M pairs
            nbr_fea_idx (Tensor): Indices of the second element in each of the M pairs
            cry_elem_idx (Tensor): Mapping from the elem idx to crystal idx
            aug_cry_idx (Tensor): Mapping from the crystal idx to augmentation idx

        Returns:
            Tensor: returns the crystal features of the materials in the batch
        """
        # embed the original features into the graph layer description
        elem_fea = self.elem_embed(elem_fea)
        sym_fea = self.sym_embed(torch.cat([sym_fea, elem_weights], dim=1))

        elem_fea = torch.cat([elem_fea, sym_fea], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        cry_fea = scatter_mean(
            torch.mean(torch.stack(head_fea), dim=0), aug_cry_idx, dim=0
        )

        return cry_fea

    def __repr__(self) -> str:
        return self.__class__.__name__
