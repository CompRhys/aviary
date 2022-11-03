from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_scatter import scatter_mean

from aviary.core import BaseModelClass
from aviary.networks import ResidualNetwork, SimpleNetwork
from aviary.segments import MessageLayer, WeightedAttentionPooling


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
        cry_heads: int = 3,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
        trunk_hidden: Sequence[int] = (1024, 512),
        out_hidden: Sequence[int] = (256, 128, 64),
        **kwargs,
    ) -> None:
        """Composition plus symmetry model.

        Args:
            robust (bool): If True, the number of model outputs is doubled. 2nd output for each
                target will be an estimate for the aleatoric uncertainty (uncertainty inherent to
                the sample) which can be used with a robust loss function to attenuate the weighting
                of uncertain samples.
            n_targets (list[int]): Number of targets to train on. 1 for regression and number of
                different class labels for classification.
            elem_emb_len (int): Number of features in initial element embedding
            sym_emb_len (int): Number of features in initial Wyckoff letter embedding
            elem_fea_len (int, optional): Number of hidden features to use to encode elements.
                Defaults to 32.
            sym_fea_len (int, optional): Number of hidden features to use to encode Wyckoff letters.
                Defaults to 32.
            n_graph (int, optional): Number of message passing operations to carry out.
                Defaults to 3.
            elem_heads (int, optional): Number of parallel attention heads per message passing
                operation. Defaults to 1.
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

        model_params = {
            "robust": robust,
            "n_targets": n_targets,
            "out_hidden": out_hidden,
            "trunk_hidden": trunk_hidden,
            **desc_dict,
        }
        self.model_params.update(model_params)

        # define an output neural network
        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(
            elem_fea_len + sym_fea_len, out_hidden[0], trunk_hidden
        )

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> tuple[Tensor, ...]:
        """Forward pass through the material_nn and output_nn.

        Args:
            elem_weights (Tensor): _description_
            elem_fea (Tensor): _description_
            sym_fea (Tensor): _description_
            self_idx (LongTensor): _description_
            nbr_idx (LongTensor): _description_
            cry_elem_idx (LongTensor): _description_
            aug_cry_idx (LongTensor): _description_

        Returns:
            tuple[Tensor, ...]: Predicted values for each target
        """
        crys_fea = self.material_nn(
            elem_weights,
            elem_fea,
            sym_fea,
            self_idx,
            nbr_idx,
            cry_elem_idx,
            aug_cry_idx,
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the Roost model."""

    def __init__(
        self,
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
    ):
        """Message passing section of the Roost model.

        Args:
            elem_emb_len (int): Number of features in initial element embedding
            sym_emb_len (int): Number of features in initial Wyckoff letter embedding
            elem_fea_len (int, optional): Number of hidden features to use to encode elements.
                Defaults to 32.
            sym_fea_len (int, optional): Number of hidden features to use to encode Wyckoff letters.
                Defaults to 32.
            n_graph (int, optional): Number of message passing operations to carry out.
                Defaults to 3.
            elem_heads (int, optional): Number of parallel attention heads per message passing
                operation. Defaults to 1.
            elem_gate (list[int], optional): _description_. Defaults to [256].
            elem_msg (list[int], optional): _description_. Defaults to [256].
            cry_heads (int, optional): Number of attention heads. Defaults to 1.
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
            MessageLayer(
                msg_fea_len=fea_len,
                num_msg_heads=elem_heads,
                msg_gate_layers=elem_gate,
                msg_net_layers=elem_msg,
            )
            for i in range(n_graph)
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork(fea_len, 1, cry_gate),
                message_nn=SimpleNetwork(fea_len, fea_len, cry_msg),
            )
            for _ in range(cry_heads)
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> Tensor:
        """Forward pass

        Args:
            elem_weights (Tensor): Fractional weight of each Element in its stoichiometry
            elem_fea (Tensor): Element features of each of the N elements in the batch
            sym_fea (Tensor): Wyckoff Position features of each of the N elements in the batch
            self_idx (Tensor): Indices of the first element in each of the M pairs
            nbr_idx (Tensor): Indices of the second element in each of the M pairs
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
            elem_fea = graph_func(elem_weights, elem_fea, self_idx, nbr_idx)

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
        return (
            f"{type(self).__name__}(n_graph={len(self.graphs)}, cry_heads="
            f"{len(self.cry_pool)}, elem_emb_len={self.elem_emb_len}, "
            f"sym_emb_len={self.sym_emb_len})"
        )
