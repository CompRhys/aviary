import torch
import torch.nn as nn
import torch.nn.functional as F

from roost.core import BaseModelClass
from roost.segments import (
    SimpleNetwork,
    ResidualNetwork,
    MeanPooling,
    WeightedAttentionPooling
)


class Wren(BaseModelClass):
    """
    The Roost model is comprised of a fully connected network
    and message passing graph layers.

    The message passing layers are used to determine a descriptor set
    for the fully connected network. The graphs are used to represent
    the stiochiometry of inorganic materials in a trainable manner.
    This makes them systematically improvable with more data.
    """

    def __init__(
        self,
        robust,
        n_targets,
        elem_emb_len,
        sym_emb_len,
        elem_fea_len=32,
        sym_fea_len=32,
        n_graph=3,
        elem_heads=1,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=1,
        cry_gate=[256],
        cry_msg=[256],
        trunk_hidden=[1024, 512],
        out_hidden=[256, 128, 64],
        **kwargs
    ):
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

        self.material_nn = DescriptorNetwork(**desc_dict)

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

        self.trunk_nn = ResidualNetwork(elem_fea_len+sym_fea_len, out_hidden[0], trunk_hidden)

        self.output_nns = nn.ModuleList([
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        ])

    def forward(self, elem_weights, elem_fea, sym_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, aug_cry_idx):
        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights, elem_fea, sym_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, aug_cry_idx
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        # apply neural network to map from learned features to target
        return (output_nn(crys_fea) for output_nn in self.output_nns)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        sym_emb_len,
        elem_fea_len=32,
        sym_fea_len=32,
        n_graph=3,
        elem_heads=1,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=1,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
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
                    # message_nn=nn.Linear(elem_fea_len, elem_fea_len),
                    # message_nn=nn.Identity(),
                )
                for _ in range(cry_heads)
            ]
        )

        self.aug_pool = MeanPooling()

    def forward(self, elem_weights, elem_fea, sym_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, aug_cry_idx):
        """Forward pass

        Args:
            elem_weights (torch.Tensor): Fractional weight of each
                Element in its stiochiometry
            elem_fea (torch.Tensor): Element features of each of
                the N elems in the batch
            sym_fea (torch.Tensor): Wyckoff Position features of each
                of the N elems in the batch
            self_fea_idx (torch.Tensor): Indices of the first element in
                each of the M pairs
            nbr_fea_idx (torch.Tensor): Indices of the second element in
                each of the M pairs
            cry_elem_idx (torch.Tensor): Mapping from the elem idx to
                crystal idx
            aug_cry_idx (torch.Tensor): Mapping from the crystal idx to
                augmentation idx

        Returns:
            torch.Tensor: returns the crystal features of the materials
                in the batch
        """

        # embed the original features into the graph layer description
        elem_fea = self.elem_embed(elem_fea)
        sym_fea = self.sym_embed(torch.cat([sym_fea, elem_weights], dim=1))

        # # do this so that we can examine the embeddings without
        # # influence of the weights
        elem_fea = torch.cat([elem_fea, sym_fea], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # print(len(self_fea_idx), len(nbr_fea_idx), len(cry_elem_idx))
        # print(elem_fea.size())

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        cry_fea = self.aug_pool(torch.mean(torch.stack(head_fea), dim=0), aug_cry_idx)

        return cry_fea

    def __repr__(self):
        return f"{self.__class__.__name__}"


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    in the stiochiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super().__init__()

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                    # message_nn=nn.Linear(2*elem_fea_len, elem_fea_len),
                    # message_nn=nn.Identity(),
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return f"{self.__class__.__name__}"
