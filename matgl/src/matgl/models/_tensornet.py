"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import dgl
import torch
from torch import nn
import dgl.function as fn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.layers._embedding import TensorEmbedding
from matgl.layers._graph_convolution import TensorNetInteraction
from matgl.utils.maths import decompose_tensor, tensor_norm
import math
from ._core import MatGLModel
from matgl.utils.cutoff import polynomial_cutoff


if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)

rankings = [None, None]


def poly_fn(normed_dists, exp):
    res = 1 - (exp + 1) * (exp + 2) / 2 * torch.pow(normed_dists, exp) + exp * (exp + 2) * torch.pow(normed_dists, exp + 1) - exp * (exp + 1) / 2 * torch.pow(normed_dists, exp + 2)
    return torch.where(0 <= res, res, 0)


def fast_prune(g, edge_feat_name, hard_cutoff, poly_exp, sig_exp, weight_mean, weight_std):
    src, dst = g.edges()
    if len(dst) == 0:
        # If there aren't edges, then just return
        return g, torch.empty([0], device=dst.device)

    num_nodes = g.number_of_nodes()
    dists = g.edata[edge_feat_name]

    ### 1. Create padded adjacency matrix, corresponding distance matrix, and corresponding mask ###
    # sort by dst to group equal dst nodes together
    sorted_dst, sorted_idx = torch.sort(dst)
    sorted_src = src[sorted_idx] # src sorted according to dst
    sorted_dists = dists[sorted_idx] # dists sorted according to dst

    # find where groups change
    change_idx = torch.nonzero(sorted_dst[1:] != sorted_dst[:-1]).flatten() + 1 # indices where groups change
    boundaries = torch.cat((torch.tensor([0], device=dst.device), change_idx, torch.tensor([len(dst)], device=dst.device)))
    group_sizes = boundaries[1:] - boundaries[:-1] # number of src nodes in each group (num_groups, )
    group_labels = sorted_dst[boundaries[:-1]] # which node id is associated with which group
    max_len = group_sizes.max().item() if len(group_sizes) > 0 else 0 # max number of neighbors

    # Build mapping for indexing
    row_ids = torch.repeat_interleave(torch.arange(len(group_labels), device=dst.device), group_sizes) # (# of edges, )
    col_ids = torch.arange(len(sorted_src), device=dst.device) - torch.repeat_interleave(boundaries[:-1], group_sizes)

    # initialize output/mask
    adj = torch.zeros((num_nodes, max_len), dtype=dst.dtype, device=dst.device)
    adj_dists = torch.ones((num_nodes, max_len), dtype=dists.dtype, device=dst.device) * hard_cutoff
    mask = torch.zeros_like(adj, dtype=torch.bool, device=dst.device)

    # index into padded tensor
    adj[group_labels[row_ids], col_ids] = sorted_src # (num_nodes, max_neighbors) where non-masked elements are src nodes pointing to the dst node
    adj_dists[group_labels[row_ids], col_ids] = sorted_dists # same as adj but elements are distances corresponding to those edges
    mask[group_labels[row_ids], col_ids] = True

    ### 2. Calculate sigmoid-based differences ###
    poly_envs = poly_fn(adj_dists / hard_cutoff, poly_exp) * mask # (num_nodes, max_neighbors)
    diff = adj_dists.unsqueeze(2) - adj_dists.unsqueeze(1) # (num_nodes, max_neighbors, max_neighbors). Note: large amount of padding in highly unhomogenous density systems
    sig_rank_expanded = torch.special.expit(sig_exp * diff) # (num_nodes, max_neighbors, max_neighbors)

    ### 3. Apply normal distribution weighting to corresponding distances
    weighted_rank = sig_rank_expanded * poly_envs.unsqueeze(1) # (num_nodes, max_neighbors, max_neighbors)
    sig_mask = 1 - torch.eye(adj_dists.shape[1], device=adj_dists.device) # (max_neighbors, max_neighbors) mask used since differences were also taken between a src node and itself in sig_rank_expanded
    weighted_rank = weighted_rank * sig_mask.unsqueeze(0) # (num_nodes, max_neighbors, max_neighbors)
    ranks = weighted_rank.sum(dim=2) # (num_nodes, max_neighbors)

    EPS = 1e-6

    rank_weights = torch.exp(math.log(1e4) - math.log(weight_std) - 0.5 * math.log(2 * math.pi) - 0.5 * ((ranks - weight_mean) / weight_std) ** 2) + EPS # (num_nodes, max_neighbors)
    rank_weights = rank_weights * poly_envs # (num_nodes, max_neighbors)


    sorted_idxs = torch.argsort(adj_dists[0])
    gold_ranks = torch.zeros_like(adj_dists[0], dtype=torch.long)
    gold_ranks[sorted_idxs] = torch.arange(len(adj_dists[0]))

    rankings[0] = ranks[0].cpu()
    rankings[1] = gold_ranks.cpu()


    d_sums = torch.sum(rank_weights * adj_dists, dim=1) + hard_cutoff * EPS
    w_sums = torch.sum(rank_weights, dim=1) + EPS
    cutoffs = d_sums / w_sums # (len of edges in pre-pruned graph)


    cutoffs_edge = cutoffs[dst]
    keep = dists < cutoffs_edge


    # Remove edges of old graph
    remove_ids = torch.nonzero(~keep).squeeze().to(torch.int32)
    new_g = dgl.remove_edges(g, remove_ids)

    cutoffs = cutoffs_edge[keep]

    return new_g, cutoffs



class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        units: int = 64,
        ntypes_state: int | None = None,
        dim_state_embedding: int = 0,
        dim_state_feats: int | None = None,
        include_state: bool = False,
        nblocks: int = 2,
        num_rbf: int = 32,
        max_n: int = 3,
        max_l: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        use_smooth: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff: float = 5.0,
        equivariance_invariance_group: str = "O(3)",
        dtype: torch.dtype = matgl.float_th,
        width: float = 0.5,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        max_neighbors=False,
        poly_exp = 50,
        sig_exp = 10,
        weight_mean = 40,
        weight_std = 4,
        **kwargs,
    ):
        r"""

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            units (int, optional): Hidden embedding size.
                (default: :obj:`64`)
            ntypes_state (int): Number of state labels
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            include_state (bool): Whether to include states features
            nblocks (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis Gaussian functions :math:`\mu`.
                (default: :obj:`32`)
            max_n (int): maximum of n in spherical Bessel functions
            max_l (int): maximum of l in spherical Bessel functions
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            use_smooth (bool): Whether to use the smooth version of SphericalBessel functions.
                This is particularly important for the smoothness of PES.
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            cutoff (float): cutoff distance for interatomic interactions.
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
               (default :obj:`"O(3)"`)
            dtype (torch.dtype): data type for all variables
            width (float): the width of Gaussian radial basis functions
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
            **kwargs: For future flexibility. Not used at the moment.

        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(
            cutoff=cutoff,
            rbf_type=rbf_type,
            final=cutoff + 1.0,
            num_centers=num_rbf,
            width=width,
            smooth=use_smooth,
            max_n=max_n,
            max_l=max_l,
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], "Unknown group representation. Choose O(3) or SO(3)."

        self.units = units
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = nblocks
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff = cutoff
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.task_type = task_type

        # make sure the number of radial basis functions correct for tensor embedding
        if rbf_type == "SphericalBessel":
            num_rbf = max_n

        self.tensor_embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            dim_state_embedding=dim_state_embedding,
            dim_state_feats=dim_state_feats,
            ntypes_state=ntypes_state,
            include_state=include_state,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            {
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            }
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        if is_intensive:
            input_feats = units
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set, field=field
                )
                readout_feats = 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type:ignore[assignment]
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=units,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.is_intensive = is_intensive
        self.reset_parameters()

        self.max_neighbors = max_neighbors
        self.poly_exp = poly_exp
        self.sig_exp = sig_exp
        self.weight_mean = weight_mean
        self.weight_std = weight_std

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, **kwargs):
        """

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: output: Output property for a batch of graphs
        """
        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)


        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(g.edata["bond_dist"])
        g.edata["edge_attr"] = edge_attr

        # Graph Pruning
        if self.max_neighbors:
            g, cutoffs = fast_prune(g, "bond_dist", self.cutoff, self.poly_exp, self.sig_exp, self.weight_mean, self.weight_std)

            X, edge_feat, state_feat = self.tensor_embedding(g, state_attr, cutoffs=cutoffs)
        else:
            X, edge_feat, state_feat = self.tensor_embedding(g, state_attr, cutoffs=self.cutoff)


        # Embedding layer
        # Interaction layers
        for layer in self.layers:
            if self.max_neighbors:
                X = layer(g, X, cutoffs)
            else:
                X = layer(g, X, self.cutoff)
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_metrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.out_norm(x)
        x = self.linear(x)

        g.ndata["node_feat"] = x
        if self.is_intensive:
            node_vec = self.readout(g)
            vec = node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            return torch.squeeze(output)
        g.ndata["atomic_properties"] = self.final_layer(g)
        output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
