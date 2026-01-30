# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics
from e3nn.util.jit import compile_mode

from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.utils.compile import conditional_torchscript_jit
from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from ..utils import with_edge_vectors_

from typing import Optional, List, Dict, Union

import math

def poly_fn(normed_dists, exp:float):
    res = 1 - (exp + 1) * (exp + 2) / 2 * torch.pow(normed_dists, exp) + exp * (exp + 2) * torch.pow(normed_dists, exp + 1) - exp * (exp + 1) / 2 * torch.pow(normed_dists, exp + 2)
    return torch.where(0 <= res, res, 0)

def fast_prune(src, dst, dists, num_nodes: int, hard_cutoff: float, poly_exp:float, sig_exp:float, weight_mean:float, weight_std:float):
    if len(dst) == 0:
        # If there aren't edges, then just return
        return torch.empty([0], device=dst.device), torch.empty([0], device=dst.device)

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
    max_len = group_sizes.max() if len(group_sizes) > 0 else torch.tensor(0, dtype=torch.int32) # max number of neighbors

    # Build mapping for indexing
    row_ids = torch.repeat_interleave(torch.arange(len(group_labels), device=dst.device), group_sizes) # (# of edges, )
    col_ids = torch.arange(len(sorted_src), device=dst.device) - torch.repeat_interleave(boundaries[:-1], group_sizes)

    # initialize output/mask
    adj = torch.zeros((int(num_nodes), int(max_len)), dtype=dst.dtype, device=dst.device)
    adj_dists = torch.ones((int(num_nodes), int(max_len)), dtype=dists.dtype, device=dst.device) * hard_cutoff
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

    d_sums = torch.sum(rank_weights * adj_dists, dim=1) + hard_cutoff * EPS
    w_sums = torch.sum(rank_weights, dim=1) + EPS
    cutoffs = d_sums / w_sums # (len of edges in pre-pruned graph)

    cutoffs_edge = cutoffs[dst]
    keep = dists < cutoffs_edge
    cutoffs = cutoffs_edge[keep]

    return cutoffs, keep


def _process_per_edge_type_cutoff(
    type_names: List[str], per_edge_type_cutoff, r_max: float
) -> torch.Tensor:
    num_types: int = len(type_names)

    # map dicts from type name to thing into lists
    processed_cutoffs = {}
    for source_type in type_names:
        if source_type in per_edge_type_cutoff:
            e = per_edge_type_cutoff[source_type]
            if not isinstance(e, float):
                cutoffs_for_source = []
                for target_type in type_names:
                    if target_type in e:
                        cutoffs_for_source.append(e[target_type])
                    else:
                        # default missing target types to `r_max`
                        cutoffs_for_source.append(r_max)
                processed_cutoffs[source_type] = cutoffs_for_source
            else:
                processed_cutoffs[source_type] = [e] * num_types
        else:
            # default missing source types to `r_max`
            processed_cutoffs[source_type] = [r_max] * num_types

    per_edge_type_cutoff = [processed_cutoffs[k] for k in type_names]
    per_edge_type_cutoff = torch.as_tensor(
        per_edge_type_cutoff, dtype=_GLOBAL_DTYPE
    ).contiguous()
    assert per_edge_type_cutoff.shape == (num_types, num_types)
    assert torch.all(per_edge_type_cutoff > 0)
    assert torch.all(per_edge_type_cutoff <= r_max)
    return per_edge_type_cutoff


@compile_mode("script")
class EdgeLengthNormalizer(GraphModuleMixin, torch.nn.Module):
    num_types: int
    r_max: float
    _per_edge_type: bool

    def __init__(
        self,
        r_max: float,
        type_names: List[str],
        per_edge_type_cutoff: Optional[
            Dict[str, Union[float, Dict[str, float]]]
        ] = None,
        # bookkeeping
        edge_type_field: str = AtomicDataDict.EDGE_TYPE_KEY,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()

        self.r_max = float(r_max)
        self.num_types = len(type_names)
        self.edge_type_field = edge_type_field
        self.norm_length_field = norm_length_field

        self._per_edge_type = False
        if per_edge_type_cutoff is not None:
            # process per_edge_type_cutoff
            self._per_edge_type = True
            per_edge_type_cutoff = _process_per_edge_type_cutoff(
                type_names, per_edge_type_cutoff, self.r_max
            )
            # compute 1/rmax and flatten for how they're used in forward, i.e. (n_type, n_type) -> (n_type^2,)
            rmax_recip = per_edge_type_cutoff.reciprocal().view(-1)
        else:
            rmax_recip = torch.as_tensor(1.0 / self.r_max, dtype=_GLOBAL_DTYPE)
        self.register_buffer("_rmax_recip", rmax_recip)

        irreps_out = {self.norm_length_field: Irreps([(1, (0, 1))])}
        if self._per_edge_type:
            irreps_out.update({self.edge_type_field: None})

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == get lengths with shape (num_edges, 1) ==
        data = with_edge_vectors_(data, with_lengths=True)
        r = data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1, 1)
        # == get norm ==
        rmax_recip = self._rmax_recip
        if self._per_edge_type:
            # get edge types with shape (2, num_edges) form first
            edge_type = torch.index_select(
                data[AtomicDataDict.ATOM_TYPE_KEY].view(-1),
                0,
                data[AtomicDataDict.EDGE_INDEX_KEY].view(-1),
            ).view(2, -1)
            data[self.edge_type_field] = edge_type
            # then convert into row-major NxN matrix index with shape (num_edges,)
            edge_type = edge_type[0] * self.num_types + edge_type[1]
            # (num_type^2,), (num_edges,) -> (num_edges, 1)
            rmax_recip = torch.index_select(rmax_recip, 0, edge_type).unsqueeze(-1)
        data[self.norm_length_field] = r * rmax_recip
        return data


@compile_mode("script")
class BesselEdgeLengthEncoding(GraphModuleMixin, torch.nn.Module):
    r"""Bessel edge length encoding.

    Args:
        num_bessels (int): number of Bessel basis functions
        trainable (bool): whether the :math:`n \pi` coefficients are trainable
        cutoff (torch.nn.Module): ``torch.nn.Module`` to apply a cutoff function that smoothly goes to zero at the cutoff radius
    """

    def __init__(
        self,
        cutoff: torch.nn.Module,
        num_bessels: int = 8,
        trainable: bool = False,
        # bookkeeping
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
        dyn_cutoff=False
    ):
        super().__init__()
        # === process inputs ===
        self.cutoff = conditional_torchscript_jit(cutoff)
        self.num_bessels = num_bessels
        self.trainable = trainable
        self.edge_invariant_field = edge_invariant_field
        self.norm_length_field = norm_length_field

        # === bessel weights ===
        bessel_weights = torch.linspace(
            start=1.0,
            end=self.num_bessels,
            steps=self.num_bessels,
            dtype=_GLOBAL_DTYPE,
        ).unsqueeze(0)  # (1, num_bessel)
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.edge_invariant_field: Irreps([(self.num_bessels, (0, 1))]),
                AtomicDataDict.EDGE_CUTOFF_KEY: "0e",
            },
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

        self.dyn_cutoff = dyn_cutoff

    def extra_repr(self) -> str:
        return f"num_bessels={self.num_bessels}"

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == Bessel basis ==
        x = data[self.norm_length_field]  # (num_edges, 1)
        # (num_edges, 1), (1, num_bessel) -> (num_edges, num_bessel)
        bessel = (torch.sinc(x * self.bessel_weights) * self.bessel_weights).to(
            self._output_dtype
        )

        # == polynomial cutoff ==
        if self.dyn_cutoff:
            cutoffs, keep = fast_prune(data[AtomicDataDict.EDGE_INDEX_KEY][0],
                                       data[AtomicDataDict.EDGE_INDEX_KEY][1],
                                       data[AtomicDataDict.EDGE_LENGTH_KEY],
                                       data[AtomicDataDict.POSITIONS_KEY].shape[0],
                                       6.0, 50.0, 10.0, float(self.dyn_cutoff), 4.0
                                    )
            
            # Prune the graph using keep
            data[AtomicDataDict.EDGE_INDEX_KEY] = data[AtomicDataDict.EDGE_INDEX_KEY][:, keep]
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][keep]
            data[AtomicDataDict.EDGE_VECTORS_KEY] = data[AtomicDataDict.EDGE_VECTORS_KEY][keep]
            data[AtomicDataDict.EDGE_ATTRS_KEY] = data[AtomicDataDict.EDGE_ATTRS_KEY][keep]
            data[AtomicDataDict.EDGE_LENGTH_KEY] = data[AtomicDataDict.EDGE_LENGTH_KEY][keep]
            data[AtomicDataDict.NORM_LENGTH_KEY] = data[AtomicDataDict.NORM_LENGTH_KEY][keep]
            data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoffs
            
            bessel = bessel[keep]

            # Apply the envelope weighting
            data[self.edge_invariant_field] = bessel * self.cutoff(data[AtomicDataDict.EDGE_LENGTH_KEY] / cutoffs).to(torch.float32).unsqueeze(1)

        else:
            cutoff = self.cutoff(x).to(self._output_dtype)
            data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff


            # == save product ==
            data[self.edge_invariant_field] = bessel * cutoff
        return data


@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors_(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh.to(self._output_dtype)
        return data


@compile_mode("script")
class AddRadialCutoffToData(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        cutoff: torch.nn.Module,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.cutoff = conditional_torchscript_jit(cutoff)
        self.norm_length_field = norm_length_field
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.EDGE_CUTOFF_KEY: "0e"}
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.EDGE_CUTOFF_KEY not in data:
            x = data[self.norm_length_field]
            cutoff = self.cutoff(x).to(self._output_dtype)
            data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff
        return data
