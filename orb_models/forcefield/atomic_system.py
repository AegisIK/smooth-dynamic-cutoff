from dataclasses import dataclass
from typing import Iterable, List, Optional
from copy import deepcopy

import ase
import torch
from ase import constraints
from ase.geometry.cell import cell_to_cellpar
from ase.calculators.singlepoint import SinglePointCalculator

from orb_models.forcefield import featurization_utilities as feat_util
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod


@dataclass
class SystemConfig:
    """Config controlling how to featurize a system of atoms.

    Args:
        radius: radius for edge construction
        max_num_neighbors: maximum number of neighbours each node can send messages to.
    """

    radius: float
    max_num_neighbors: int


def atom_graphs_to_ase_atoms(
    graphs: AtomGraphs,
    energy: Optional[torch.Tensor] = None,
    forces: Optional[torch.Tensor] = None,
    stress: Optional[torch.Tensor] = None,
) -> List[ase.Atoms]:
    """Converts a list of graphs to a list of ase.Atoms."""
    graphs = graphs.to("cpu")

    atomic_numbers = torch.argmax(graphs.atomic_numbers_embedding, dim=-1)
    atomic_numbers_split = torch.split(atomic_numbers, graphs.n_node.tolist())
    positions_split = torch.split(graphs.positions, graphs.n_node.tolist())
    assert graphs.tags is not None and graphs.system_features is not None
    tags = torch.split(graphs.tags, graphs.n_node.tolist())

    calculations = {}
    if energy is not None:
        energy_list = torch.unbind(energy.cpu().detach())
        assert len(energy_list) == len(atomic_numbers_split)
        calculations["energy"] = energy_list
    if forces is not None:
        forces_list = torch.split(forces.cpu().detach(), graphs.n_node.tolist())
        assert len(forces_list) == len(atomic_numbers_split)
        calculations["forces"] = forces_list  # type: ignore
    if stress is not None:
        stress_list = torch.unbind(stress.cpu().detach())
        assert len(stress_list) == len(atomic_numbers_split)
        calculations["stress"] = stress_list

    atoms_list = []
    for index, (n, p, c, t) in enumerate(
        zip(atomic_numbers_split, positions_split, graphs.cell, tags)
    ):
        atoms = ase.Atoms(
            numbers=n.detach(),
            positions=p.detach(),
            cell=c.detach(),
            tags=t.detach(),
            pbc=torch.any(c != 0),
        )
        if calculations != {}:
            # note: important to save scalar energy as a float not array
            spc = SinglePointCalculator(
                atoms=atoms,
                **{
                    key: (
                        val[index].item()
                        if val[index].nelement() == 1
                        else val[index].numpy()
                    )
                    for key, val in calculations.items()
                },
            )
            atoms.calc = spc
        atoms_list.append(atoms)

    return atoms_list



def poly_fn(normed_dists, exp):
    res = 1 - (exp + 1) * (exp + 2) / 2 * torch.pow(normed_dists, exp) + exp * (exp + 2) * torch.pow(normed_dists, exp + 1) - exp * (exp + 1) / 2 * torch.pow(normed_dists, exp + 2)
    return torch.where(0 <= res, res, 0)

import math

def fast_prune(src, dst, num_nodes, dists, hard_cutoff, poly_exp, sig_exp, weight_mean, weight_std):
    if len(dst) == 0:
        # If there aren't edges, then just return
        return torch.empty([0], device=dst.device, dtype=torch.int32), torch.empty([0], device=dst.device, dtype=torch.int32)

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

    d_sums = torch.sum(rank_weights * adj_dists, dim=1) + hard_cutoff * EPS
    w_sums = torch.sum(rank_weights, dim=1) + EPS
    cutoffs = d_sums / w_sums # (len of edges in pre-pruned graph)

    cutoffs_edge = cutoffs[dst]
    keep = dists < cutoffs_edge

    cutoffs = cutoffs_edge[keep]

    return cutoffs, keep

def ase_atoms_to_atom_graphs(
    atoms: ase.Atoms,
    system_config: SystemConfig,
    *,
    wrap: bool = True,
    edge_method: Optional[EdgeCreationMethod] = None,
    max_num_neighbors: Optional[int] = None,
    system_id: Optional[int] = None,
    half_supercell: Optional[bool] = None,
    device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
    graph_construction_dtype: Optional[torch.dtype] = None
) -> AtomGraphs:
    """Convert an ase.Atoms object into AtomGraphs format, ready for use in a model.

    Args:
        atoms: ase.Atoms object
        wrap: whether to wrap atomic positions into the central unit cell (if there is one).
        edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
            If None, the edge method is chosen as follows:
            * knn_brute_force: If device is not CPU, and cuML is not installed or num_atoms is < 5000 (PBC)
                or < 30000 (non-PBC).
            * knn_cuml_rbc: If device is not CPU, and cuML is installed, and num_atoms is >= 5000 (PBC) or
                >= 30000 (non-PBC).
            * knn_scipy (default): If device is CPU.
            On GPU, for num_atoms ≲ 5000 (PBC) or ≲ 30000 (non-PBC), knn_brute_force is faster than knn_cuml_*,
            but uses more memory. For num_atoms ≳ 5000 (PBC) or ≳ 30000 (non-PBC), knn_cuml_* is faster and uses
            less memory, but requires cuML to be installed. knn_scipy is typically fastest on the CPU.
        system_config: The system configuration to use for graph construction.
        max_num_neighbors: Maximum number of neighbors each node can send messages to.
            If None, will use system_config.max_num_neighbors.
        system_id: Optional index that is relative to a particular dataset.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
            This flag does not affect the resulting graph; it is purely an optimization that can double
            throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can harm
            performance due to additional computation to enforce max_num_neighbors.
        device: The device to put the tensors on.
        output_dtype: The dtype to use for all floating point tensors stored on the AtomGraphs object.
        graph_construction_dtype: The dtype to use for floating point tensors in the graph construction.
    Returns:
        AtomGraphs object
    """
    if isinstance(atoms.pbc, Iterable) and any(atoms.pbc) and not all(atoms.pbc):
        raise NotImplementedError(
            "We do not support periodicity along a subset of axes. Please ensure atoms.pbc is "
            "True/False for all axes and you have padded your systems with sufficient vacuum if necessary."
        )
    output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
    graph_construction_dtype = (
        torch.get_default_dtype()
        if graph_construction_dtype is None
        else graph_construction_dtype
    )
    if output_dtype == torch.float64:
        _check_floating_point_tensors_are_fp64(atoms.info)

    max_num_neighbors = max_num_neighbors or system_config.max_num_neighbors
    atomic_numbers = torch.from_numpy(atoms.numbers).to(torch.long)
    atomic_numbers_embedding = atoms.info.get("node_features", {}).get(
        "atomic_numbers_embedding",
        feat_util.get_atom_embedding(atoms, k_hot=False),
    )
    positions = torch.from_numpy(atoms.positions)
    cell = torch.from_numpy(atoms.cell.array)
    pbc = torch.from_numpy(atoms.pbc)
    lattice = torch.from_numpy(cell_to_cellpar(cell))
    if wrap and (torch.any(cell != 0) and torch.any(pbc)):
        positions = feat_util.map_to_pbc_cell(positions, cell)

    edge_index, edge_vectors, unit_shifts = feat_util.compute_pbc_radius_graph(
        positions=positions,
        cell=cell,
        pbc=pbc,
        radius=system_config.radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
        half_supercell=half_supercell,
        float_dtype=graph_construction_dtype,
        device=device,
    )
    senders, receivers = edge_index[0], edge_index[1]

    node_feats = {
        **atoms.info.get("node_features", {}),
        # NOTE: positions are stored as features on the AtomGraphs,
        # but not actually used as input features to the model.
        "positions": positions,
        "atomic_numbers": atomic_numbers.to(torch.long),
        "atomic_numbers_embedding": atomic_numbers_embedding,
        "atom_identity": torch.arange(len(atoms)).to(torch.long),
    }
    edge_feats = {
        **atoms.info.get("edge_features", {}),
        "vectors": edge_vectors,
        "unit_shifts": unit_shifts,
    }
    graph_feats = {
        **atoms.info.get("graph_features", {}),
        "cell": cell,
        "pbc": pbc,
        "lattice": lattice,
        **_get_charge_and_spin(atoms)
    }

    # Add a batch dimension to non-scalar graph features/targets
    graph_feats = {
        k: v.unsqueeze(0) if v.numel() > 1 else v for k, v in graph_feats.items()
    }
    graph_targets = {
        k: v.unsqueeze(0) if v.numel() > 1 else v
        for k, v in atoms.info.get("graph_targets", {}).items()
    }

    atom_graphs = AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=torch.tensor([len(positions)]),
            n_edge=torch.tensor([len(senders)]),
            node_features=node_feats,
            edge_features=edge_feats,
            system_features=graph_feats,
            node_targets=deepcopy(atoms.info.get("node_targets", {})),
            edge_targets=deepcopy(atoms.info.get("edge_targets", {})),
            system_targets=deepcopy(graph_targets),
            fix_atoms=ase_fix_atoms_to_tensor(atoms),
            tags=_get_ase_tags(atoms),
            radius=system_config.radius,
            max_num_neighbors=torch.tensor([max_num_neighbors]),
            system_id=torch.LongTensor([system_id]) if system_id is not None else system_id
        ).to(device=device, dtype=output_dtype)
    return atom_graphs


def _get_charge_and_spin(atoms: ase.Atoms) -> dict:
    out = {}
    if "charge" in atoms.info or "spin" in atoms.info:
        assert (
            "charge" in atoms.info and "spin" in atoms.info
        ), "Charge and spin must be present together"

        chg, spin = atoms.info["charge"], atoms.info["spin"]
        assert isinstance(chg, (float, int)), "Charge must be a float or int"
        assert isinstance(spin, (float, int)), "Spin must be a float or int"
        out["total_charge"] = torch.tensor([chg], dtype=torch.get_default_dtype())
        out["total_spin"] = torch.tensor([spin], dtype=torch.get_default_dtype())

    return out
    
def _get_ase_tags(atoms: ase.Atoms) -> torch.Tensor:
    """Get tags from ase.Atoms object."""
    tags = atoms.get_tags()
    if tags is not None:
        tags = torch.Tensor(tags)
    else:
        tags = torch.zeros(len(atoms))
    return tags


def ase_fix_atoms_to_tensor(atoms: ase.Atoms) -> Optional[torch.Tensor]:
    """Get fixed atoms from ase.Atoms object."""
    fixed_atoms = None
    if atoms.constraints is not None and len(atoms.constraints) > 0:
        constraint = atoms.constraints[0]
        if isinstance(constraint, constraints.FixAtoms):
            fixed_atoms = torch.zeros((len(atoms)), dtype=torch.bool)
            fixed_atoms[constraint.index] = True
    return fixed_atoms


def _check_floating_point_tensors_are_fp64(obj):
    """Recursively check that all floating point tensors are fp64."""
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        if obj.dtype != torch.float64:
            raise ValueError("All torch tensors stored in atoms.info must be fp64")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_floating_point_tensors_are_fp64(v)
