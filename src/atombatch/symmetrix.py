"""Batched evaluation for Symmetrix releases without ``calculate_batch``.

The structures of a batch are joined into one disconnected neighbor graph and
evaluated by a single native call of the Symmetrix evaluator. Message passing
runs only along graph edges, so the structures do not interact. Energies are
summed per structure from node energies, forces are reduced per atom, and
stresses are reduced per structure from the directed pair forces.

Adapted from ``symmetrix.batch`` of Symmetrix-XL (MIT License), which provides
``Symmetrix.calculate_batch`` in later releases. It uses only evaluator calls
present since Symmetrix-XL 0.1.1.
"""

import hashlib
import warnings
from collections import OrderedDict

import numpy as np
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ._edges import edge_geometry

SOURCE_KEY = "atombatch_source"


def supports_graph_batching(calculator):
    """Return whether ``calculator`` exposes the evaluator calls used here."""
    return all(
        hasattr(calculator, name) for name in ("evaluator", "cutoff", "_compute_mace")
    )


class GraphBatcher:
    """Evaluate structures with one Symmetrix calculator in one native call.

    Neighbor lists are cached by positions, cell, and periodicity, so
    structures that differ only in species, such as Monte Carlo trials on one
    lattice, share a neighbor list.
    """

    def __init__(self, calculator, neighbor_cache_size=64, trajectory_cache_size=1024):
        self.calculator = calculator
        self.neighbor_cache_size = int(neighbor_cache_size)
        self._neighbor_cache = OrderedDict()
        self.trajectory_cache_size = int(trajectory_cache_size)
        self._trajectory_caches = OrderedDict()
        numbers = [int(value) for value in calculator.evaluator.atomic_numbers]
        self._type_lookup = np.full(max(numbers) + 1, -1, dtype=np.int64)
        self._type_lookup[numbers] = np.arange(len(numbers))
        self.unsupported = _unsupported_reason(calculator)
        if uses_ase_neighbor_list():
            warnings.warn(
                "Symmetrix is using ASE's Python neighbor list because matscipy "
                "is not installed. Batched evaluation rebuilds neighbor lists "
                "for every structure, which then dominates the run time, and "
                "FP32 results can differ in rounding from the Symmetrix "
                "calculator. Install matscipy (pip install matscipy).",
                stacklevel=3,
            )

    def evaluate(self, structures, properties):
        structures = list(structures)
        compute_stress = "stress" in properties
        if self.unsupported:
            return self._sequential(structures, properties)
        if compute_stress:
            volumes = np.array([abs(atoms.cell.volume) for atoms in structures])
            if not np.all(volumes > 0.0):
                raise ValueError("Stress requires three-dimensional cells.")
        totals, energies, forces, virials = self._native(structures, compute_stress)
        results = []
        for index, energy in enumerate(totals):
            result = {
                "energy": float(energy),
                "free_energy": float(energy),
                "energies": energies[index],
                "forces": forces[index],
            }
            if compute_stress:
                result["stress"] = full_3x3_to_voigt_6_stress(
                    virials[index] / volumes[index]
                )
            results.append(result)
        return results

    def _sequential(self, structures, properties):
        # Dispersion corrections and field-coupled models need the ASE path.
        results = []
        for atoms in structures:
            self.calculator.calculate(atoms.copy(), list(properties), all_changes)
            results.append(
                {
                    key: np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in self.calculator.results.items()
                }
            )
        return results

    def _native(self, structures, compute_stress):
        calculator = self.calculator
        evaluator = calculator.evaluator
        atom_counts = np.array([len(atoms) for atoms in structures], dtype=np.int64)
        atom_offsets = np.concatenate(([0], np.cumsum(atom_counts)))
        num_nodes = int(atom_offsets[-1])
        graphs = [self._neighbors(atoms) for atoms in structures]
        edge_offsets = np.concatenate(
            ([0], np.cumsum([len(graph[0]) for graph in graphs]))
        ).astype(np.int64)
        i_list = np.concatenate(
            [graph[0] + offset for graph, offset in zip(graphs, atom_offsets)]
        ).astype(np.int32)
        j_list = np.concatenate(
            [graph[1] + offset for graph, offset in zip(graphs, atom_offsets)]
        ).astype(np.int32)
        r = np.concatenate([graph[2] for graph in graphs])
        xyz = np.concatenate([graph[3] for graph in graphs])
        node_types = self._node_types(
            np.concatenate([atoms.numbers for atoms in structures])
        )
        num_neigh = np.bincount(i_list, minlength=num_nodes)

        properties = ["energy", "forces"] + (["stress"] if compute_stress else [])
        tuning = getattr(calculator, "_apply_kernel_launch_tuning", None)
        if callable(tuning):
            tuning(num_nodes, len(i_list), properties)
        mace_inputs = (
            num_nodes,
            node_types.tolist(),
            num_neigh,
            j_list,
            node_types[j_list].tolist(),
            xyz,
            r,
            i_list,
        )
        generation = calculator._compute_mace(mace_inputs, atoms=None)

        # Graph-wide observables accumulate in FP64, as in the ASE calculator.
        node_energies = np.array(evaluator.node_energies, dtype=np.float64)[
            :num_nodes
        ].copy()
        pair_forces = None
        reduce_atom_forces = getattr(evaluator, "_reduce_atom_forces", None)
        if callable(reduce_atom_forces):
            forces = np.asarray(
                reduce_atom_forces(num_nodes, i_list, j_list, generation),
                dtype=np.float64,
            ).reshape(num_nodes, 3)
        else:
            pair_forces = edge_forces(evaluator, len(i_list))
            forces = np.stack(
                [
                    np.bincount(j_list, pair_forces[:, c], minlength=num_nodes)
                    - np.bincount(i_list, pair_forces[:, c], minlength=num_nodes)
                    for c in range(3)
                ],
                axis=1,
            )
        virials = None
        if compute_stress:
            reduce_segmented = getattr(evaluator, "_reduce_segmented_stress", None)
            if callable(reduce_segmented):
                # The native reduction divides by a volume; unit volumes give
                # virials.
                virials = np.asarray(
                    reduce_segmented(np.ones(len(structures)), edge_offsets, xyz),
                    dtype=np.float64,
                ).reshape(len(structures), 3, 3)
            else:
                if pair_forces is None:
                    pair_forces = edge_forces(evaluator, len(i_list))
                virials = segment_virials(pair_forces, xyz, edge_offsets)
        totals = np.bincount(
            np.repeat(np.arange(len(structures)), atom_counts),
            weights=node_energies,
            minlength=len(structures),
        )
        energies = np.split(node_energies, atom_offsets[1:-1])
        forces = np.split(forces, atom_offsets[1:-1])
        return totals, energies, forces, virials

    def _neighbors(self, atoms):
        """Return receiver-major ``(i, j, r, xyz)`` edges of one structure."""
        source = atoms.info.get(SOURCE_KEY)
        if source is not None and getattr(self.calculator, "neighbor_skin", 0.0) > 0:
            return self._trajectory_neighbors(atoms, source)
        from symmetrix.calculator import (
            _receiver_major_neighbor_arrays,
            neighbor_list,
        )

        digest = hashlib.blake2b(digest_size=16)
        for array in (atoms.positions, atoms.cell.array, atoms.pbc):
            digest.update(np.ascontiguousarray(array).tobytes())
        key = digest.digest()
        cached = self._neighbor_cache.get(key)
        if cached is not None:
            self._neighbor_cache.move_to_end(key)
            return cached
        cutoff = self.calculator.cutoff
        if not np.any(atoms.pbc) and atoms.cell.volume == 0.0:
            # A non-periodic cell only bounds the neighbor search.
            atoms = atoms.copy()
            atoms.set_cell(np.ptp(atoms.positions, axis=0) + 2.0 * cutoff + 1.0)
        receivers, sources, shifts = neighbor_list("ijS", atoms, cutoff)
        receivers, sources, shifts = _receiver_major_neighbor_arrays(
            receivers, sources, shifts
        )
        positions = np.asarray(atoms.positions, dtype=np.float64)
        xyz = np.ascontiguousarray(
            positions[sources] - positions[receivers] + shifts @ atoms.cell.array
        )
        cached = (receivers, sources, np.linalg.norm(xyz, axis=1), xyz)
        if self.neighbor_cache_size > 0:
            self._neighbor_cache[key] = cached
            while len(self._neighbor_cache) > self.neighbor_cache_size:
                self._neighbor_cache.popitem(last=False)
        return cached

    def _trajectory_neighbors(self, atoms, source):
        """Return edges from the Symmetrix neighbor cache of one trajectory.

        Each trajectory (one batched calculator) keeps its own Symmetrix
        neighbor cache, which is rebuilt with ``cutoff + neighbor_skin`` only
        when atoms have moved beyond the skin. Symmetrix decides whether to
        reuse or rebuild the cache; the edge geometry of the step is then
        computed by :func:`atombatch._edges.edge_geometry`. The edges are
        those the Symmetrix ASE calculator would use for the same trajectory.
        """
        calculator = self.calculator
        saved = getattr(calculator, "_neighbor_cache", None)
        previous = self._trajectory_caches.pop(source, None)
        calculator._neighbor_cache = previous
        try:
            generation = None if previous is None else previous.geometry_generation
            # Called with native_geometry=True, Symmetrix updates the cache and
            # returns before computing the edge geometry on the host.
            edges = calculator._cached_neighbor_geometry(
                atoms, native_geometry=True, allow_device_neighbor_graph=False
            )
            cache = calculator._neighbor_cache
            if edges[2] is None:
                edges = self._edge_geometry(atoms, cache, previous, generation)
            self._trajectory_caches[source] = cache
        finally:
            calculator._neighbor_cache = saved
        while len(self._trajectory_caches) > self.trajectory_cache_size:
            self._trajectory_caches.popitem(last=False)
        i_list, j_list, r, xyz = edges
        return (
            np.asarray(i_list),
            np.asarray(j_list),
            np.asarray(r, dtype=np.float64),
            np.asarray(xyz, dtype=np.float64),
        )

    def _edge_geometry(self, atoms, cache, previous, generation):
        """Return the edges of a cache that Symmetrix has just updated."""
        from symmetrix.calculator import _PREPARED_EXECUTION_MODES

        calculator = self.calculator
        positions = np.asarray(atoms.positions, dtype=float)
        if cache is previous and cache.geometry_generation == generation:
            # Reused without a geometry reset: the displacements since the
            # reference positions, as Symmetrix computes them.
            pbc = np.asarray(atoms.pbc, dtype=bool)
            reference = (
                cache.reference_positions
                if np.all(pbc) and cache.fractional_geometry_eligible
                else cache.topology_reference_positions
            )
            displacements = calculator._minimum_image_displacements(
                positions, reference, np.asarray(atoms.cell, dtype=float), pbc
            )
        else:
            # Built or reset in this call: the reference is the current state.
            displacements = np.zeros_like(positions)
        calculator._materialize_neighbor_cache_geometry()
        clamp = (
            getattr(calculator.evaluator, "streamed_edges_mode", None)
            in _PREPARED_EXECUTION_MODES
        )
        return edge_geometry(
            cache.reference_xyz,
            cache.sources,
            cache.receivers,
            displacements,
            calculator.cutoff,
            clamp,
        )

    def _node_types(self, numbers):
        numbers = np.asarray(numbers, dtype=np.int64)
        known = (numbers >= 0) & (numbers < len(self._type_lookup))
        types = np.full(len(numbers), -1, dtype=np.int64)
        types[known] = self._type_lookup[numbers[known]]
        if np.any(types < 0):
            unsupported = sorted(set(numbers[types < 0].tolist()))
            supported = np.flatnonzero(self._type_lookup >= 0).tolist()
            raise ValueError(
                f"Model does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers are {supported}."
            )
        return types


def uses_ase_neighbor_list():
    """Return whether Symmetrix fell back to ``ase.neighborlist``."""
    from symmetrix import calculator

    return calculator.neighbor_list.__module__.startswith("ase.")


def edge_forces(evaluator, num_edges):
    """Return the directed pair forces of the last evaluation."""
    forces = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(-1, 3)
    return forces[:num_edges].copy()


def segment_virials(pair_forces, xyz, edge_offsets):
    """Return the virial ``-sum(F_e outer xyz_e)`` of every edge segment."""
    count = len(edge_offsets) - 1
    owner = np.repeat(np.arange(count), np.diff(edge_offsets))
    products = pair_forces[:, :, None] * xyz[:, None, :]
    virials = np.empty((count, 3, 3))
    for row in range(3):
        for column in range(3):
            virials[:, row, column] = -np.bincount(
                owner, weights=products[:, row, column], minlength=count
            )
    return virials


def _unsupported_reason(calculator):
    field = getattr(calculator, "_has_native_field_coupling", None)
    if callable(field) and field():
        return "electric-field (MACEField) models"
    if getattr(calculator, "_dispersion_calculator", None) is not None:
        return "dispersion corrections"
    fixed = getattr(calculator, "_fixed_workspace_plan_selected", None)
    if callable(fixed) and fixed():
        return "fixed-workspace tiled plans"
    return None
