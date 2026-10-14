import os

import numpy as np
import pytest
from ase import Atoms
from ase.optimize import FIRE
from ase.stress import full_3x3_to_voigt_6_stress

from atombatch import Scheduler, SymmetrixBackend, relax, single_point
from atombatch.symmetrix import (
    SOURCE_KEY,
    GraphBatcher,
    edge_forces,
    segment_virials,
)

MODEL = os.environ.get("ATOMBATCH_SYMMETRIX_MODEL")


@pytest.fixture(scope="module")
def symmetrix_calculator():
    if not MODEL:
        pytest.skip("set ATOMBATCH_SYMMETRIX_MODEL to a Symmetrix JSON model")
    symmetrix = pytest.importorskip("symmetrix")
    return symmetrix.Symmetrix(MODEL, dtype="float64")


@pytest.fixture(params=["calculate_batch", "graph"])
def backend(request, symmetrix_calculator):
    """Both batched paths: Symmetrix's own, and atombatch's for 0.1.1."""
    backend = SymmetrixBackend(symmetrix_calculator)
    if request.param == "calculate_batch":
        if not callable(getattr(symmetrix_calculator, "calculate_batch", None)):
            pytest.skip("installed Symmetrix has no calculate_batch")
        backend._graph = None
    return backend


def _perovskite(seed):
    a = 3.95
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=seed)
    return atoms


def test_symmetrix_calculator_selects_the_symmetrix_backend(symmetrix_calculator):
    scheduler = Scheduler()
    scheduler.add_model("sx", symmetrix_calculator)
    assert isinstance(scheduler._backends["sx"], SymmetrixBackend)


def test_batched_symmetrix_matches_its_ase_calculator(backend, symmetrix_calculator):
    scheduler = Scheduler()
    scheduler.add_model("sx", backend)
    structures = [_perovskite(seed) for seed in range(4)]
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "sx")

    results = scheduler.run()

    assert scheduler.batch_log == [("sx", 4, 160)]
    for result, atoms in zip(results, structures):
        reference = atoms.copy()
        reference.calc = symmetrix_calculator
        np.testing.assert_allclose(
            result.value["energy"], reference.get_potential_energy(), atol=1e-9
        )
        np.testing.assert_allclose(
            result.value["forces"], reference.get_forces(), atol=1e-9
        )
        np.testing.assert_allclose(
            result.value["stress"], reference.get_stress(), atol=1e-10
        )


def test_symmetrix_relaxation_matches_independent_run(backend, symmetrix_calculator):
    structures = [_perovskite(seed) for seed in range(3)]
    scheduler = Scheduler()
    scheduler.add_model("sx", backend)
    for atoms in structures:
        scheduler.submit(relax(atoms, fmax=0.05, steps=60), "sx")

    results = scheduler.run()

    for result, atoms in zip(results, structures):
        reference = atoms.copy()
        reference.calc = symmetrix_calculator
        dynamics = FIRE(reference, logfile=None)
        converged = dynamics.run(fmax=0.05, steps=60)
        assert result.value["converged"] == converged
        assert result.value["steps"] == dynamics.nsteps
        np.testing.assert_allclose(
            result.value["atoms"].positions, reference.positions, atol=1e-8
        )


def test_pair_force_stress_matches_its_ase_calculator(symmetrix_calculator):
    # Symmetrix-XL 0.1.1 has no native per-structure stress reduction, so the
    # graph path reduces stresses from the pair forces in NumPy.
    batcher = GraphBatcher(symmetrix_calculator)
    structures = [_perovskite(seed) for seed in range(3)]
    batcher.evaluate(structures, ("energy", "forces"))
    graphs = [batcher._neighbors(atoms) for atoms in structures]
    offsets = np.concatenate(([0], np.cumsum([len(g[0]) for g in graphs])))
    xyz = np.concatenate([graph[3] for graph in graphs])

    virials = segment_virials(
        edge_forces(symmetrix_calculator.evaluator, len(xyz)), xyz, offsets
    )

    for virial, atoms in zip(virials, structures):
        reference = atoms.copy()
        reference.calc = symmetrix_calculator
        np.testing.assert_allclose(
            full_3x3_to_voigt_6_stress(virial / atoms.get_volume()),
            reference.get_stress(),
            atol=1e-10,
        )


def test_trajectory_neighbor_cache_matches_the_calculator(symmetrix_calculator):
    # A relaxation reuses one neighbor cache per trajectory, as the Symmetrix
    # ASE calculator does, so the edges match its own at every step.
    from symmetrix import Symmetrix

    batcher = GraphBatcher(symmetrix_calculator)
    reference = Symmetrix(MODEL, dtype="float64")
    atoms = _perovskite(0)
    rng = np.random.default_rng(0)
    rebuilds = reference.neighbor_cache_reuse_count
    for step in range(6):
        snapshot = atoms.copy()
        snapshot.info[SOURCE_KEY] = 7
        cached = batcher._neighbors(snapshot)
        expected = reference._cached_neighbor_geometry(
            atoms, native_geometry=False, allow_device_neighbor_graph=False
        )
        for got, want in zip(cached, expected):
            np.testing.assert_array_equal(got, want)
        # Small moves reuse the cache; the cumulative moves exceed half the
        # skin and force a rebuild.
        atoms.positions += rng.normal(scale=0.06, size=atoms.positions.shape)
    reused = reference.neighbor_cache_reuse_count - rebuilds
    assert 0 < reused < 6
    assert symmetrix_calculator._neighbor_cache is not batcher._trajectory_caches[7]


@pytest.mark.parametrize("compiled", [True, False])
def test_trajectory_edges_equal_the_calculator_bit_for_bit(
    symmetrix_calculator, compiled, monkeypatch
):
    # Reused, rebuilt, and cell-reset caches all give the calculator's edges.
    from symmetrix import Symmetrix

    from atombatch import _edges

    if not compiled:
        monkeypatch.setattr(_edges, "numba", None)
    batcher = GraphBatcher(Symmetrix(MODEL, dtype="float64"))
    reference = Symmetrix(MODEL, dtype="float64")
    rng = np.random.default_rng(1)
    atoms = _perovskite(3)
    for step in range(30):
        snapshot = atoms.copy()
        snapshot.info[SOURCE_KEY] = 0
        got = batcher._neighbors(snapshot)
        want = reference._cached_neighbor_geometry(
            atoms, native_geometry=False, allow_device_neighbor_graph=False
        )
        for value, expected in zip(got, want):
            np.testing.assert_array_equal(value, expected)
        kind = step % 10
        if kind < 6:
            atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
        elif kind < 8:
            atoms.set_cell(atoms.cell * (1 + rng.normal(scale=0.003)), scale_atoms=True)
        else:
            atoms.positions += rng.normal(scale=0.15, size=atoms.positions.shape)
    assert reference.neighbor_cache_build_count > 1
    assert reference.neighbor_cache_reuse_count > 10
