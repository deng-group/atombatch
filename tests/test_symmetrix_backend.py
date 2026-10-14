import os

import numpy as np
import pytest
from ase import Atoms
from ase.optimize import FIRE
from ase.stress import full_3x3_to_voigt_6_stress

from atombatch import Scheduler, SymmetrixBackend, relax, single_point
from atombatch.symmetrix import GraphBatcher, edge_forces, segment_virials

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
    if request.param == "graph":
        backend._graph = GraphBatcher(symmetrix_calculator)
    elif backend._graph is not None:
        pytest.skip("installed Symmetrix has no calculate_batch")
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
