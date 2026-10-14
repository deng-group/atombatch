import os

import numpy as np
import pytest
from ase import Atoms
from ase.optimize import FIRE

from atombatch import Scheduler, SymmetrixBackend, relax, single_point

MODEL = os.environ.get("ATOMBATCH_SYMMETRIX_MODEL")


@pytest.fixture(scope="module")
def symmetrix_calculator():
    if not MODEL:
        pytest.skip("set ATOMBATCH_SYMMETRIX_MODEL to a Symmetrix JSON model")
    symmetrix = pytest.importorskip("symmetrix")
    calculator = symmetrix.Symmetrix(MODEL, dtype="float64")
    if not callable(getattr(calculator, "calculate_batch", None)):
        pytest.skip("installed Symmetrix has no batched evaluation")
    return calculator


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


def test_symmetrix_backend_is_selected_and_exact(symmetrix_calculator):
    scheduler = Scheduler()
    scheduler.add_model("sx", symmetrix_calculator)
    assert isinstance(scheduler._backends["sx"], SymmetrixBackend)
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


def test_symmetrix_relaxation_matches_independent_run(symmetrix_calculator):
    structures = [_perovskite(seed) for seed in range(3)]
    scheduler = Scheduler()
    scheduler.add_model("sx", symmetrix_calculator)
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
