import os

import numpy as np
import pytest
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from atombatch import Scheduler, TorchSimBackend, relax, single_point

MODEL = os.environ.get("ATOMBATCH_MACE_MODEL")


@pytest.fixture(scope="module")
def mace():
    if not MODEL:
        pytest.skip("set ATOMBATCH_MACE_MODEL to a MACE-MP model name or path")
    torch = pytest.importorskip("torch")
    foundations = pytest.importorskip("mace.calculators.foundations_models")
    torch_sim_mace = pytest.importorskip("torch_sim.models.mace")
    options = {"model": MODEL, "default_dtype": "float64", "device": "cpu"}
    model = torch_sim_mace.MaceModel(
        foundations.mace_mp(return_raw_model=True, **options),
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_stress=True,
    )
    return model, foundations.mace_mp(**options)


def _salts():
    structures = []
    for index, repeat in enumerate([(1, 1, 1), (2, 1, 1), (1, 2, 2)]):
        atoms = bulk("NaCl", "rocksalt", a=5.6, cubic=True).repeat(repeat)
        atoms.rattle(0.05, seed=index)
        structures.append(atoms)
    return structures


def test_batched_mace_matches_its_ase_calculator(mace):
    model, calculator = mace
    structures = _salts()
    scheduler = Scheduler()
    scheduler.add_model("mace", model)
    assert isinstance(scheduler._backends["mace"], TorchSimBackend)
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "mace")

    results = scheduler.run()

    assert [size for _, size, _ in scheduler.batch_log] == [3]
    for result, atoms in zip(results, structures):
        reference = atoms.copy()
        reference.calc = calculator
        np.testing.assert_allclose(
            result.value["energy"], reference.get_potential_energy(), atol=1e-9
        )
        np.testing.assert_allclose(
            result.value["forces"], reference.get_forces(), atol=1e-9
        )
        np.testing.assert_allclose(
            result.value["stress"], reference.get_stress(), atol=1e-9
        )


def test_batched_mace_cell_relaxations_match_independent_runs(mace):
    model, calculator = mace
    structures = _salts()
    scheduler = Scheduler()
    scheduler.add_model("mace", model)
    for atoms in structures:
        scheduler.submit(
            relax(atoms, fmax=0.02, steps=200, cell_filter=FrechetCellFilter), "mace"
        )

    results = scheduler.run()

    for result, atoms in zip(results, structures):
        reference = atoms.copy()
        reference.calc = calculator
        dynamics = FIRE(FrechetCellFilter(reference), logfile=None)
        converged = dynamics.run(fmax=0.02, steps=200)
        assert result.value["converged"] == converged
        assert result.value["steps"] == dynamics.nsteps
        np.testing.assert_allclose(
            result.value["atoms"].positions, reference.positions, atol=1e-9
        )
        np.testing.assert_allclose(
            result.value["atoms"].cell.array, reference.cell.array, atol=1e-9
        )
