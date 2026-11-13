"""MACE-POLAR-1 through Symmetrix-XL 0.2.

Polar models have long-range electrostatics, so atombatch never joins them
into one graph. Where Symmetrix provides ``calculate_polar_batch`` they are
evaluated in shared native calls, each structure with its own cell and graph
inputs; otherwise one at a time through the calculator. These tests compare
every path with the calculator alone. They run only when
``ATOMBATCH_SYMMETRIX_POLAR_MODEL`` names an exported MACE-POLAR-1 JSON model.
"""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.optimize import FIRE

from atombatch import Scheduler, molecular_dynamics, relax, single_point

MODEL = os.environ.get("ATOMBATCH_SYMMETRIX_POLAR_MODEL")


@pytest.fixture(scope="module")
def symmetrix():
    if not MODEL:
        pytest.skip("set ATOMBATCH_SYMMETRIX_POLAR_MODEL to a MACE-POLAR-1 JSON model")
    return pytest.importorskip("symmetrix")


def _calculator(symmetrix):
    try:
        calculator = symmetrix.Symmetrix(MODEL, dtype="float64")
    except ValueError as error:
        if "model_type" not in str(error):
            raise
        pytest.skip(f"installed Symmetrix-XL cannot load MACE-POLAR-1 ({error})")
    if getattr(calculator, "_native_model_type", None) != "MACE_Polar":
        pytest.skip("the model is not a MACE-POLAR-1 model")
    return calculator


def _water_box(seed):
    rng = np.random.default_rng(seed)
    atoms = Atoms()
    for _ in range(8):
        water = molecule("H2O")
        water.rotate(rng.uniform(0, 360), "z")
        water.translate(rng.uniform(0, 6, 3))
        atoms += water
    atoms.set_cell([7.0, 7.0, 7.0])
    atoms.pbc = True
    atoms.wrap()
    return atoms


def _close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def _reference(atoms, symmetrix):
    reference = atoms.copy()
    reference.calc = _calculator(symmetrix)
    return reference


def _batches(calculator):
    return getattr(calculator.evaluator, "polar_whole_model_batch_count", 0)


def test_polar_models_are_batched_natively_and_exactly(symmetrix):
    calculator = _calculator(symmetrix)
    scheduler = Scheduler()
    scheduler.add_model("polar", calculator)
    graph = scheduler._backends["polar"]._graph
    assert graph.unsupported == "MACE_Polar models"
    structures = [_water_box(seed) for seed in range(3)]
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "polar")
    before = _batches(calculator)

    results = scheduler.run()

    assert scheduler.batch_log == [("polar", 3, sum(map(len, structures)))]
    if graph.polar_batch:
        assert _batches(calculator) - before == 1

    for atoms, result in zip(structures, results):
        reference = _reference(atoms, symmetrix)
        _close(result.value["energy"], reference.get_potential_energy())
        _close(result.value["forces"], reference.get_forces())
        _close(result.value["stress"], reference.get_stress(), atol=1e-10)


def test_polar_batches_add_dispersion_corrections(symmetrix):
    corrections = pytest.importorskip("symmetrix.corrections")
    calculator = symmetrix.Symmetrix(
        MODEL, dtype="float64", corrections=[corrections.D3Correction()]
    )
    structures = [_water_box(seed) for seed in range(2)]
    scheduler = Scheduler()
    scheduler.add_model("polar", calculator)
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "polar")

    results = scheduler.run()

    for atoms, result in zip(structures, results):
        reference = atoms.copy()
        reference.calc = symmetrix.Symmetrix(
            MODEL, dtype="float64", corrections=[corrections.D3Correction()]
        )
        _close(result.value["energy"], reference.get_potential_energy())
        _close(result.value["forces"], reference.get_forces())
        _close(result.value["stress"], reference.get_stress(), atol=1e-10)


def test_polar_charge_and_dipole_follow_each_structure(symmetrix):
    neutral = molecule("H2O")
    cation = molecule("H2O")
    cation.info["total_charge"] = 1.0

    def task(atoms):
        def run(context):
            image = atoms.copy()
            image.calc = context.calculator()
            return image.get_potential_energy(), image.get_dipole_moment()

        return run

    scheduler = Scheduler()
    scheduler.add_model("polar", _calculator(symmetrix))
    for atoms in (neutral, cation):
        scheduler.submit(task(atoms), "polar")

    results = scheduler.run()

    for atoms, result in zip((neutral, cation), results):
        energy, dipole = result.value
        reference = _reference(atoms, symmetrix)
        _close(energy, reference.get_potential_energy())
        _close(dipole, reference.get_dipole_moment())
    assert abs(results[0].value[0] - results[1].value[0]) > 0.1


def test_polar_relaxation_and_md_match_independent_runs(symmetrix):
    structures = [_water_box(seed) for seed in range(2)]
    scheduler = Scheduler()
    scheduler.add_model("polar", _calculator(symmetrix))
    for atoms in structures:
        scheduler.submit(relax(atoms, fmax=0.1, steps=8), "polar")
    for seed, atoms in enumerate(structures):
        scheduler.submit(
            molecular_dynamics(
                atoms,
                Langevin,
                5,
                timestep=0.5,
                temperature_K=300,
                friction=0.01,
                rng=np.random.default_rng(seed),
            ),
            "polar",
        )

    results = scheduler.run()

    for atoms, result in zip(structures, results[:2]):
        reference = _reference(atoms, symmetrix)
        dynamics = FIRE(reference, logfile=None)
        dynamics.run(fmax=0.1, steps=8)
        assert result.value["steps"] == dynamics.nsteps
        _close(result.value["atoms"].positions, reference.positions, atol=1e-8)
    for seed, (atoms, result) in enumerate(zip(structures, results[2:])):
        reference = _reference(atoms, symmetrix)
        Langevin(
            reference,
            timestep=0.5,
            temperature_K=300,
            friction=0.01,
            rng=np.random.default_rng(seed),
            logfile=None,
        ).run(5)
        _close(result.value["atoms"].positions, reference.positions, atol=1e-8)
