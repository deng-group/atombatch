import numpy as np
import pytest
from ase.build import add_adsorbate, bulk, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, BFGSLineSearch

from atombatch import (
    Backend,
    Scheduler,
    SequentialBackend,
    molecular_dynamics,
    neb,
    relax,
    single_point,
)


def assert_same(actual, expected):
    # EMT reuses its neighbor list between steps in an independent run, while
    # batched requests are evaluated afresh, so sums differ at rounding level.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


def _crystals(count, seed=0):
    structures = []
    for index in range(count):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
        atoms.numbers[
            np.random.default_rng(seed + index).choice(len(atoms), 4, replace=False)
        ] = 79
        atoms.rattle(0.05, seed=seed + index)
        structures.append(atoms)
    return structures


def _scheduler(**kwargs):
    scheduler = Scheduler(**kwargs)
    scheduler.add_model("emt", EMT())
    return scheduler


def _independent_relax(atoms, optimizer, cell_filter=None):
    atoms = atoms.copy()
    atoms.calc = EMT()
    target = cell_filter(atoms) if cell_filter else atoms
    dynamics = optimizer(target, logfile=None)
    converged = dynamics.run(fmax=0.05, steps=200)
    return atoms, converged, dynamics.nsteps


def test_single_points_are_batched_and_exact():
    structures = _crystals(5)
    scheduler = _scheduler()
    for atoms in structures:
        scheduler.submit(single_point(atoms, ("energy", "forces", "stress")), "emt")

    results = scheduler.run()

    assert scheduler.batch_log == [("emt", 5, 5 * 32)]
    for result, atoms in zip(results, structures):
        assert result.ok
        reference = atoms.copy()
        reference.calc = EMT()
        assert result.value["energy"] == reference.get_potential_energy()
        np.testing.assert_array_equal(result.value["forces"], reference.get_forces())
        np.testing.assert_array_equal(result.value["stress"], reference.get_stress())


@pytest.mark.parametrize("optimizer", [FIRE, BFGS, BFGSLineSearch])
def test_relaxations_match_independent_ase_runs(optimizer):
    structures = _crystals(4)
    scheduler = _scheduler()
    for atoms in structures:
        scheduler.submit(relax(atoms, fmax=0.05, steps=200, optimizer=optimizer), "emt")

    results = scheduler.run()

    assert scheduler.batch_log[0][1] == 4
    for result, atoms in zip(results, structures):
        reference, converged, nsteps = _independent_relax(atoms, optimizer)
        assert result.value["converged"] == converged
        assert result.value["steps"] == nsteps
        assert_same(result.value["atoms"].positions, reference.positions)


def test_cell_relaxation_matches_independent_ase_run():
    structures = _crystals(2)
    structures[1].set_cell(structures[1].cell * 1.03, scale_atoms=True)
    scheduler = _scheduler()
    for atoms in structures:
        scheduler.submit(
            relax(atoms, fmax=0.05, steps=200, cell_filter=FrechetCellFilter), "emt"
        )

    results = scheduler.run()

    # One evaluation per structure and step: forces and stress share a request.
    assert sum(size for _, size, _ in scheduler.batch_log) == sum(
        result.value["steps"] + 1 for result in results
    )
    for result, atoms in zip(results, structures):
        reference, _, nsteps = _independent_relax(atoms, FIRE, FrechetCellFilter)
        assert result.value["steps"] == nsteps
        assert result.value["stress"].shape == (6,)
        assert_same(result.value["atoms"].cell.array, reference.cell.array)
        assert_same(result.value["atoms"].positions, reference.positions)


def _adatom_hop():
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    initial = slab.copy()
    final = slab.copy()
    final[-1].x += final.get_cell()[0, 0] / 2
    for endpoint in (initial, final):
        endpoint.calc = EMT()
        BFGS(endpoint, logfile=None).run(fmax=0.02)
        endpoint.calc = None
    return [initial] + [initial.copy() for _ in range(3)] + [final]


@pytest.mark.parametrize(
    "neb_kwargs", [{}, {"climb": True}, {"remove_rotation_and_translation": True}]
)
def test_neb_matches_independent_ase_neb(neb_kwargs):
    images = _adatom_hop()
    scheduler = _scheduler()
    scheduler.submit(
        neb(images, fmax=0.05, steps=100, neb_kwargs=neb_kwargs, interpolate="linear"),
        "emt",
    )
    scheduler.submit(
        neb(images, fmax=0.05, steps=100, neb_kwargs=neb_kwargs, interpolate="idpp"),
        "emt",
    )

    results = scheduler.run()

    for result, method in zip(results, ("linear", "idpp")):
        reference_images = [image.copy() for image in images]
        for image in reference_images:
            image.calc = EMT()
        band = NEB(reference_images, **neb_kwargs)
        band.interpolate(method=method)
        dynamics = FIRE(band, logfile=None)
        converged = dynamics.run(fmax=0.05, steps=100)
        assert result.ok, result.error
        assert result.value["converged"] == converged
        assert result.value["steps"] == dynamics.nsteps
        for image, expected in zip(result.value["images"], reference_images):
            assert_same(image.positions, expected.positions)
    # Both bands' moving images share each evaluation.
    assert max(size for _, size, _ in scheduler.batch_log) >= 6


def _thermalized(seed):
    atoms = _crystals(1, seed=seed)[0]
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=600, rng=np.random.default_rng(seed)
    )
    return atoms


@pytest.mark.parametrize("integrator", ["verlet", "langevin"])
def test_molecular_dynamics_matches_independent_ase_runs(integrator):
    def options(seed):
        if integrator == "verlet":
            return VelocityVerlet, {"timestep": 2.0}
        return Langevin, {
            "timestep": 2.0,
            "temperature_K": 600,
            "friction": 0.02,
            "rng": np.random.default_rng(100 + seed),
        }

    scheduler = _scheduler()
    for seed in range(3):
        dynamics, kwargs = options(seed)
        scheduler.submit(
            molecular_dynamics(_thermalized(seed), dynamics, 20, **kwargs), "emt"
        )

    results = scheduler.run()

    assert all(size == 3 for _, size, _ in scheduler.batch_log)
    for seed, result in enumerate(results):
        dynamics, kwargs = options(seed)
        reference = _thermalized(seed)
        reference.calc = EMT()
        dynamics(reference, logfile=None, **kwargs).run(20)
        assert_same(result.value["atoms"].positions, reference.positions)
        assert_same(result.value["atoms"].get_velocities(), reference.get_velocities())


def test_mixed_tasks_share_batches():
    scheduler = _scheduler()
    scheduler.submit(single_point(_crystals(1)[0]), "emt", name="point")
    scheduler.submit(relax(_crystals(1, seed=3)[0], steps=50), "emt", name="relax")
    scheduler.submit(
        neb(_adatom_hop(), steps=20, interpolate="linear"), "emt", name="neb"
    )
    scheduler.submit(
        molecular_dynamics(_thermalized(7), VelocityVerlet, 30, timestep=2.0),
        "emt",
        name="md",
    )

    results = scheduler.run()

    assert [result.name for result in results] == ["point", "relax", "neb", "md"]
    assert all(result.ok for result in results)
    # First round: one single point, one relaxation, five NEB images, one MD state.
    assert scheduler.batch_log[0][1] == 8


def test_refill_keeps_at_most_max_active_tasks_running():
    structures = _crystals(6)
    scheduler = _scheduler(max_active=2)
    for index, atoms in enumerate(structures):
        scheduler.submit(relax(atoms, steps=5 + 10 * index), "emt")

    results = scheduler.run()

    assert all(result.ok for result in results)
    assert max(size for _, size, _ in scheduler.batch_log) == 2
    assert sum(size for _, size, _ in scheduler.batch_log) == sum(
        result.value["steps"] + 1 for result in results
    )


def test_max_atoms_splits_batches():
    scheduler = Scheduler()
    scheduler.add_model("emt", EMT(), max_atoms=64)
    for atoms in _crystals(5):
        scheduler.submit(single_point(atoms), "emt")

    scheduler.run()

    assert [size for _, size, _ in scheduler.batch_log] == [2, 2, 1]
    assert all(atoms <= 64 for _, _, atoms in scheduler.batch_log)


def test_models_are_batched_separately():
    scheduler = Scheduler()
    scheduler.add_model("a", EMT())
    scheduler.add_model("b", SequentialBackend(EMT()))
    for atoms in _crystals(3):
        scheduler.submit(single_point(atoms), "a")
    for atoms in _crystals(2, seed=10):
        scheduler.submit(single_point(atoms), "b")

    scheduler.run()

    assert sorted(scheduler.batch_log) == [("a", 3, 96), ("b", 2, 64)]


def test_task_and_backend_errors_stay_with_their_tasks():
    class Failing(Backend):
        def evaluate(self, structures, properties):
            raise RuntimeError("model failure")

    def broken(context):
        raise ValueError("task failure")

    scheduler = _scheduler()
    scheduler.add_model("failing", Failing())
    scheduler.submit(broken, "emt", name="broken")
    scheduler.submit(single_point(_crystals(1)[0]), "failing", name="failing")
    scheduler.submit(relax(_crystals(1)[0], steps=10), "emt", name="healthy")

    broken_result, failing_result, healthy_result = scheduler.run()

    assert isinstance(broken_result.error, ValueError)
    assert isinstance(failing_result.error, RuntimeError)
    assert healthy_result.ok


def test_unknown_model_is_rejected():
    scheduler = _scheduler()
    with pytest.raises(KeyError, match="Unknown model"):
        scheduler.submit(single_point(_crystals(1)[0]), "missing")
