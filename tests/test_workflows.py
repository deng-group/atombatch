from itertools import combinations

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState

from atombatch import (
    Scheduler,
    displacement_forces,
    equation_of_state,
    swap_monte_carlo,
)


def _scheduler(model=None):
    scheduler = Scheduler()
    scheduler.add_model("emt", EMT() if model is None else model)
    return scheduler


def _alloy(seed=0, repeat=(2, 2, 2), gold=4):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(repeat)
    rng = np.random.default_rng(seed)
    atoms.numbers[rng.choice(len(atoms), gold, replace=False)] = 79
    return atoms


def _emt(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    return atoms


def test_evaluate_batches_independent_structures_in_one_call():
    structures = [_alloy(seed) for seed in range(5)]
    scheduler = _scheduler()

    def task(context):
        return context.evaluate(structures, properties=("energy", "forces", "stress"))

    scheduler.submit(task, "emt")

    (result,) = scheduler.run()

    assert scheduler.batch_log == [("emt", 5, 5 * 32)]
    for values, atoms in zip(result.value, structures):
        reference = _emt(atoms)
        assert values["energy"] == reference.get_potential_energy()
        np.testing.assert_array_equal(values["forces"], reference.get_forces())
        np.testing.assert_array_equal(values["stress"], reference.get_stress())


def test_equation_of_state_matches_ase_fit():
    atoms = bulk("Cu", "fcc", a=3.6)
    scales = np.linspace(0.96, 1.04, 9)
    scheduler = _scheduler()
    scheduler.submit(equation_of_state(atoms, scales), "emt")

    (result,) = scheduler.run()

    assert scheduler.batch_log == [("emt", 9, 9)]
    volumes, energies = [], []
    for factor in scales:
        image = _emt(atoms)
        image.set_cell(atoms.cell * factor, scale_atoms=True)
        volumes.append(image.get_volume())
        energies.append(image.get_potential_energy())
    expected = EquationOfState(volumes, energies, "birchmurnaghan").fit()
    np.testing.assert_allclose(
        [result.value["volume"], result.value["energy"], result.value["bulk_modulus"]],
        expected,
    )


def _l12_unit_cell():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.numbers[0] = 79
    return atoms


def _phonopy_atoms(atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell[:],
        scaled_positions=atoms.get_scaled_positions(),
    )


def _sequential_forces(supercells):
    size = len(next(cell for cell in supercells if cell is not None))
    forces = []
    for cell in supercells:
        if cell is None:
            forces.append(np.zeros((size, 3)))
            continue
        atoms = Atoms(
            cell.symbols,
            cell=cell.cell,
            scaled_positions=cell.scaled_positions,
            pbc=True,
        )
        forces.append(_emt(atoms).get_forces())
    return np.array(forces)


def test_phonopy_force_constants_match_sequential_evaluation():
    phonopy = pytest.importorskip("phonopy")

    def phonon():
        result = phonopy.Phonopy(
            _phonopy_atoms(_l12_unit_cell()), supercell_matrix=np.eye(3) * 2
        )
        result.generate_displacements(distance=0.01)
        return result

    batched = phonon()
    scheduler = _scheduler()
    scheduler.submit(displacement_forces(batched), "emt")

    (result,) = scheduler.run()

    assert result.ok, result.error
    assert scheduler.batch_log == [("emt", 2, 64)]
    batched.produce_force_constants(show_drift=False)
    reference = phonon()
    reference.forces = _sequential_forces(reference.supercells_with_displacements)
    reference.produce_force_constants(show_drift=False)
    np.testing.assert_array_equal(batched.force_constants, reference.force_constants)


def test_phono3py_fc3_and_fc2_supercells_share_one_request():
    phono3py = pytest.importorskip("phono3py")

    def phonon():
        result = phono3py.Phono3py(
            _phonopy_atoms(_l12_unit_cell()),
            supercell_matrix=np.eye(3) * 2,
            phonon_supercell_matrix=np.eye(3) * 2,
        )
        result.generate_displacements(distance=0.03, cutoff_pair_distance=3.0)
        return result

    batched = phonon()
    scheduler = _scheduler()
    scheduler.submit(displacement_forces(batched), "emt")

    (result,) = scheduler.run()

    assert result.ok, result.error
    evaluated = sum(cell is not None for cell in batched.supercells_with_displacements)
    evaluated += len(batched.phonon_supercells_with_displacements)
    assert [size for _, size, _ in scheduler.batch_log] == [evaluated]
    batched.produce_fc3()
    batched.produce_fc2()
    reference = phonon()
    reference.forces = _sequential_forces(reference.supercells_with_displacements)
    reference.phonon_forces = _sequential_forces(
        reference.phonon_supercells_with_displacements
    )
    reference.produce_fc3()
    reference.produce_fc2()
    np.testing.assert_array_equal(batched.fc3, reference.fc3)
    np.testing.assert_array_equal(batched.fc2, reference.fc2)


def _pair_energy(atoms, coupling=0.5):
    # Repulsion between Au atoms, decaying with distance, on a fixed lattice.
    gold = np.flatnonzero(atoms.numbers == 79)
    if len(gold) < 2:
        return 0.0
    pairs = [atoms.get_distance(i, j, mic=True) for i, j in combinations(gold, 2)]
    return coupling * sum(1.0 / distance for distance in pairs)


def _pair_model(structures, properties):
    return [
        {"energy": _pair_energy(atoms), "forces": np.zeros((len(atoms), 3))}
        for atoms in structures
    ]


def test_swap_monte_carlo_samples_the_boltzmann_distribution():
    # Two Au atoms on eight fcc sites: 28 configurations, enumerated exactly.
    lattice = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    lattice.numbers[[0, 1]] = 79
    levels = {}
    for pair in combinations(range(len(lattice)), 2):
        atoms = lattice.copy()
        atoms.numbers[:] = 29
        atoms.numbers[list(pair)] = 79
        energy = round(_pair_energy(atoms), 8)
        levels[energy] = levels.get(energy, 0) + 1
    energies = np.array(sorted(levels))
    degeneracy = np.array([levels[energy] for energy in energies])
    temperatures = [300.0, 900.0]
    scheduler = _scheduler(_pair_model)
    scheduler.submit(
        swap_monte_carlo(lattice, temperatures, 20_000, exchange_interval=5, rng=1),
        "emt",
    )

    (result,) = scheduler.run()

    assert result.ok, result.error
    trace = np.round(result.value["trace"][1000:], 8)
    for column, temperature in enumerate(temperatures):
        weights = degeneracy * np.exp(-energies / (units.kB * temperature))
        expected = weights / weights.sum()
        sampled = np.array([np.mean(trace[:, column] == level) for level in energies])
        np.testing.assert_allclose(sampled, expected, atol=0.02)
    assert 0 < result.value["exchange_acceptance"][0] < 1


def test_swap_monte_carlo_batches_replicas_and_tracks_energies():
    lattice = _alloy(seed=3)
    temperatures = [0.0, 300.0, 1000.0]
    scheduler = _scheduler()
    scheduler.submit(
        swap_monte_carlo(lattice, temperatures, 30, exchange_interval=0, rng=0), "emt"
    )

    (result,) = scheduler.run()

    assert result.ok, result.error
    assert all(size == 3 for _, size, _ in scheduler.batch_log)
    assert len(scheduler.batch_log) == 31
    value = result.value
    for atoms, energy in zip(value["replicas"], value["energies"]):
        assert sorted(atoms.numbers) == sorted(lattice.numbers)
        np.testing.assert_allclose(energy, _emt(atoms).get_potential_energy())
    # Without exchanges, the zero-temperature replica never moves uphill.
    assert np.all(np.diff(value["trace"][:, 0]) <= 0)
    assert value["lowest_energy"] == value["trace"].min()
    np.testing.assert_allclose(
        value["lowest_energy"], _emt(value["lowest"]).get_potential_energy()
    )


def test_swap_monte_carlo_rejects_a_single_species():
    scheduler = _scheduler()
    scheduler.submit(swap_monte_carlo(bulk("Cu", cubic=True), [300.0], 5), "emt")

    (result,) = scheduler.run()

    assert isinstance(result.error, ValueError)
