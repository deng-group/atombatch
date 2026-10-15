import shutil

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

from atombatch import Scheduler, airss, hot_airss
from atombatch.workflows.airss import buildcell, parse_cell, random_structure, rank


def _copper(rng):
    return random_structure("Cu4", volume_per_atom=12.0, minsep=2.0, rng=rng)


def _run(tasks):
    scheduler = Scheduler()
    scheduler.add_model("emt", EMT())
    for task in tasks:
        scheduler.submit(task, "emt")
    results = scheduler.run()
    assert all(result.ok for result in results), [r.error for r in results]
    return scheduler, [result.value for result in results]


def _relax_by_hand(atoms, pressure, fmax=0.05, steps=1000):
    dynamics = FIRE(
        FrechetCellFilter(atoms, scalar_pressure=pressure * units.GPa), logfile=None
    )
    converged = dynamics.run(fmax=fmax, steps=steps)
    return converged, dynamics.nsteps


def test_random_structure_respects_its_constraints():
    for seed in range(5):
        atoms = random_structure("Si2O4", volume_per_atom=10.0, minsep=1.4, rng=seed)
        assert sorted(atoms.get_chemical_symbols()) == sorted(
            ["Si", "Si", "O"] + ["O"] * 3
        )
        assert all(atoms.pbc)
        distances = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        assert distances.min() >= 1.4
        assert 0.9 * 60 <= atoms.get_volume() <= 1.1 * 60
    first = random_structure("Si2O4", 10.0, 1.4, rng=3)
    np.testing.assert_array_equal(
        first.positions, random_structure("Si2O4", 10.0, 1.4, rng=3).positions
    )


def test_random_structure_reports_impossible_packing():
    with pytest.raises(RuntimeError, match="Could not place"):
        random_structure("Cu8", volume_per_atom=2.0, minsep=3.0, rng=0, max_attempts=20)


@pytest.mark.parametrize("pressure", [0.0, 20.0])
def test_airss_matches_a_relaxation_by_hand(pressure):
    _, (result,) = _run([airss(_copper, seed=4, pressure=pressure)])

    atoms = _copper(np.random.default_rng(4))
    atoms.calc = EMT()
    converged, nsteps = _relax_by_hand(atoms, pressure)
    assert result["converged"] == converged
    assert result["steps"] == {"relax": nsteps}
    np.testing.assert_allclose(result["atoms"].positions, atoms.positions, atol=1e-9)
    np.testing.assert_allclose(result["atoms"].cell.array, atoms.cell.array, atol=1e-9)
    energy = atoms.get_potential_energy()
    assert result["enthalpy"] == pytest.approx(
        energy + pressure * units.GPa * atoms.get_volume()
    )
    np.testing.assert_array_equal(
        result["initial"].positions, _copper(np.random.default_rng(4)).positions
    )


def test_hot_airss_matches_its_stages_by_hand():
    temperature = 1500.0
    _, (result,) = _run(
        [hot_airss(_copper, seed=2, temperature_K=temperature, md_steps=50)]
    )

    rng = np.random.default_rng(2)
    atoms = _copper(rng)
    atoms.calc = EMT()
    _, prerelax = _relax_by_hand(atoms, 0.0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=temperature,
        friction=0.02 / units.fs,
        rng=rng,
        logfile=None,
    ).run(50)
    atoms.set_momenta(np.zeros_like(atoms.positions))
    converged, relax = _relax_by_hand(atoms, 0.0)

    assert result["steps"] == {"prerelax": prerelax, "md": 50, "relax": relax}
    assert result["converged"] == converged
    np.testing.assert_allclose(result["atoms"].positions, atoms.positions, atol=1e-8)


def test_search_finds_fcc_copper_and_ranks_duplicates():
    scheduler, results = _run([airss(_copper, seed=index) for index in range(24)])

    # The searches share batched calls.
    assert scheduler.batch_log[0][1] == 24
    ranked = rank(results)
    fcc = bulk("Cu", "fcc", a=3.6, cubic=True)
    fcc.calc = EMT()
    _relax_by_hand(fcc, 0.0, fmax=0.01)
    best = ranked[0]["result"]
    assert best["enthalpy_per_atom"] == pytest.approx(
        fcc.get_potential_energy() / len(fcc), abs=2e-3
    )
    assert ranked[0]["count"] > 1
    assert sum(entry["count"] for entry in ranked) == sum(
        r["converged"] for r in results
    )
    enthalpies = [entry["result"]["enthalpy_per_atom"] for entry in ranked]
    assert enthalpies == sorted(enthalpies)


@pytest.mark.skipif(shutil.which("buildcell") is None, reason="buildcell not on PATH")
def test_buildcell_seed_text_generates_structures():
    seed = """
%BLOCK LATTICE_CART
3 0 0
0 3 0
0 0 3
%ENDBLOCK LATTICE_CART
#VARVOL=12
%BLOCK POSITIONS_FRAC
Cu 0 0 0 # Cu1 % NUM=4
%ENDBLOCK POSITIONS_FRAC
#MINSEP=2.0
#NFORM=1
"""
    generate = buildcell(seed)
    atoms = generate()
    assert atoms.get_chemical_symbols() == ["Cu"] * 4
    _, (result,) = _run([airss(generate, seed=0)])
    assert result["converged"]


@pytest.mark.skipif(shutil.which("buildcell") is None, reason="buildcell not on PATH")
def test_buildcell_with_airsspy_seed_atoms():
    airsspy = pytest.importorskip("airsspy")
    seed = airsspy.SeedAtoms("Cu", cell=[3, 3, 3], pbc=True)
    seed.gentags.varvol = 12
    seed.gentags.minsep = 2.0
    seed.get_atom_tag(0).num = 4
    generate = buildcell(seed)

    atoms = generate()

    assert atoms.get_chemical_symbols() == ["Cu"] * 4
    distances = atoms.get_all_distances(mic=True)[np.triu_indices(4, 1)]
    assert distances.min() >= 2.0 - 1e-6


def test_parse_cell_reads_buildcell_output():
    text = """
#   Space group: P21
%BLOCK LATTICE_ABC
      4.0   5.0   6.0
     90.0  90.0  120.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_FRAC
   C   0.1  0.2  0.3
   O   0.5  0.5  0.5  # comment
%ENDBLOCK POSITIONS_FRAC
"""
    atoms = parse_cell(text)

    assert atoms.get_chemical_symbols() == ["C", "O"]
    np.testing.assert_allclose(atoms.cell.cellpar(), [4, 5, 6, 90, 90, 120])
    np.testing.assert_allclose(atoms.get_scaled_positions()[0], [0.1, 0.2, 0.3])
