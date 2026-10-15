# Batched workflows

A task waits on one request at a time. A workflow that evaluates its structures one after another, as phonopy or an EOS script normally does, therefore contributes one structure per round. When the evaluations are independent of each other, the workflow submits them together with {meth}`TaskContext.evaluate <atombatch.TaskContext.evaluate>`, and a single workflow fills a batch on its own. The workflows below are built this way, and several of them can run in one scheduler alongside relaxations, NEB, and MD.

## Force constants with phonopy and phono3py

{func}`~atombatch.displacement_forces` takes a `Phonopy` or `Phono3py` object on which `generate_displacements` has been called. It evaluates all displaced supercells in one request and assigns the forces. For phono3py, the fc3 supercells and the separate fc2 supercells share that request. Force constants are then produced with phonopy as usual.

```python
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from atombatch import Scheduler, displacement_forces

unit = bulk("Cu", "fcc", a=3.6, cubic=True)
unit.numbers[0] = 79  # Cu3Au, L1_2
phonon = Phonopy(
    PhonopyAtoms(
        symbols=unit.get_chemical_symbols(),
        cell=unit.cell[:],
        scaled_positions=unit.get_scaled_positions(),
    ),
    supercell_matrix=np.eye(3) * 2,
)
phonon.generate_displacements(distance=0.01)

scheduler = Scheduler()
scheduler.add_model("emt", EMT())
scheduler.submit(displacement_forces(phonon), "emt")
scheduler.run()

phonon.produce_force_constants()
print(phonon.force_constants.shape)
```

Force constants from batched evaluation are identical to those from sequential evaluation, for phonopy fc2 and for phono3py fc2 and fc3. phono3py often generates hundreds to thousands of displaced supercells of identical size, which is close to the ideal batch.

## Equation of state

{func}`~atombatch.equation_of_state` scales the cell isotropically, evaluates all scaled cells in one request, and fits them with `ase.eos.EquationOfState` (Birch–Murnaghan by default).

```python
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from atombatch import Scheduler, equation_of_state

scheduler = Scheduler()
scheduler.add_model("emt", EMT())
for metal, a in [("Cu", 3.6), ("Ag", 4.1), ("Au", 4.1)]:
    scheduler.submit(equation_of_state(bulk(metal, "fcc", a=a)), "emt", name=metal)
for result in scheduler.run():
    print(result.name, result.value["volume"], result.value["bulk_modulus"] / units.GPa)
```

The three EOS scans share each round, so 21 cells are evaluated in one call.

## Swap Monte Carlo with replica exchange

{func}`~atombatch.swap_monte_carlo` samples site occupancies on a fixed lattice. One replica runs at each temperature. A step proposes one swap of two unlike atoms per replica, chosen uniformly among unlike pairs of the given sites, and evaluates the trial configurations of all replicas in one request. Every `exchange_interval` steps, neighbouring temperatures attempt to exchange configurations.

```python
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from atombatch import Scheduler, swap_monte_carlo

alloy = bulk("Cu", "fcc", a=3.7, cubic=True).repeat(2)
alloy.numbers[np.random.default_rng(0).choice(len(alloy), 8, replace=False)] = 79

scheduler = Scheduler()
scheduler.add_model("emt", EMT())
scheduler.submit(
    swap_monte_carlo(alloy, [300, 600, 1200], steps=500, exchange_interval=10, rng=0),
    "emt",
)
(result,) = scheduler.run()
print(result.value["lowest_energy"], result.value["acceptance"])
```

The result holds the final configuration and energy at each temperature, the lowest-energy configuration visited, the swap and exchange acceptance ratios, and the energy trace with one row per step. The test suite checks that each replica samples the exact Boltzmann populations of a lattice model small enough to enumerate.

Positions stay on the input lattice, so energies are those of unrelaxed configurations. Sampling with relaxed energies would need one relaxation per trial and is not implemented. To screen many independent orderings, submit one Monte Carlo task per starting configuration or per composition. Their replicas then share each batched call.

## Random structure searching: AIRSS and hot-AIRSS

{func}`~atombatch.airss` runs one sample of ab initio random structure searching (AIRSS): it generates a random but sensible structure and relaxes positions and cell together at a given pressure. {func}`~atombatch.hot_airss` runs one sample of hot-AIRSS: random structure, relaxation, a short Langevin anneal at high temperature, and a second relaxation, which lets a structure leave a poor basin. Every sample is a task, so the samples of a search, in whatever stage they are, share each batched call, and finished samples are replaced from the queue.

```python
from ase.calculators.emt import EMT
from atombatch import Scheduler, airss
from atombatch.workflows.airss import random_structure, rank


def generate(rng):
    return random_structure("Cu4", volume_per_atom=12.0, minsep=2.0, rng=rng)


scheduler = Scheduler()
scheduler.add_model("emt", EMT())
for index in range(24):
    scheduler.submit(airss(generate, seed=index, pressure=0.0), "emt")
ranked = rank([result.value for result in scheduler.run()])
best = ranked[0]
print(best["result"]["enthalpy_per_atom"], best["count"])
```

`generate(rng)` returns a random `Atoms`, with `rng` a NumPy generator seeded by `seed`. {func}`~atombatch.workflows.airss.random_structure` places atoms at random in a random cell of a target volume, at least `minsep` apart, without symmetry. {func}`~atombatch.workflows.airss.buildcell` runs the AIRSS `buildcell` program on a seed, given as the text of a `.cell` file or as an airsspy `SeedAtoms`, which supports symmetry, fragments, and the other buildcell constraints. AIRSS is distributed by its authors as source code, and `buildcell` builds with gfortran.

Each result holds the relaxed and initial structures, energy, enthalpy `H = E + pV` (per cell and per atom), volume per atom, convergence, and the steps of every stage. {func}`~atombatch.workflows.airss.rank` sorts results by enthalpy per atom and merges duplicates by enthalpy and volume, optionally confirmed by a structure comparison such as pymatgen's `StructureMatcher`.

`examples/airss_carbon.py` searches 8-atom carbon cells at 20 GPa with 48 AIRSS and 48 hot-AIRSS samples in one scheduler (2 minutes on an M1 Max with Symmetrix). Diamond is the lowest minimum, found three times, followed by lonsdaleite (25 meV/atom higher) and a graphite-like layered structure (202 meV/atom higher), in agreement with direct relaxations of diamond and graphite. In a buildcell seed, `VARVOL` is the target volume per atom, and `NFORM=1` keeps the number of atoms fixed; without it, buildcell picks a random number of formula units. Models without a dispersion correction do not bind layered or chain structures, which expand into separated fragments when searched at zero pressure.

The tests check that both tasks reproduce the same steps carried out by hand, that the enthalpy includes the `pV` term, and that a search of four-atom copper cells with EMT finds fcc copper several times.

## Workflows that need no special support

Some workflows are already sets of ordinary tasks:

- **Structure search.** Random structure searching and genetic algorithms relax populations of independent candidates. Submit one {func}`~atombatch.relax` task per candidate.
- **Migration barriers.** Barrier tables for kinetic Monte Carlo, or comparisons of migration paths across dopants, are many NEB bands. Submit one {func}`~atombatch.neb` task per path. All moving images of all bands share each round.
- **Temperature series of MD.** Submit one {func}`~atombatch.molecular_dynamics` task per temperature or per starting configuration.

For other workflows, write a task function that calls `context.evaluate` directly, as described in {doc}`custom-tasks`.
