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

## Workflows that need no special support

Some workflows are already sets of ordinary tasks:

- **Structure search.** Random structure searching and genetic algorithms relax populations of independent candidates. Submit one {func}`~atombatch.relax` task per candidate.
- **Migration barriers.** Barrier tables for kinetic Monte Carlo, or comparisons of migration paths across dopants, are many NEB bands. Submit one {func}`~atombatch.neb` task per path. All moving images of all bands share each round.
- **Temperature series of MD.** Submit one {func}`~atombatch.molecular_dynamics` task per temperature or per starting configuration.

For other workflows, write a task function that calls `context.evaluate` directly, as described in {doc}`custom-tasks`.
