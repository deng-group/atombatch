# Quickstart

A run has three steps. Register one or more models with {meth}`~atombatch.Scheduler.add_model`, submit tasks with {meth}`~atombatch.Scheduler.submit`, and call {meth}`~atombatch.Scheduler.run`, which returns one {class}`~atombatch.TaskResult` per task in submission order.

```python
import numpy as np
from ase import units
from ase.build import add_adsorbate, bulk, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from atombatch import Scheduler, molecular_dynamics, neb, relax, single_point

scheduler = Scheduler(max_active=64)
scheduler.add_model("emt", EMT())

crystal = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
crystal.rattle(0.05, seed=0)

# A single point that also returns the stress.
scheduler.submit(single_point(crystal, ("energy", "forces", "stress")), "emt", name="point")

# Relaxations of positions, and of positions and cell together.
scheduler.submit(relax(crystal, fmax=0.05, optimizer=BFGS), "emt", name="relax")
scheduler.submit(
    relax(crystal, fmax=0.05, cell_filter=FrechetCellFilter), "emt", name="cell-relax"
)

# A climbing-image NEB for an adatom hop on Al(100).
slab = fcc100("Al", size=(2, 2, 3))
add_adsorbate(slab, "Au", 1.7, "hollow")
slab.center(axis=2, vacuum=4.0)
slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
initial, final = slab.copy(), slab.copy()
final[-1].x += final.get_cell()[0, 0] / 2
for endpoint in (initial, final):
    endpoint.calc = EMT()
    BFGS(endpoint, logfile=None).run(fmax=0.02)
    endpoint.calc = None
images = [initial] + [initial.copy() for _ in range(3)] + [final]
scheduler.submit(
    neb(images, fmax=0.05, neb_kwargs={"climb": True}, interpolate="idpp"),
    "emt",
    name="neb",
)

# Langevin dynamics at 600 K.
hot = crystal.copy()
MaxwellBoltzmannDistribution(hot, temperature_K=600, rng=np.random.default_rng(0))
scheduler.submit(
    molecular_dynamics(
        hot, Langevin, 200, timestep=2 * units.fs, temperature_K=600,
        friction=0.02 / units.fs, rng=np.random.default_rng(1),
    ),
    "emt",
    name="md",
)

results = scheduler.run()
for result in results:
    print(result.name, result.ok)
print(scheduler.batch_log[:3])
```

All five tasks run at once. Each round, the scheduler collects the structures that the tasks are waiting on and evaluates them together. {attr}`~atombatch.Scheduler.batch_log` records each call as `(model, structures, atoms)`:

```text
[('emt', 7, 129), ('emt', 2, 64), ('emt', 5, 103)]
```

The first round makes two calls, because requests that need stress are evaluated separately. One call holds the relaxation, the MD state, and all five NEB images (the endpoint energies are needed once). The other holds the single point and the cell relaxation, which read stress. From the second round on, the NEB band contributes only its three moving images.

## Results

Each {class}`~atombatch.TaskResult` has a `name`, a `value`, and an `error`. A task that raises an exception stores it in `error` and does not affect the other tasks. The values of the built-in tasks are dictionaries:

| Task | Keys |
| --- | --- |
| {func}`~atombatch.single_point` | `atoms` and the requested properties |
| {func}`~atombatch.relax` | `atoms`, `energy`, `forces`, `converged`, `steps` (and `stress` with a cell filter) |
| {func}`~atombatch.neb` | `images`, `energies`, `converged`, `steps` |
| {func}`~atombatch.molecular_dynamics` | `atoms`, `energy`, `forces`, `steps` |

Input structures are copied, so the structures passed to the tasks are not modified.

## Equivalence with independent runs

A batched task receives exactly the energies and forces it requested, evaluated at exactly the positions and cell it requested. Optimizers and integrators therefore take the same steps as in an independent ASE run. The test suite checks step counts and final positions against independent runs for FIRE, BFGS, BFGSLineSearch, cell filters, NEB with and without climbing image, and velocity Verlet and Langevin dynamics.
