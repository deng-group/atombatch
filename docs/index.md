# atombatch

atombatch batches the energy and force evaluations of many independent ASE tasks into shared model calls. Relaxations, NEB bands, molecular dynamics, phonon displacements, and Monte Carlo trials run as ordinary ASE or phonopy code. Their pending requests are collected and evaluated together in one batched call per model, and each task receives exactly the result it would get from an independent run.

A machine-learned interatomic potential evaluated on one small structure leaves a GPU mostly idle, because each call pays a fixed cost for kernel launches, synchronization, and host bookkeeping. Evaluating the requests of many tasks together amortizes that cost. With Symmetrix on an NVIDIA RTX 4090, joining 64 SrTiO{sub}`3` cells of 40 atoms into one evaluation reduced the cost from 62 to 4.3 us/atom (FP32), and 64 FIRE relaxations ran 4.2 times faster than a sequential ASE loop, with identical step counts; on an Apple M1 Max GPU they ran 4.8 times faster ({doc}`performance`).

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from atombatch import Scheduler, relax

scheduler = Scheduler(max_active=64)
scheduler.add_model("emt", EMT())
for seed in range(8):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=seed)
    scheduler.submit(relax(atoms, fmax=0.05, optimizer=BFGS), "emt")
results = scheduler.run()
print([result.value["steps"] for result in results])
```

EMT stands in for a model here so that the example runs without an MLIP. {doc}`models` describes how to attach Symmetrix, MACE, fairchem, and other potentials.

atombatch is in early development. Correctness is tested against independent ASE runs and against the models' own ASE calculators. The gain is largest for machine-learned potentials on a GPU. On a CPU it is smaller and depends on the batch size.

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

installation
quickstart
examples
models
workflows
custom-tasks
performance
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api
design
```
