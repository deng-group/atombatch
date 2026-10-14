# atombatch

atombatch batches the energy and force evaluations of many independent ASE tasks into shared model calls. Single points, relaxations, NEB bands, and molecular dynamics run their ordinary, unmodified ASE optimizers and integrators. Each force request is routed to the model it names, and the pending requests of all running tasks are evaluated together in one batched call per model.

Status: early development. The scheduler, the Symmetrix and sequential backends, and the single-point, relaxation, NEB, and molecular-dynamics tasks are implemented and tested against independent ASE runs. Performance has not been measured yet.

```python
from ase import units
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from atombatch import Scheduler, molecular_dynamics, neb, relax, single_point

scheduler = Scheduler(max_active=256)
scheduler.add_model("omat", symmetrix_calculator, max_atoms=50_000)
for atoms in orderings:
    scheduler.submit(relax(atoms, fmax=0.05, optimizer=BFGS), "omat")
for images in paths:
    scheduler.submit(neb(images, neb_kwargs={"climb": True}, interpolate="idpp"), "omat")
scheduler.submit(
    molecular_dynamics(
        cell, Langevin, 10_000, timestep=2 * units.fs, temperature_K=600, friction=0.02 / units.fs
    ),
    "omat",
)
results = scheduler.run()      # one TaskResult per task, in submission order
```

## Why

A machine-learned interatomic potential evaluated on one small structure leaves a GPU mostly idle, because each call pays a fixed cost for kernel launches, synchronization, and host bookkeeping. Screening many orderings of a disordered lattice, relaxing hundreds of candidates, or running many NEB paths therefore spends most of its time on per-call overhead. Evaluating the requests of many tasks together amortizes that cost: with Symmetrix on an Apple M1 Max GPU, joining 64 small structures into one evaluation reduced the cost from about 150 to 20 us/atom.

## Scope

- Tasks: single points, relaxations (any ASE optimizer, cell filters included), NEB (regular and climbing image, several bands at once), and molecular dynamics (any ASE integrator), mixed in one run.
- Scheduling: tasks that finish are replaced from a queue so that batches stay full, and batches are sized to a per-model atom or edge budget.
- Models: several models in one run, batched per model. Backends provide batched evaluation (Symmetrix natively); any other ASE calculator is supported through a sequential fallback.
- Results are the same as running each task independently, because each task receives exactly the forces it requested.

## Related work

- [TorchSim](https://github.com/TorchSim/torch-sim) and the [NVIDIA ALCHEMI toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) batch many systems by reimplementing optimizers and integrators as vectorized GPU code. atombatch keeps ASE's algorithms and batches only the model evaluation.
- The fairchem [`InferenceBatcher`](https://github.com/facebookresearch/fairchem/blob/main/docs/core/common_tasks/inference_batcher.md) batches requests from concurrent ASE simulations through Ray Serve for fairchem models. atombatch aims at the same pattern without Ray and independently of the model package.
- MatterSim's [`BatchRelaxer`](https://github.com/microsoft/mattersim/blob/main/src/mattersim/applications/batch_relax.py) steps ASE optimizers in lockstep for relaxations with one model.
- [i-PI](https://github.com/i-pi/i-pi) batches force requests across replicas and systems within its own simulation engine.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
pytest
```

## License

atombatch is distributed under the GNU General Public License, version 3 or later. See [LICENSE](LICENSE).
