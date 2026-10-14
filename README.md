<p align="center"><img src="https://raw.githubusercontent.com/deng-group/atombatch/main/docs/_static/logo.svg" alt="atombatch logo" width="120"></p>

# atombatch

atombatch batches the energy and force evaluations of many independent ASE tasks into shared model calls. Single points, relaxations, NEB bands, and molecular dynamics run their ordinary, unmodified ASE optimizers and integrators. Each force request is routed to the model it names, and the pending requests of all running tasks are evaluated together in one batched call per model.

Status: early development. The scheduler, the Symmetrix, TorchSim, and sequential backends, and the single-point, relaxation, NEB, and molecular-dynamics tasks are implemented and tested against independent ASE runs. The phonon, equation-of-state, and swap Monte Carlo workflows are tested against sequential evaluation and, for Monte Carlo, against exact Boltzmann populations. First benchmarks are in [docs/performance.md](https://github.com/deng-group/atombatch/blob/main/docs/performance.md).

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

With a TorchSim model, the same tasks are submitted unchanged:

```python
import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

model = MaceModel(mace_mp(model="medium", return_raw_model=True), device=torch.device("cuda"), dtype=torch.float64, compute_stress=True)
scheduler.add_model("mace", model, max_atoms=20_000)
```

## Why

A machine-learned interatomic potential evaluated on one small structure leaves a GPU mostly idle, because each call pays a fixed cost for kernel launches, synchronization, and host bookkeeping. Screening many orderings of a disordered lattice, relaxing hundreds of candidates, or running many NEB paths therefore spends most of its time on per-call overhead. Evaluating the requests of many tasks together amortizes that cost: with Symmetrix on an Apple M1 Max GPU, joining 64 SrTiO3 cells of 40 atoms into one evaluation reduced the cost from 145 to 23 us/atom, and 64 FIRE relaxations ran three times faster than a sequential ASE loop with identical step counts. On a CPU the gain is smaller and depends on the batch size (see [performance](https://github.com/deng-group/atombatch/blob/main/docs/performance.md)).

## Scope

- Tasks: single points, relaxations (any ASE optimizer, cell filters included), NEB (regular and climbing image, several bands at once), and molecular dynamics (any ASE integrator), mixed in one run.
- Batched workflows: tasks that need many independent evaluations submit them together through `TaskContext.evaluate`, so one workflow fills a batch on its own.
  - `displacement_forces`: forces on all displaced supercells of a phonopy or phono3py object (fc2, fc3, and the separate fc2 supercells of phono3py) in one request.
  - `equation_of_state`: all scaled cells in one request, fitted with `ase.eos`.
  - `swap_monte_carlo`: Metropolis swaps of site occupancies with replica exchange, the trial configurations of all temperatures evaluated in one request per step.
  - Custom workflows call `context.evaluate(structures)` directly from any task function.
- Scheduling: tasks that finish are replaced from a queue so that batches stay full, and batches are sized to a per-model atom or edge budget.
- Models: several models in one run, batched per model. `add_model` accepts:
  - a Symmetrix calculator (Symmetrix-XL 0.1.1 or later), with all structures of a round joined into one native call;
  - a [TorchSim](https://github.com/TorchSim/torch-sim) model, which provides batched evaluation for MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models (`pip install "atombatch[torchsim]"`, Python 3.12 or later);
  - a function `evaluate(structures, properties)` returning one result dictionary per structure, for a model with its own batched inference;
  - any other ASE calculator, evaluated one structure at a time. This keeps scheduling and refill but gives no batching speedup, because the ASE calculator interface evaluates one structure per call.
- Results are the same as running each task independently, because each task receives exactly the forces it requested.

## Related work

- [TorchSim](https://github.com/TorchSim/torch-sim) and the [NVIDIA ALCHEMI toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) batch many systems by reimplementing optimizers and integrators as vectorized GPU code. atombatch keeps ASE's algorithms and batches only the model evaluation.
- The fairchem [`InferenceBatcher`](https://github.com/facebookresearch/fairchem/blob/main/docs/core/common_tasks/inference_batcher.md) batches requests from concurrent ASE simulations through Ray Serve for fairchem models. atombatch aims at the same pattern without Ray and independently of the model package.
- MatterSim's [`BatchRelaxer`](https://github.com/microsoft/mattersim/blob/main/src/mattersim/applications/batch_relax.py) steps ASE optimizers in lockstep for relaxations with one model.
- [i-PI](https://github.com/i-pi/i-pi) batches force requests across replicas and systems within its own simulation engine.

## Documentation

The documentation in [`docs/`](https://github.com/deng-group/atombatch/blob/main/docs) covers installation, models, batched workflows, writing tasks, performance, and the API. Build it with:

```bash
uv pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
```

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
pytest
```

## License

atombatch is distributed under the GNU General Public License, version 3 or later. See [LICENSE](https://github.com/deng-group/atombatch/blob/main/LICENSE).
