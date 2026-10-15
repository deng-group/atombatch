<p align="center"><img src="https://raw.githubusercontent.com/deng-group/atombatch/main/docs/_static/logo.svg" alt="atombatch logo" width="120"></p>

# atombatch

atombatch batches the energy and force evaluations of many independent ASE tasks into shared model calls. Relaxations, NEB bands, molecular dynamics, phonon displacements, and Monte Carlo trials run their ordinary, unmodified ASE code. The pending requests of all running tasks are evaluated together in one batched call per model, and each task receives exactly the result it would get from an independent run.

A machine-learned interatomic potential evaluated on one small structure leaves a GPU mostly idle, because each call pays a fixed cost for kernel launches, synchronization, and host bookkeeping. Batching amortizes that cost. For 64 FIRE relaxations of 40-atom SrTiO3 cells, with identical step counts to a sequential ASE loop:

| Model | GPU | Throughput gain at 64 cells per call | Relaxations |
| --- | --- | --- | --- |
| Symmetrix, MACE-OMat-0 medium (FP32) | NVIDIA RTX 4090 | 14.3x | 4.2x |
| Symmetrix, MACE-OMat-0 medium (FP64) | NVIDIA RTX 4090 | 5.4x | 3.7x |
| MACE-MP small via TorchSim (FP32) | NVIDIA RTX 4090 | 4.6x | 4.1x |
| Symmetrix, MACE-OMat-0 medium (FP32) | Apple M1 Max | 6.3x | 4.8x |

On a CPU the gain is small and depends on the batch size. See [performance](https://deng-group.github.io/atombatch/performance.html) for the full measurements.

## Installation

```bash
pip install atombatch
```

atombatch needs Python 3.10 or later, NumPy, and ASE. The model is installed separately.

### Symmetrix

[Symmetrix-XL](https://github.com/bonan-group/symmetrix-xl) evaluates MACE models natively on CPUs and GPUs. Install the package for your hardware together with matscipy, which Symmetrix needs for fast neighbor lists, and Numba, which atombatch uses to update cached neighbor lists:

| Hardware | Install |
| --- | --- |
| Linux, CPU | `pip install "atombatch[symmetrix]"` |
| Linux, NVIDIA GPU | `pip install atombatch matscipy numba symmetrix-xl-cuda12-sm89` (choose `sm80`, `sm86`, `sm89`, `sm90`, or `cuda13-sm120` for your GPU, or run `symmetrix backend install` after installing `symmetrix-xl`) |
| macOS, Apple GPU | `pip install atombatch matscipy numba "symmetrix-xl @ git+https://github.com/bonan-group/symmetrix-xl.git@main#subdirectory=symmetrix"` (builds the Metal backend from source in a few minutes; needs the Xcode command line tools), or the prebuilt wheels of the [metal-preview-1](https://github.com/deng-group/symmetrix-xl/releases/tag/metal-preview-1) release |

Symmetrix evaluates a JSON export of a MACE model. For example, for MACE-MP-0b medium (the conversion needs `mace-torch`):

```bash
curl -L -o mace_agnesi_medium.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model
symmetrix_extract_mace --model mace_agnesi_medium.model --output mace-mp-0b-medium.json
```

On machines where an old CUDA toolkit is on the library path, the Symmetrix CUDA wheels can load the wrong `libnvrtc.so` or `libcudart.so` and fail with `nvrtcGetCUBIN` or `invalid device ordinal`. The [installation guide](https://deng-group.github.io/atombatch/installation.html) gives the fix.

### Other models

MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models are batched through [TorchSim](https://github.com/TorchSim/torch-sim) (Python 3.12 or later):

```bash
pip install "atombatch[torchsim]" mace-torch
```

Any other ASE calculator also works, evaluated one structure at a time. Further extras: `atomate2` (Python 3.11 or later), `phonons` (phonopy and phono3py), `test`, and `docs`.

## Quick start

Relax 64 rattled SrTiO3 supercells with Symmetrix. Each round, the forces of all 64 cells are evaluated in one call:

```python
from ase import Atoms
from ase.optimize import FIRE
from symmetrix import Symmetrix

from atombatch import Scheduler, relax


def perovskite(seed):
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[3.95, 3.95, 3.95],
        pbc=True,
    ).repeat(2)
    atoms.rattle(0.05, seed=seed)
    return atoms


calculator = Symmetrix("mace-mp-0b-medium.json", dtype="float32")  # add metal=True on Apple silicon
scheduler = Scheduler(max_active=64)
scheduler.add_model("mace", calculator)
for seed in range(64):
    scheduler.submit(relax(perovskite(seed), fmax=0.05, optimizer=FIRE), "mace")

results = scheduler.run()  # one TaskResult per task, in submission order
print(sum(r.value["converged"] for r in results), "of", len(results), "converged")
print(scheduler.batch_log[0])  # ('mace', 64, 2560): 64 cells, 2560 atoms in the first call
```

Other tasks are submitted the same way and share the same batched calls. Here `images` is a list of NEB images, and `atoms` and `alloy` are structures of your own:

```python
from ase import units
from ase.md.langevin import Langevin
from atombatch import equation_of_state, molecular_dynamics, neb, swap_monte_carlo

scheduler.submit(neb(images, neb_kwargs={"climb": True}, interpolate="idpp"), "mace")
scheduler.submit(molecular_dynamics(atoms, Langevin, 10_000, timestep=2 * units.fs, temperature_K=600, friction=0.02 / units.fs), "mace")
scheduler.submit(equation_of_state(atoms), "mace")
scheduler.submit(swap_monte_carlo(alloy, [300, 600, 1200], steps=1000), "mace")
```

More complete scripts, for cell relaxations of all orderings, several NEB bands, MD at several temperatures, phonons, and swap Monte Carlo, are in [`examples/`](https://github.com/deng-group/atombatch/tree/main/examples) (described on the [Examples](https://deng-group.github.io/atombatch/examples.html) page), with a SLURM job template.

With a TorchSim model, only the model changes:

```python
import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

model = MaceModel(mace_mp(model="small", return_raw_model=True, default_dtype="float32"), device=torch.device("cuda"), dtype=torch.float32, compute_stress=True)
scheduler.add_model("mace", model)
```

### atomate2

atomate2 force-field and ASE makers run batched without changes, and return atomate2's own documents:

```python
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atombatch.integrations.atomate2 import atomate2_task

for structure in structures:
    scheduler.submit(atomate2_task(ForceFieldRelaxMaker(relax_cell=True), structure), "mace")
documents = [result.value for result in scheduler.run()]  # ForceFieldTaskDocuments
```

For 28 cell relaxations with Symmetrix on an M1 Max, the batched jobs ran 2.4 times faster than the same jobs one after another, with identical step counts. See the [atomate2 page](https://deng-group.github.io/atombatch/atomate2.html).

## Scope

- Tasks: single points, relaxations (any ASE optimizer, cell filters included), NEB (regular and climbing image, several bands at once), and molecular dynamics (any ASE integrator), mixed in one run.
- Batched workflows: tasks that need many independent evaluations submit them together through `TaskContext.evaluate`, so one workflow fills a batch on its own.
  - `displacement_forces`: forces on all displaced supercells of a phonopy or phono3py object in one request.
  - `equation_of_state`: all scaled cells in one request, fitted with `ase.eos`.
  - `swap_monte_carlo`: Metropolis swaps of site occupancies with replica exchange, the trial configurations of all temperatures evaluated in one request per step.
  - Custom workflows call `context.evaluate(structures)` from any task function.
- Scheduling: finished tasks are replaced from a queue (`max_active`) so that batches stay full, and calls are split at a per-model atom budget (`max_atoms`).
- Models: several models in one run, batched per model. Symmetrix-XL from 0.1.1 is batched natively, TorchSim models in one model call, and user functions with the signature `evaluate(structures, properties)` connect any model with its own batched inference.
- Results are the same as running each task independently, because each task receives exactly the energies and forces it requested. The tests check this against independent ASE runs and against each model's own ASE calculator.

## Related work

- [TorchSim](https://github.com/TorchSim/torch-sim) and the [NVIDIA ALCHEMI toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) batch many systems by reimplementing optimizers and integrators as vectorized GPU code. atombatch keeps ASE's algorithms and batches only the model evaluation.
- The fairchem [`InferenceBatcher`](https://github.com/facebookresearch/fairchem/blob/main/docs/core/common_tasks/inference_batcher.md) batches requests from concurrent ASE simulations through Ray Serve for fairchem models. atombatch aims at the same pattern without Ray and independently of the model package.
- MatterSim's [`BatchRelaxer`](https://github.com/microsoft/mattersim/blob/main/src/mattersim/applications/batch_relax.py) steps ASE optimizers in lockstep for relaxations with one model.
- [i-PI](https://github.com/i-pi/i-pi) batches force requests across replicas and systems within its own simulation engine.

## Documentation

The documentation is at **https://deng-group.github.io/atombatch/**. It covers installation, examples, models, batched workflows, writing tasks, performance, and the API. To build it locally:

```bash
pip install -e ".[docs]"
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
