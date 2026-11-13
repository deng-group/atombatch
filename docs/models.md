# Models

{meth}`~atombatch.Scheduler.add_model` registers a model under a name, and each task names the model it uses. Several models can be registered in one run. Requests are batched per model, because one batched call evaluates one set of weights. The argument can be any of the following, and atombatch selects the backend from its type.

| Argument | Backend | Batched |
| --- | --- | --- |
| A Symmetrix calculator (Symmetrix-XL 0.1.1 or later) | {class}`~atombatch.SymmetrixBackend` | yes, natively |
| A TorchSim `ModelInterface` | {class}`~atombatch.TorchSimBackend` | yes |
| A function `evaluate(structures, properties)` | {class}`~atombatch.FunctionBackend` | as the function implements it |
| Any other ASE calculator | {class}`~atombatch.SequentialBackend` | no, one structure per call |
| A {class}`~atombatch.Backend` instance | itself | as implemented |

The ASE calculator interface evaluates one `Atoms` object per call, and periodic structures with different cells cannot be joined into one `Atoms`. Batching therefore needs access below the calculator to the model's own batched input, which the first three backends provide. A plain ASE calculator still gains scheduling and refill but evaluates one structure at a time.

## Symmetrix

```python
from symmetrix import Symmetrix

calculator = Symmetrix("model.json", dtype="float64")
scheduler.add_model("omat", calculator, max_atoms=50_000)
```

For runs with large structures (1000 atoms or more), create the backend from the model file, so that each large trajectory can get a Symmetrix calculator of its own (see {doc}`performance`):

```python
from atombatch import SymmetrixBackend

scheduler.add_model("omat", SymmetrixBackend.from_model("model.json", dtype="float32"))
```

Any Symmetrix-XL release from 0.1.1 is batched. atombatch joins the structures into one disconnected neighbor graph and calls the Symmetrix evaluator directly. Symmetrix-XL 0.2 recognizes the trajectories in such a batch and evaluates them as one prepared segment group with the geometry kept on the device. Results agree with the Symmetrix ASE calculator to rounding (energies and forces to 1e-9, stresses to 1e-10).

Only model families whose energy is a sum of node energies over local graphs are joined: standard MACE and MACE-MH-1. MACE-POLAR-1 (Symmetrix-XL 0.2) has long-range electrostatics that couple all atoms of a structure, so it is never joined into one graph. Where Symmetrix provides `Symmetrix.calculate_polar_batch` (the deng-group `v0.2-deng` branch), atombatch evaluates the MACE-POLAR-1 requests of a round in shared native calls, one per periodicity, in which each structure keeps its own cell, graph inputs, and long-range solver. Results equal the Symmetrix calculator's to rounding, including D3 corrections when the installed Symmetrix adds them to batches. Otherwise, and for MACELES, MACEField, and other calculators with dispersion corrections, requests are evaluated one structure at a time through the Symmetrix calculator. Those tasks still run under the scheduler, with refill and mixed workflows, but gain no batching speedup. MACE-POLAR-1 reads the total charge, total spin, Fermi level, and external field of each structure from `Atoms.info` (`total_charge`, `total_spin`, `fermi_level`, `electric_field`) unless they are set on the calculator, and tasks can read its dipole with `atoms.get_dipole_moment()`. Symmetrix-XL 0.1 on PyPI publishes Linux wheels. The Apple GPU backend (`metal=True`) and MACE-POLAR-1 need a source build of Symmetrix-XL 0.2 ({doc}`installation`).

Install matscipy with Symmetrix (the `symmetrix` extra does). Without it, Symmetrix falls back to ASE's Python neighbor list, and atombatch warns: the batched path rebuilds a neighbor list for every structure in every round, which then dominates the run time. With the `metal-preview-1` wheel on an M1 Max, 64 FIRE relaxations of 40-atom SrTiO{sub}`3` cells took 54 s batched against 18 s sequential without matscipy, and 10.6 s against 20.9 s with it, with identical step counts.

## TorchSim models: MACE, fairchem, SevenNet, ORB, MatterSim, and others

[TorchSim](https://github.com/TorchSim/torch-sim) maintains batched interfaces for MACE, fairchem, SevenNet, ORB, MatterSim, NequIP, Nequix, and metatomic models. atombatch joins the structures of a round into one TorchSim state, calls the model once, and splits the energies, forces, and stresses back per structure.

```python
import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

model = MaceModel(
    mace_mp(model="medium", return_raw_model=True, default_dtype="float64"),
    device=torch.device("cuda"),
    dtype=torch.float64,
    compute_stress=True,
)
scheduler.add_model("mace", model, max_atoms=20_000)
```

Tasks that read stress, such as relaxations with a cell filter, need a model constructed with `compute_stress=True`. Otherwise those tasks fail with a `ValueError` and the other tasks continue. For MACE-MP in float64, batched results agree with MACE's own ASE calculator to rounding (energies exactly, forces to about 1e-15 eV/Å), and batched cell relaxations take the same steps as independent ASE runs.

## Your own batched model

A model with its own batched inference connects through a function that takes a list of `Atoms` and the requested properties and returns one dictionary per structure:

```python
def evaluate(structures, properties):
    batch = my_model.prepare(structures)
    output = my_model(batch)
    return [
        {"energy": energy, "forces": forces}
        for energy, forces in zip(output.energies, output.split_forces())
    ]

scheduler.add_model("mine", evaluate)
```

Each dictionary holds at least `energy` (float) and `forces` (array of shape `(n_atoms, 3)`), and `stress` in ASE Voigt order when `"stress"` is in `properties`. Validate a new backend against the model's own ASE calculator before relying on it.

## Batch size

`max_atoms` bounds the atoms in one call. When the pending requests of a round exceed it, they are split into several calls in that round. A structure larger than `max_atoms` is evaluated in a call of its own. Requests that need stress and requests that do not are evaluated in separate calls.
