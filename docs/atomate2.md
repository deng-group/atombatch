# atomate2

[atomate2](https://github.com/materialsproject/atomate2) defines workflows as *makers*: dataclasses of settings whose `make` method returns a jobflow job. Its makers that run through ASE, such as `ForceFieldRelaxMaker`, `ForceFieldStaticMaker`, and `AseRelaxMaker` subclasses, compute one structure per job and call the model once per optimizer step. {func}`~atombatch.integrations.atomate2.atomate2_task` runs such a maker inside an atombatch task, with its calculator replaced by a batched one, so many jobs share each model call. The task returns the document the atomate2 job would return.

```python
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from symmetrix import Symmetrix

from atombatch import Scheduler
from atombatch.integrations.atomate2 import atomate2_task

symmetrix = {"model_file": "mace-mp-0b-medium.json", "dtype": "float32"}
maker = ForceFieldRelaxMaker(
    force_field_name={"@module": "symmetrix", "@callable": "Symmetrix"},
    calculator_kwargs=symmetrix,
    relax_cell=True,
)

scheduler = Scheduler(max_active=64)
scheduler.add_model("symmetrix", Symmetrix(**symmetrix))
for structure in structures:  # pymatgen Structures
    scheduler.submit(atomate2_task(maker, structure), "symmetrix")
documents = [result.value for result in scheduler.run()]  # ForceFieldTaskDocuments
```

Install with `pip install "atombatch[atomate2]"` (Python 3.11 or later).

## What runs where

The model registered with {meth}`~atombatch.Scheduler.add_model` performs every evaluation. The maker's force-field settings are not used to build a calculator inside atombatch; they label the documents and let atomate2 build the same calculator when the maker runs on its own. Naming the calculator class, as above, records the model that was actually used (`forcefield_name: Symmetrix`).

The job runs as a plain function, not as a jobflow `Job`: it is not added to a flow or stored in a `JobStore` by atombatch. The documents are the ones the jobs would return and can be stored as usual.

## Equivalence and performance

The tests run `AseRelaxMaker` and `ForceFieldRelaxMaker` jobs (with and without cell relaxation) one after another and batched, and compare the documents: same document type, final energy, structure, cell, and number of ionic steps. Each recorded state costs one evaluation, because energies, forces, and stresses are requested together; atomate2's trajectory observer reads the stress of periodic structures at every step, also when the cell is fixed.

`examples/atomate2_relax.py` relaxes the 28 Ba orderings of a 2x2x2 SrTiO{sub}`3` supercell with a Symmetrix `ForceFieldRelaxMaker`. On an Apple M1 Max GPU, the jobs took 24.7 s one after another and 10.1 s batched (2.4 times faster), with the same number of steps and energies equal to 2e-5 eV (FP32 rounding). atomate2's own work per step (trajectory recording, documents) adds to the batched run, which takes about twice as long as the same relaxations with {func}`~atombatch.relax`.

## Thread safety

atomate2 silences the optimizer log by redirecting `sys.stdout`, which is shared by all threads. Concurrent relaxations would restore each other's buffers and leave `sys.stdout` redirected after the run. During the relaxation, atombatch therefore sends the optimizer log to no file and lets atomate2 redirect stdout to itself; the documents record the maker's own `relax_kwargs`.
