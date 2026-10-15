"""Run atomate2 force-field relaxations batched with atombatch.

The 28 Ba orderings of a 2x2x2 SrTiO3 supercell (as in relax_orderings.py)
are relaxed with an atomate2 ``ForceFieldRelaxMaker`` whose calculator is
Symmetrix. The jobs run twice: one after another, as atomate2 runs them, and
batched with atombatch, where each step evaluates all relaxations still
running in one Symmetrix call. Both return atomate2 ``ForceFieldTaskDocument``s,
which are compared.

Needs atomate2 (``pip install "atombatch[atomate2]"``).

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python atomate2_relax.py
"""

import itertools
import os
import sys
import time

from ase import Atoms
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.io.ase import AseAtomsAdaptor
from symmetrix import Symmetrix

from atombatch import Scheduler
from atombatch.integrations.atomate2 import atomate2_task

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
SYMMETRIX = {"model_file": MODEL, "dtype": "float32", "metal": sys.platform == "darwin"}


def orderings():
    host = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[3.905, 3.905, 3.905],
        pbc=True,
    ).repeat(2)
    a_sites = [atom.index for atom in host if atom.symbol == "Sr"]
    structures = []
    for pair in itertools.combinations(a_sites, 2):
        atoms = host.copy()
        atoms.symbols[list(pair)] = "Ba"
        structures.append(AseAtomsAdaptor.get_structure(atoms))
    return structures


def main():
    structures = orderings()
    # The maker names Symmetrix as its calculator, so atomate2 can build it
    # for the sequential run and the documents record which model was used.
    maker = ForceFieldRelaxMaker(
        force_field_name={"@module": "symmetrix", "@callable": "Symmetrix"},
        calculator_kwargs=SYMMETRIX,
        relax_cell=True,
        steps=300,
        relax_kwargs={"fmax": 0.02},
    )

    # atomate2 as usual: the job function of each structure in turn.
    start = time.perf_counter()
    sequential = [type(maker).make.original(maker, s) for s in structures]
    sequential_seconds = time.perf_counter() - start

    # atombatch: the same jobs, batched.
    scheduler = Scheduler(max_active=64)
    scheduler.add_model("symmetrix", Symmetrix(**SYMMETRIX))
    for structure in structures:
        scheduler.submit(atomate2_task(maker, structure), "symmetrix")
    start = time.perf_counter()
    results = scheduler.run()
    batched_seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error
    batched = [result.value for result in results]

    print(f"{len(structures)} atomate2 ForceFieldRelaxMaker jobs")
    print(f"  one after another   {sequential_seconds:6.1f} s")
    print(
        f"  batched             {batched_seconds:6.1f} s "
        f"({len(scheduler.batch_log)} batched calls), "
        f"{sequential_seconds / batched_seconds:.1f}x faster"
    )
    same_steps = all(
        a.output.n_steps == b.output.n_steps for a, b in zip(batched, sequential)
    )
    largest = max(
        abs(a.output.energy - b.output.energy) for a, b in zip(batched, sequential)
    )
    print(f"  same steps: {same_steps}, largest energy difference {largest:.1e} eV")
    print(
        f"  document type: {type(batched[0]).__name__}, model: {batched[0].forcefield_name}"
    )


if __name__ == "__main__":
    main()
