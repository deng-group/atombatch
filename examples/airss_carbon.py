"""Random structure search for 8-atom carbon cells at 20 GPa: AIRSS and hot-AIRSS.

48 AIRSS samples (random structure -> relaxation) and 48 hot-AIRSS samples
(random structure -> relaxation -> 2000 K Langevin anneal -> relaxation) run
in one scheduler, so samples in every stage share each batched model call.
The minima are ranked by enthalpy per atom, labelled by space group, and the
two methods are compared by how often a sample reaches the lowest structure.
At 20 GPa diamond is the stable form of carbon. At zero pressure, a model
without a dispersion correction does not bind graphite layers or chains, and
relaxations expand into separated fragments; searches are therefore run under
pressure, as is usual.

Random structures come from the AIRSS ``buildcell`` program when it is on the
PATH (2 to 4 symmetry operations), and otherwise from the simple built-in
generator without symmetry.

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python airss_carbon.py
"""

import os
import shutil
import sys
import time

import spglib
from symmetrix import Symmetrix

from atombatch import Scheduler, airss, hot_airss
from atombatch.workflows.airss import buildcell, random_structure, rank

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
SAMPLES = 48
PRESSURE = 20.0  # GPa
SEED = """
%BLOCK LATTICE_CART
4 0 0
0 4 0
0 0 4
%ENDBLOCK LATTICE_CART
#VARVOL=5.5
#NFORM=1
%BLOCK POSITIONS_FRAC
C 0 0 0 # C1 % NUM=8
%ENDBLOCK POSITIONS_FRAC
#MINSEP=1.3
#SYMMOPS=2-4
"""


def calculator():
    return Symmetrix(MODEL, dtype="float32", metal=sys.platform == "darwin")


def generator():
    if shutil.which("buildcell"):
        return "buildcell", buildcell(SEED)

    def generate(rng):
        return random_structure("C8", volume_per_atom=5.5, minsep=1.3, rng=rng)

    return "built-in generator", generate


def space_group(atoms):
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    return spglib.get_spacegroup(cell, symprec=0.1) or "?"


def main():
    source, generate = generator()
    scheduler = Scheduler()
    scheduler.add_model("mace", calculator())
    for index in range(SAMPLES):
        scheduler.submit(
            airss(generate, seed=index, pressure=PRESSURE),
            "mace",
            name=f"airss-{index}",
        )
        scheduler.submit(
            hot_airss(
                generate,
                seed=index,
                pressure=PRESSURE,
                temperature_K=2000.0,
                md_steps=200,
            ),
            "mace",
            name=f"hot-{index}",
        )
    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error
    by_method = {
        "AIRSS": [r.value for r in results if r.name.startswith("airss")],
        "hot-AIRSS": [r.value for r in results if r.name.startswith("hot")],
    }
    evaluations = sum(size for _, size, _ in scheduler.batch_log)
    print(
        f"{len(results)} samples from {source} in {seconds:.0f} s: "
        f"{evaluations} evaluations in {len(scheduler.batch_log)} batched calls\n"
    )

    ranked = rank([r.value for r in results])
    best = ranked[0]["result"]["enthalpy_per_atom"]
    print("Lowest minima (all samples)")
    print("  dH (meV/atom)  V (A^3/atom)  space group      found")
    for entry in ranked[:6]:
        result = entry["result"]
        print(
            f"  {1000 * (result['enthalpy_per_atom'] - best):13.1f}"
            f"  {result['volume_per_atom']:12.2f}  {space_group(result['atoms']):15s}"
            f"  {entry['count']:5d}"
        )

    print("\nSamples within 10 meV/atom of the lowest minimum")
    for method, values in by_method.items():
        hits = sum(
            v["converged"] and v["enthalpy_per_atom"] - best < 0.010 for v in values
        )
        print(f"  {method:10s} {hits:3d} of {len(values)}")


if __name__ == "__main__":
    main()
