"""Relax every Ba ordering in a SrTiO3 supercell in one batched run.

Two of the eight Sr sites of a 2x2x2 SrTiO3 supercell are replaced by Ba,
which gives 28 orderings. Each is relaxed with its cell (FIRE with a
Frechet cell filter); the forces and stresses of all orderings that are
still running are evaluated together in one Symmetrix call per step.

Symmetry-equivalent orderings must reach the same energy, so the output
groups them by energy: the pairs of Ba at distances a, a*sqrt(2), and
a*sqrt(3) form three groups of 12, 12, and 4 orderings.

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python relax_orderings.py
"""

import itertools
import os
import sys
import time

from ase import Atoms
from ase.filters import FrechetCellFilter
from symmetrix import Symmetrix

from atombatch import Scheduler, relax

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")


def calculator():
    # Apple silicon runs the Metal backend in FP32; CUDA builds use the GPU
    # automatically.
    return Symmetrix(MODEL, dtype="float32", metal=sys.platform == "darwin")


def perovskite(a=3.905, repeat=2):
    return Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ).repeat(repeat)


def main():
    host = perovskite()
    a_sites = [atom.index for atom in host if atom.symbol == "Sr"]
    orderings = list(itertools.combinations(a_sites, 2))

    scheduler = Scheduler(max_active=64)
    scheduler.add_model("mace", calculator())
    for pair in orderings:
        atoms = host.copy()
        atoms.symbols[list(pair)] = "Ba"
        scheduler.submit(
            relax(atoms, fmax=0.02, steps=300, cell_filter=FrechetCellFilter),
            "mace",
            name=f"Ba{pair[0]}-Ba{pair[1]}",
        )

    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error

    evaluations = sum(size for _, size, _ in scheduler.batch_log)
    print(
        f"{len(results)} relaxations, {evaluations} evaluations in "
        f"{len(scheduler.batch_log)} batched calls, {seconds:.1f} s"
    )
    print(f"all converged: {all(r.value['converged'] for r in results)}")

    groups = {}
    for result in results:
        atoms = result.value["atoms"]
        ba = [atom.index for atom in atoms if atom.symbol == "Ba"]
        distance = atoms.get_distance(*ba, mic=True)
        key = round(result.value["energy"], 3)
        groups.setdefault(key, []).append(distance)
    lowest = min(groups)
    print("\nEnergy above lowest (meV/cell)  orderings  Ba-Ba distance (A)")
    for energy in sorted(groups):
        distances = groups[energy]
        print(
            f"{1000 * (energy - lowest):26.1f}  {len(distances):9d}"
            f"  {min(distances):.2f}"
        )


if __name__ == "__main__":
    main()
