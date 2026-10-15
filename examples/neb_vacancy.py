"""Oxygen-vacancy migration barriers in SrTiO3 from several NEB bands at once.

An oxygen vacancy in a 3x3x3 SrTiO3 supercell hops to a neighboring oxygen
site. Three hops are compared: in pure SrTiO3, and with one Ba on the A site
next to the start or the end of the path. The endpoints of all hops are
relaxed in a first batched run, and the three climbing-image NEB bands are
optimized together in a second; each step evaluates the moving images of all
bands in one call.

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python neb_vacancy.py
"""

import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from symmetrix import Symmetrix

from atombatch import Scheduler, neb, relax

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
IMAGES = 5


def calculator():
    return Symmetrix(MODEL, dtype="float32", metal=sys.platform == "darwin")


def perovskite(a=3.905, repeat=3):
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


def hop(host, barium_near):
    """Return (initial, final) of a vacancy hop, optionally with one Ba.

    The Ba replaces a Sr next to the start (or end) of the path that is not
    also next to the other end, so the two cases are mirror images: their
    reaction energies must be equal and opposite.
    """
    oxygens = [atom.index for atom in host if atom.symbol == "O"]
    vacancy = oxygens[0]
    distances = host.get_distances(vacancy, oxygens, mic=True)
    target = oxygens[int(np.argsort(distances)[1])]  # nearest other oxygen
    atoms = host.copy()
    if barium_near is not None:
        near, far = (vacancy, target) if barium_near == "start" else (target, vacancy)
        strontium = [atom.index for atom in atoms if atom.symbol == "Sr"]
        to_near = atoms.get_distances(near, strontium, mic=True)
        to_far = atoms.get_distances(far, strontium, mic=True)
        candidates = np.flatnonzero(np.isclose(to_near, to_near.min()))
        chosen = candidates[np.argmax(to_far[candidates])]
        atoms.symbols[strontium[chosen]] = "Ba"
    vacancy_position = atoms.positions[vacancy].copy()
    del atoms[vacancy]
    target -= target > vacancy
    initial = atoms.copy()
    final = atoms.copy()
    final.positions[target] = vacancy_position
    return initial, final


def main():
    host = perovskite()
    hops = {
        "pure SrTiO3": hop(host, None),
        "Ba next to start": hop(host, "start"),
        "Ba next to end": hop(host, "end"),
    }
    model = calculator()

    # Stage 1: relax all endpoints together (fixed cell).
    scheduler = Scheduler()
    scheduler.add_model("mace", model)
    for name, (initial, final) in hops.items():
        scheduler.submit(
            relax(initial, fmax=0.02, optimizer=FIRE), "mace", f"{name}/initial"
        )
        scheduler.submit(
            relax(final, fmax=0.02, optimizer=FIRE), "mace", f"{name}/final"
        )
    start = time.perf_counter()
    endpoints = {}
    for result in scheduler.run():
        if not result.ok:
            raise result.error
        endpoints[result.name] = result.value
    print(f"endpoints relaxed in {time.perf_counter() - start:.1f} s")

    # Stage 2: one climbing-image NEB band per hop, optimized together.
    scheduler = Scheduler()
    scheduler.add_model("mace", model)
    for name in hops:
        initial = endpoints[f"{name}/initial"]["atoms"]
        final = endpoints[f"{name}/final"]["atoms"]
        images = [initial] + [initial.copy() for _ in range(IMAGES)] + [final]
        scheduler.submit(
            neb(
                images,
                fmax=0.05,
                steps=500,
                neb_kwargs={"climb": True},
                interpolate="idpp",
            ),
            "mace",
            name=name,
        )
    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error
    calls = len(scheduler.batch_log)
    largest = max(size for _, size, _ in scheduler.batch_log)
    print(
        f"{len(results)} NEB bands in {seconds:.1f} s, {calls} batched calls of up to {largest} images\n"
    )

    print(
        f"{'hop':18s} {'barrier (eV)':>12s} {'reaction energy (eV)':>21s} {'steps':>6s} converged"
    )
    for result in results:
        energies = np.array(result.value["energies"])
        print(
            f"{result.name:18s} {energies.max() - energies[0]:12.3f} "
            f"{energies[-1] - energies[0]:21.3f} {result.value['steps']:6d} {result.value['converged']}"
        )


if __name__ == "__main__":
    main()
