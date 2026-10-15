"""Phonons of cubic SrTiO3 at several lattice constants, computed together.

Cubic SrTiO3 has two soft modes: the zone-boundary mode at R, whose
instability drives the antiferrodistortive transition and grows as the
lattice contracts, and the polar mode at Gamma, which softens as the lattice
expands. Imaginary frequencies are printed as negative numbers. Phonopy
generates the displaced supercells for five lattice constants, and all of
them are evaluated in one batched run with ``displacement_forces``. Force
constants and frequencies are then computed with phonopy as usual.

Needs phonopy (``pip install "atombatch[phonons]"``).

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python phonons.py
"""

import os
import sys
import time

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from symmetrix import Symmetrix

from atombatch import Scheduler, displacement_forces

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
LATTICE_CONSTANTS = [3.85, 3.88, 3.91, 3.94, 3.97]
THZ_TO_CM = 33.356


def calculator():
    # Phonons need accurate small forces, so evaluate in FP64 where the
    # device supports it (Metal is FP32 only).
    if sys.platform == "darwin":
        return Symmetrix(MODEL, dtype="float32", metal=True)
    return Symmetrix(MODEL, dtype="float64")


def phonon(a):
    unit_cell = PhonopyAtoms(
        symbols=["Sr", "Ti", "O", "O", "O"],
        cell=np.eye(3) * a,
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
    )
    result = Phonopy(unit_cell, supercell_matrix=np.eye(3, dtype=int) * 2)
    result.generate_displacements(distance=0.01)
    return result


def main():
    phonons = {a: phonon(a) for a in LATTICE_CONSTANTS}
    scheduler = Scheduler()
    scheduler.add_model("mace", calculator())
    for a, item in phonons.items():
        scheduler.submit(displacement_forces(item), "mace", name=f"a = {a}")

    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    assert all(result.ok for result in results), [r.error for r in results]
    cells = sum(size for _, size, _ in scheduler.batch_log)
    print(
        f"{cells} displaced supercells for {len(phonons)} lattice constants "
        f"in {len(scheduler.batch_log)} batched call(s), {seconds:.1f} s\n"
    )

    print("a (A)   lowest optical mode at Gamma (cm^-1)   lowest mode at R (cm^-1)")
    for a, item in phonons.items():
        item.produce_force_constants(show_drift=False)
        gamma, r_point = (
            item.get_frequencies(q) * THZ_TO_CM for q in ([0, 0, 0], [0.5, 0.5, 0.5])
        )
        # The three acoustic modes at Gamma are those closest to zero.
        optical = gamma[np.argsort(np.abs(gamma))[3:]]
        print(f"{a:5.2f}   {optical.min():33.1f}   {r_point.min():34.1f}")


if __name__ == "__main__":
    main()
