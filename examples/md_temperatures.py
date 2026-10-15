"""Oxygen-vacancy diffusion in SrTiO3 at several temperatures, run together.

One Langevin trajectory per temperature runs for a 3x3x3 SrTiO3 supercell
with one oxygen vacancy. All trajectories advance together, and each MD step
evaluates the forces of all temperatures in one call. An observer records
the oxygen displacements, and the output counts the oxygen atoms that have
hopped into a neighboring site (moved by more than 1.5 A), which happens by
exchange with the vacancy. Diffusion coefficients would need much longer
runs; this example shows how to batch trajectories and attach observers.

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python md_temperatures.py
"""

import os
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from symmetrix import Symmetrix

from atombatch import Scheduler, molecular_dynamics

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
TEMPERATURES = [1200, 1500, 1800, 2100]
STEPS = 2000  # 2 fs steps: 4 ps per temperature
INTERVAL = 50


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


def displacement_recorder(atoms, records):
    """Return an observer that appends the oxygen displacements to ``records``."""
    oxygens = atoms.symbols == "O"
    start = atoms.positions[oxygens].copy()

    def record(moving):
        # Positions are not wrapped during the run, so differences are
        # displacements.
        records.append(np.linalg.norm(moving.positions[oxygens] - start, axis=1))

    return record


def main():
    host = perovskite()
    del host[next(atom.index for atom in host if atom.symbol == "O")]

    scheduler = Scheduler()
    scheduler.add_model("mace", calculator())
    traces = {}
    for temperature in TEMPERATURES:
        atoms = host.copy()
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature, rng=np.random.default_rng(temperature)
        )
        traces[temperature] = []
        scheduler.submit(
            molecular_dynamics(
                atoms,
                Langevin,
                STEPS,
                observers=[
                    (displacement_recorder(atoms, traces[temperature]), INTERVAL)
                ],
                timestep=2 * units.fs,
                temperature_K=temperature,
                friction=0.01 / units.fs,
                rng=np.random.default_rng(1000 + temperature),
            ),
            "mace",
            name=f"{temperature} K",
        )

    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error
    atom_steps = sum(atoms for _, _, atoms in scheduler.batch_log)
    print(
        f"{len(results)} trajectories x {STEPS} steps in {seconds:.1f} s "
        f"({len(scheduler.batch_log)} batched calls, "
        f"{1e6 * seconds / atom_steps:.1f} us per atom-step)\n"
    )

    print("T (K)   O hopped after 4 ps   O mean-squared displacement (A^2)")
    for temperature in TEMPERATURES:
        displacements = traces[temperature][-1]
        hopped = int((displacements > 1.5).sum())
        print(f"{temperature:5d}   {hopped:17d}   {(displacements**2).mean():33.3f}")


if __name__ == "__main__":
    main()
