"""A-site ordering in (Sr,Ba)TiO3 by swap Monte Carlo with replica exchange.

A 3x3x3 SrTiO3 supercell with 9 of its 27 Sr replaced by Ba is sampled by
swapping Sr and Ba on the A sites at four temperatures. Each step evaluates
the trial configurations of all temperatures in one call, and neighboring
temperatures exchange configurations every 10 steps.

The output reports the fraction of Ba-Ba pairs among nearest A-site
neighbors; a random solid solution with one third Ba gives 1/3.

    SYMMETRIX_MODEL=mace-mp-0b-medium.json python swap_monte_carlo.py
"""

import os
import sys
import time

import numpy as np
from ase import Atoms
from symmetrix import Symmetrix

from atombatch import Scheduler, swap_monte_carlo

MODEL = os.environ.get("SYMMETRIX_MODEL", "mace-mp-0b-medium.json")
TEMPERATURES = [300, 600, 1200, 2400]
STEPS = 1000


def calculator():
    return Symmetrix(MODEL, dtype="float32", metal=sys.platform == "darwin")


def perovskite(a=3.94, repeat=3):
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


def ba_ba_fraction(atoms, a_sites):
    """Fraction of nearest A-site neighbors of Ba that are also Ba."""
    distances = atoms.get_all_distances(mic=True)[np.ix_(a_sites, a_sites)]
    nearest = np.isclose(distances, distances[distances > 0].min(), atol=0.1)
    barium = atoms.numbers[a_sites] == 56
    return nearest[barium][:, barium].sum() / nearest[barium].sum()


def main():
    atoms = perovskite()
    a_sites = [atom.index for atom in atoms if atom.symbol == "Sr"]
    chosen = np.random.default_rng(0).choice(a_sites, 9, replace=False)
    atoms.symbols[chosen] = "Ba"

    scheduler = Scheduler()
    scheduler.add_model("mace", calculator())
    scheduler.submit(
        swap_monte_carlo(
            atoms, TEMPERATURES, STEPS, sites=a_sites, exchange_interval=10, rng=1
        ),
        "mace",
    )
    start = time.perf_counter()
    results = scheduler.run()
    seconds = time.perf_counter() - start
    for result in results:
        if not result.ok:
            raise result.error
    value = results[0].value
    print(
        f"{STEPS} steps x {len(TEMPERATURES)} replicas in {seconds:.1f} s "
        f"({len(scheduler.batch_log)} batched calls)\n"
    )
    print(
        "T (K)   swap acceptance   mean E - min E (meV/cell)   Ba-Ba neighbor fraction"
    )
    trace = value["trace"][STEPS // 2 :]
    for index, temperature in enumerate(TEMPERATURES):
        print(
            f"{temperature:5d}   {value['acceptance'][index]:15.2f}   "
            f"{1000 * (trace[:, index].mean() - value['lowest_energy']):25.1f}   "
            f"{ba_ba_fraction(value['replicas'][index], a_sites):23.2f}"
        )
    print(f"\nreplica exchange acceptance: {np.round(value['exchange_acceptance'], 2)}")
    print(
        f"lowest configuration: Ba-Ba neighbor fraction {ba_ba_fraction(value['lowest'], a_sites):.2f}"
    )


if __name__ == "__main__":
    main()
