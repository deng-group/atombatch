"""Swap Monte Carlo with replica exchange."""

import numpy as np
from ase import units


def swap_monte_carlo(
    atoms,
    temperatures_K,
    steps,
    sites=None,
    exchange_interval=10,
    rng=None,
):
    """Sample site occupancies by Metropolis swaps, with replica exchange.

    One replica runs at each temperature in ``temperatures_K``. A step
    proposes one swap of two unlike atoms per replica, chosen uniformly
    among unlike pairs of ``sites`` (default: all atoms), and evaluates the
    trial configurations of all replicas in one request. Every
    ``exchange_interval`` steps, neighboring temperatures attempt to
    exchange configurations. Positions are not relaxed: energies are those
    of the configurations on the fixed lattice of ``atoms``.

    The result holds the final configuration and energy at each temperature,
    the lowest-energy configuration visited, swap and exchange acceptance
    ratios, and the energy trace with one row per step.
    """
    temperatures = np.asarray(temperatures_K, dtype=float)
    candidates = np.arange(len(atoms)) if sites is None else np.asarray(sites)
    generator = np.random.default_rng(rng)

    def beta(temperature):
        return np.inf if temperature == 0 else 1.0 / (units.kB * temperature)

    def accept(log_ratio):
        # Metropolis test on log(acceptance ratio); NaN arises only from
        # inf - inf at zero temperature and means no energy change.
        if np.isnan(log_ratio) or log_ratio >= 0:
            return True
        return generator.random() < np.exp(log_ratio)

    def propose(numbers):
        while True:
            first, second = generator.choice(candidates, 2, replace=False)
            if numbers[first] != numbers[second]:
                return first, second

    def task(context):
        if len(set(atoms.numbers[candidates])) < 2:
            raise ValueError("Swap Monte Carlo needs two species among the sites.")
        replicas = [atoms.copy() for _ in temperatures]
        energies = np.array([r["energy"] for r in context.evaluate(replicas)])
        lowest = (energies.min(), replicas[int(energies.argmin())].copy())
        accepted = np.zeros(len(replicas))
        exchanges = np.zeros(max(len(replicas) - 1, 0))
        attempts = np.zeros_like(exchanges)
        trace = [energies.copy()]
        for step in range(1, steps + 1):
            trials = []
            for replica in replicas:
                trial = replica.copy()
                first, second = propose(trial.numbers)
                numbers = trial.numbers
                numbers[[first, second]] = numbers[[second, first]]
                trial.numbers = numbers
                trials.append(trial)
            results = context.evaluate(trials)
            for index, (trial, result) in enumerate(zip(trials, results)):
                change = result["energy"] - energies[index]
                if accept(-beta(temperatures[index]) * change):
                    replicas[index] = trial
                    energies[index] = result["energy"]
                    accepted[index] += 1
                    if energies[index] < lowest[0]:
                        lowest = (energies[index], trial.copy())
            if exchange_interval and step % exchange_interval == 0:
                offset = (step // exchange_interval) % 2
                for low in range(offset, len(replicas) - 1, 2):
                    high = low + 1
                    attempts[low] += 1
                    log_ratio = (beta(temperatures[low]) - beta(temperatures[high])) * (
                        energies[low] - energies[high]
                    )
                    if accept(log_ratio):
                        replicas[low], replicas[high] = replicas[high], replicas[low]
                        energies[[low, high]] = energies[[high, low]]
                        exchanges[low] += 1
            trace.append(energies.copy())
        for replica in replicas:
            replica.calc = None
        return {
            "replicas": replicas,
            "energies": energies,
            "lowest_energy": lowest[0],
            "lowest": lowest[1],
            "acceptance": accepted / max(steps, 1),
            "exchange_acceptance": np.divide(
                exchanges, attempts, out=np.zeros_like(exchanges), where=attempts > 0
            ),
            "trace": np.array(trace),
        }

    return task
