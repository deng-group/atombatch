"""Task functions for common ASE workflows.

Each function returns a task for :meth:`Scheduler.submit`: a callable that
receives a :class:`TaskContext` and runs ordinary ASE code with batched
calculators. Structures are copied, so the caller's objects are unchanged.
"""

import numpy as np
from ase import Atoms, units
from ase.eos import EquationOfState
from ase.optimize import FIRE


def _result(atoms, properties=("energy", "forces")):
    # The calculator requests all declared properties at once, so the first
    # read evaluates and the others come from its cache.
    result = {"atoms": atoms}
    for prop in properties:
        if prop == "energy":
            result["energy"] = atoms.get_potential_energy()
        elif prop == "forces":
            result["forces"] = atoms.get_forces()
        elif prop == "stress":
            result["stress"] = atoms.get_stress()
    return result


def single_point(atoms, properties=("energy", "forces")):
    """Evaluate one structure."""

    def task(context):
        image = atoms.copy()
        image.calc = context.calculator(properties=properties)
        return _result(image, properties)

    return task


def relax(
    atoms,
    fmax=0.05,
    steps=500,
    optimizer=FIRE,
    optimizer_kwargs=None,
    cell_filter=None,
    filter_kwargs=None,
):
    """Relax one structure with any ASE optimizer, optionally with its cell."""

    def task(context):
        properties = (
            ("energy", "forces", "stress") if cell_filter else ("energy", "forces")
        )
        image = atoms.copy()
        image.calc = context.calculator(properties=properties)
        target = cell_filter(image, **(filter_kwargs or {})) if cell_filter else image
        dynamics = optimizer(target, **{"logfile": None, **(optimizer_kwargs or {})})
        converged = dynamics.run(fmax=fmax, steps=steps)
        result = _result(image, properties)
        result.update(converged=bool(converged), steps=dynamics.nsteps)
        return result

    return task


def neb(
    images,
    fmax=0.05,
    steps=500,
    optimizer=FIRE,
    optimizer_kwargs=None,
    neb_kwargs=None,
    interpolate=None,
):
    """Optimize one NEB band built from ``images`` (endpoints included).

    ``neb_kwargs`` are passed to ``ase.mep.NEB`` (for example ``climb=True``);
    ``interpolate`` is ``None`` for images that are already placed, or an
    ``NEB.interpolate`` method such as ``"linear"`` or ``"idpp"``.
    """

    def task(context):
        from ase.mep import NEB

        band_images = [image.copy() for image in images]
        # NEB queries images in turn; linked calculators submit the whole band.
        context.linked_calculators(band_images)
        band = NEB(band_images, **(neb_kwargs or {}))
        if interpolate is not None:
            band.interpolate(method=interpolate)
        dynamics = optimizer(band, **{"logfile": None, **(optimizer_kwargs or {})})
        converged = dynamics.run(fmax=fmax, steps=steps)
        return {
            "images": band_images,
            "energies": [image.get_potential_energy() for image in band_images],
            "converged": bool(converged),
            "steps": dynamics.nsteps,
        }

    return task


def molecular_dynamics(atoms, dynamics, steps, observers=(), **dynamics_kwargs):
    """Run ``steps`` of any ASE molecular-dynamics integrator.

    ``dynamics`` is an integrator class such as ``VelocityVerlet`` or
    ``Langevin``; ``dynamics_kwargs`` are passed to it. ``observers`` are
    ``(function, interval)`` pairs attached with ``dynamics.attach``; each
    function receives the moving ``Atoms``.
    """

    def task(context):
        image = atoms.copy()
        image.calc = context.calculator()
        integrator = dynamics(image, **{"logfile": None, **dynamics_kwargs})
        for function, interval in observers:
            integrator.attach(function, interval=interval, atoms=image)
        integrator.run(steps)
        result = _result(image)
        result["steps"] = integrator.nsteps
        return result

    return task


def equation_of_state(atoms, scales=None, eos="birchmurnaghan"):
    """Evaluate isotropically scaled copies of ``atoms`` and fit an EOS.

    ``scales`` are linear scale factors of the cell (default 0.97 to 1.03 in
    seven points). All scaled cells are evaluated in one request. The fit uses
    ``ase.eos.EquationOfState``; the bulk modulus is in eV/Å^3.
    """
    factors = np.linspace(0.97, 1.03, 7) if scales is None else np.asarray(scales)

    def task(context):
        cells = []
        for factor in factors:
            image = atoms.copy()
            image.set_cell(atoms.cell * factor, scale_atoms=True)
            cells.append(image)
        results = context.evaluate(cells)
        volumes = [image.get_volume() for image in cells]
        energies = [result["energy"] for result in results]
        volume, energy, bulk_modulus = EquationOfState(volumes, energies, eos).fit()
        return {
            "volumes": volumes,
            "energies": energies,
            "volume": volume,
            "energy": energy,
            "bulk_modulus": bulk_modulus,
        }

    return task


def displacement_forces(phonon):
    """Compute forces on the displaced supercells of a phonopy or phono3py object.

    ``phonon`` is a ``Phonopy`` or ``Phono3py`` object on which
    ``generate_displacements`` has been called. All displaced supercells are
    evaluated in one request and the forces are assigned to ``phonon.forces``
    (and ``phonon.phonon_forces`` when phono3py uses a separate fc2
    supercell). The task returns ``phonon``; force constants are then
    produced as usual, for example with ``produce_force_constants`` or
    ``produce_fc3``.
    """

    def task(context):
        sets = [phonon.supercells_with_displacements]
        separate = getattr(phonon, "phonon_supercells_with_displacements", None)
        if separate is not None:
            sets.append(separate)
        # phono3py marks displacement pairs beyond its cutoff with None; their
        # forces are not used and are set to zero.
        structures = [
            _ase_atoms(cell) for cells in sets for cell in cells if cell is not None
        ]
        results = iter(context.evaluate(structures))
        forces = []
        for cells in sets:
            size = len(next(cell for cell in cells if cell is not None))
            forces.append(
                np.array(
                    [
                        next(results)["forces"]
                        if cell is not None
                        else np.zeros((size, 3))
                        for cell in cells
                    ]
                )
            )
        phonon.forces = forces[0]
        if separate is not None:
            phonon.phonon_forces = forces[1]
        return phonon

    return task


def _ase_atoms(cell):
    return Atoms(
        cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True
    )


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
