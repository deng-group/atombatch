"""Ab initio random structure searching (AIRSS) and hot-AIRSS.

AIRSS (Pickard and Needs) generates random but sensible structures and relaxes
each of them; the low-enthalpy minima found repeatedly are candidate ground
states. hot-AIRSS adds a short high-temperature molecular-dynamics anneal
between two relaxations, which lets a structure leave a poor basin before it
is relaxed again.

Every sample is one task, so the relaxations and anneals of all samples share
batched model calls, and finished samples are replaced from the queue::

    for index in range(200):
        scheduler.submit(airss(generate, seed=index, pressure=10.0), "model")
    ranked = rank([result.value for result in scheduler.run()])

``generate(rng)`` returns a random ``Atoms``. :func:`buildcell` wraps the
AIRSS ``buildcell`` program through airsspy, and :func:`random_structure` is a
simple generator without dependencies.
"""

import numpy as np
from ase import Atoms, units
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

_PROPERTIES = ("energy", "forces", "stress")


def random_structure(
    symbols, volume_per_atom, minsep, rng=None, varvol=0.1, max_attempts=1000
):
    """Return random periodic ``Atoms`` with atoms at least ``minsep`` apart.

    The cell has random lengths and angles (between 60 and 120 degrees) and a
    volume of ``len(symbols) * volume_per_atom``, varied by up to ``varvol``
    (a fraction). Atoms are placed uniformly at random and rejected when
    closer than ``minsep`` (A) to an atom already placed, including periodic
    images. Unlike ``buildcell``, it imposes no symmetry.
    """
    rng = np.random.default_rng(rng)
    symbols = list(Atoms(symbols).get_chemical_symbols())
    for _ in range(max_attempts):
        lengths = rng.uniform(1.0, 2.0, 3)
        angles = rng.uniform(60.0, 120.0, 3)
        atoms = Atoms(cell=[*lengths, *angles], pbc=True)
        if atoms.cell.volume < 0.3 * np.prod(lengths):
            continue  # nearly flat cell
        volume = len(symbols) * volume_per_atom * rng.uniform(1 - varvol, 1 + varvol)
        atoms.set_cell(atoms.cell * (volume / atoms.cell.volume) ** (1 / 3))
        placed = _place(atoms, symbols, minsep, rng, max_attempts)
        if placed is not None:
            return placed
    raise RuntimeError(
        f"Could not place {len(symbols)} atoms {minsep} A apart; "
        "increase volume_per_atom or reduce minsep."
    )


def _place(cell, symbols, minsep, rng, max_attempts):
    atoms = cell.copy()
    for symbol in symbols:
        for _ in range(max_attempts):
            candidate = atoms + Atoms(
                symbol,
                scaled_positions=[rng.random(3)],
                cell=atoms.cell,
                pbc=True,
            )
            if len(candidate) == 1:
                atoms = candidate
                break
            distances = candidate.get_distances(
                len(candidate) - 1, range(len(candidate) - 1), mic=True
            )
            # The image of the new atom in the cell itself.
            image = np.min(np.linalg.norm(candidate.cell.array, axis=1))
            if distances.min() >= minsep and image >= minsep:
                atoms = candidate
                break
        else:
            return None
    return atoms


def buildcell(seed, attempts=10):
    """Return a generator that runs the AIRSS ``buildcell`` program on ``seed``.

    ``seed`` is an airsspy ``SeedAtoms`` or the text of a buildcell seed
    (``.cell`` file). Needs airsspy and ``buildcell`` on the ``PATH``.
    buildcell draws its own random numbers, so ``rng`` is not used.
    """
    if isinstance(seed, str):
        lines = seed.splitlines()

        def generate(rng=None):
            return _run_buildcell(lines, attempts)

        return generate

    def generate(rng=None):
        for _ in range(attempts):
            atoms = seed.build_random_atoms(fail_ok=True)
            if atoms is not None:
                return atoms
        raise RuntimeError(f"buildcell failed {attempts} times for this seed.")

    return generate


def _run_buildcell(lines, attempts):
    import subprocess

    for _ in range(attempts):
        completed = subprocess.run(
            ["buildcell"],
            check=False,
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode == 0 and "%BLOCK" in completed.stdout.upper():
            return parse_cell(completed.stdout)
    raise RuntimeError(f"buildcell failed {attempts} times for this seed.")


def parse_cell(text):
    """Return the ``Atoms`` of a CASTEP ``.cell`` text written by ``buildcell``.

    Reads the ``LATTICE_ABC`` or ``LATTICE_CART`` block and the
    ``POSITIONS_FRAC`` or ``POSITIONS_ABS`` block (in angstrom), which is all
    that ``buildcell`` writes. ASE's CASTEP reader also handles this but looks
    for a CASTEP installation on every call and warns when there is none.
    """
    blocks = {}
    name = None
    for line in text.splitlines():
        stripped = line.split("#")[0].split("!")[0].strip()
        upper = stripped.upper()
        if upper.startswith("%BLOCK"):
            name = upper.split()[1]
            blocks[name] = []
        elif upper.startswith("%ENDBLOCK"):
            name = None
        elif name is not None and stripped:
            blocks[name].append(stripped.split())
    if "LATTICE_ABC" in blocks:
        lengths, angles = ([float(x) for x in row[:3]] for row in blocks["LATTICE_ABC"])
        cell = [*lengths, *angles]
    else:
        cell = [[float(x) for x in row[:3]] for row in blocks["LATTICE_CART"]]
    if "POSITIONS_FRAC" in blocks:
        rows = blocks["POSITIONS_FRAC"]
        return Atoms(
            [row[0] for row in rows],
            scaled_positions=[[float(x) for x in row[1:4]] for row in rows],
            cell=cell,
            pbc=True,
        )
    rows = blocks["POSITIONS_ABS"]
    return Atoms(
        [row[0] for row in rows],
        positions=[[float(x) for x in row[1:4]] for row in rows],
        cell=cell,
        pbc=True,
    )


def airss(
    generate,
    seed=None,
    pressure=0.0,
    fmax=0.05,
    steps=1000,
    optimizer=FIRE,
    optimizer_kwargs=None,
):
    """Generate one random structure and relax it at ``pressure`` (GPa).

    ``generate(rng)`` returns the random structure; ``rng`` is a NumPy
    generator seeded with ``seed``. Positions and cell are relaxed together
    (Frechet cell filter). The result holds the relaxed and initial
    structures, energy, enthalpy ``H = E + pV`` (eV per cell and per atom),
    volume per atom, convergence, and steps.
    """

    def task(context):
        rng = np.random.default_rng(seed)
        initial = generate(rng)
        atoms = initial.copy()
        atoms.calc = context.calculator(properties=_PROPERTIES)
        converged, nsteps = _relax(
            atoms, pressure, fmax, steps, optimizer, optimizer_kwargs
        )
        return _result(atoms, initial, pressure, converged, {"relax": nsteps})

    return task


def hot_airss(
    generate,
    seed=None,
    pressure=0.0,
    temperature_K=1000.0,
    md_steps=300,
    timestep=1.0 * units.fs,
    friction=0.02 / units.fs,
    prerelax=True,
    fmax=0.05,
    steps=1000,
    optimizer=FIRE,
    optimizer_kwargs=None,
):
    """Generate a random structure, relax it, anneal it with MD, and relax it again.

    The anneal is Langevin dynamics at ``temperature_K`` for ``md_steps``
    steps in the cell of the first relaxation; velocities and the Langevin
    noise are drawn from ``seed``. With ``prerelax=False`` the anneal starts
    from the random structure. The relaxations are as in :func:`airss`, and
    the result additionally records the steps of every stage.
    """

    def task(context):
        rng = np.random.default_rng(seed)
        initial = generate(rng)
        atoms = initial.copy()
        atoms.calc = context.calculator(properties=_PROPERTIES)
        stages = {}
        if prerelax:
            _, stages["prerelax"] = _relax(
                atoms, pressure, fmax, steps, optimizer, optimizer_kwargs
            )
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
        dynamics = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            friction=friction,
            rng=rng,
            logfile=None,
        )
        dynamics.run(md_steps)
        stages["md"] = dynamics.nsteps
        atoms.set_momenta(np.zeros_like(atoms.positions))
        converged, stages["relax"] = _relax(
            atoms, pressure, fmax, steps, optimizer, optimizer_kwargs
        )
        return _result(atoms, initial, pressure, converged, stages)

    return task


def _relax(atoms, pressure, fmax, steps, optimizer, optimizer_kwargs):
    target = FrechetCellFilter(atoms, scalar_pressure=pressure * units.GPa)
    dynamics = optimizer(target, **{"logfile": None, **(optimizer_kwargs or {})})
    converged = dynamics.run(fmax=fmax, steps=steps)
    return bool(converged), dynamics.nsteps


def _result(atoms, initial, pressure, converged, stages):
    energy = atoms.get_potential_energy()
    volume = atoms.get_volume()
    enthalpy = energy + pressure * units.GPa * volume
    final = atoms.copy()
    final.calc = None
    initial = initial.copy()
    initial.calc = None
    return {
        "atoms": final,
        "initial": initial,
        "energy": energy,
        "enthalpy": enthalpy,
        "enthalpy_per_atom": enthalpy / len(atoms),
        "volume_per_atom": volume / len(atoms),
        "pressure": pressure,
        "converged": converged,
        "steps": stages,
    }


def rank(results, enthalpy_tolerance=0.002, volume_tolerance=0.02, same=None):
    """Sort search results by enthalpy per atom and merge duplicates.

    Two results are taken as the same structure when their enthalpies per
    atom differ by less than ``enthalpy_tolerance`` (eV/atom) and their
    volumes per atom by less than ``volume_tolerance`` (a fraction); a
    function ``same(atoms1, atoms2)``, such as one built on pymatgen's
    ``StructureMatcher``, can confirm the match. Results that did not converge
    are skipped. Returns a list of dictionaries with the lowest-enthalpy
    ``result`` of each structure and the number of times it was found
    (``count``), lowest enthalpy first.
    """
    ordered = sorted(
        (result for result in results if result["converged"]),
        key=lambda result: result["enthalpy_per_atom"],
    )
    unique = []
    for result in ordered:
        for entry in unique:
            first = entry["result"]
            if (
                abs(result["enthalpy_per_atom"] - first["enthalpy_per_atom"])
                < enthalpy_tolerance
                and abs(result["volume_per_atom"] / first["volume_per_atom"] - 1)
                < volume_tolerance
                and (same is None or same(result["atoms"], first["atoms"]))
            ):
                entry["count"] += 1
                break
        else:
            unique.append({"result": result, "count": 1})
    return unique
