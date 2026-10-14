"""Task functions for common ASE workflows.

Each function returns a task for :meth:`Scheduler.submit`: a callable that
receives a :class:`TaskContext` and runs ordinary ASE code with batched
calculators. Structures are copied, so the caller's objects are unchanged.
"""

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
