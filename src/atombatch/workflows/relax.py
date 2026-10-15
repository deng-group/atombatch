"""Single points and relaxations."""

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
