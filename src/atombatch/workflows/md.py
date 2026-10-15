"""Molecular dynamics with any ASE integrator."""

from .relax import _result


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
            integrator.attach(function, interval, image)
        integrator.run(steps)
        result = _result(image)
        result["steps"] = integrator.nsteps
        return result

    return task
