"""Nudged elastic band optimization."""

from ase.optimize import FIRE


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
