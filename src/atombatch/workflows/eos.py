"""Equations of state from batched scaled cells."""

import numpy as np
from ase.eos import EquationOfState


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
