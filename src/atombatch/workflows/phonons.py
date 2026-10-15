"""Force sets for phonopy and phono3py from batched displaced supercells."""

import numpy as np
from ase import Atoms


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
