"""Backend interface, sequential ASE evaluation, and user functions."""

import copy

import numpy as np
from ase.calculators.calculator import all_changes

# Key in ``Atoms.info`` of a request that identifies the calculator (trajectory)
# it came from, so that backends can keep per-trajectory state between steps.
SOURCE_KEY = "atombatch_source"


class Backend:
    """Evaluate a list of ASE structures with one model.

    ``evaluate`` returns one result dictionary per structure with at least
    ``energy`` and ``forces``, plus ``stress`` (ASE Voigt order) when
    requested. ``max_atoms`` bounds the number of atoms in one call; ``None``
    means unbounded.
    """

    max_atoms = None

    def evaluate(self, structures, properties):
        raise NotImplementedError


class SequentialBackend(Backend):
    """Evaluate structures one at a time with any ASE calculator.

    This provides scheduling for calculators without batched evaluation; it
    does not reduce the per-call cost.
    """

    def __init__(self, calculator, max_atoms=None):
        self.calculator = calculator
        self.max_atoms = max_atoms

    def evaluate(self, structures, properties):
        results = []
        for atoms in structures:
            self.calculator.calculate(atoms.copy(), list(properties), all_changes)
            results.append(
                {
                    key: np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else copy.copy(value)
                    for key, value in self.calculator.results.items()
                }
            )
        return results


class FunctionBackend(Backend):
    """Evaluate structures with a user function of the backend signature.

    ``function(structures, properties)`` returns one result dictionary per
    structure, as :meth:`Backend.evaluate` does. This connects any model with
    its own batched evaluation without subclassing :class:`Backend`.
    """

    def __init__(self, function, max_atoms=None):
        self.function = function
        self.max_atoms = max_atoms

    def evaluate(self, structures, properties):
        return self.function(structures, properties)
