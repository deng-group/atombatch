"""Evaluation backends: map a list of structures to a list of results."""

import copy

import numpy as np
from ase.calculators.calculator import all_changes


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


class SymmetrixBackend(Backend):
    """Evaluate structures in one native call with ``Symmetrix.calculate_batch``."""

    def __init__(self, calculator, max_atoms=None, batch_mode="native"):
        if not callable(getattr(calculator, "calculate_batch", None)):
            raise TypeError(
                "SymmetrixBackend requires a Symmetrix calculator with "
                "calculate_batch (batched evaluation)."
            )
        self.calculator = calculator
        self.max_atoms = max_atoms
        self.batch_mode = batch_mode

    def evaluate(self, structures, properties):
        return self.calculator.calculate_batch(
            structures, properties=tuple(properties), batch_mode=self.batch_mode
        )


def as_backend(model, max_atoms=None):
    """Return ``model`` as a backend, wrapping ASE calculators."""
    if isinstance(model, Backend):
        if max_atoms is not None:
            model.max_atoms = max_atoms
        return model
    if callable(getattr(model, "calculate_batch", None)):
        return SymmetrixBackend(model, max_atoms=max_atoms)
    if callable(getattr(model, "calculate", None)):
        return SequentialBackend(model, max_atoms=max_atoms)
    raise TypeError(f"Cannot use {type(model).__name__} as a model backend.")
