"""Evaluation backends: map a list of structures to a list of results."""

import copy
import sys

import numpy as np
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .symmetrix import GraphBatcher, supports_graph_batching


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
    """Evaluate structures with a Symmetrix calculator in one native call.

    atombatch joins the structures into one disconnected neighbor graph and
    calls the Symmetrix evaluator directly (:mod:`atombatch.symmetrix`), with
    one Symmetrix neighbor cache per trajectory so that neighbor lists are
    rebuilt only when atoms move beyond the neighbor skin. This works from
    Symmetrix-XL 0.1.1; ``Symmetrix.calculate_batch`` is used only for
    calculators without the evaluator interface. Models with dispersion
    corrections or electric-field coupling fall back to one ASE evaluation
    per structure.
    """

    def __init__(self, calculator, max_atoms=None, batch_mode="native"):
        self.calculator = calculator
        self.max_atoms = max_atoms
        self.batch_mode = batch_mode
        self._graph = None
        if supports_graph_batching(calculator):
            self._graph = GraphBatcher(calculator)
        elif not callable(getattr(calculator, "calculate_batch", None)):
            raise TypeError(
                "SymmetrixBackend requires a Symmetrix calculator "
                "(Symmetrix-XL 0.1.1 or later)."
            )

    def evaluate(self, structures, properties):
        if self._graph is not None:
            return self._graph.evaluate(structures, properties)
        return self.calculator.calculate_batch(
            structures, properties=tuple(properties), batch_mode=self.batch_mode
        )


class TorchSimBackend(Backend):
    """Evaluate structures in one call of a TorchSim ``ModelInterface``.

    TorchSim provides batched interfaces for MACE, fairchem, SevenNet, ORB,
    MatterSim, NequIP, Nequix, metatomic, and classical potentials. The
    structures are joined into one ``SimState`` on the model's device and
    dtype, and the outputs are split back per structure. Stress requires a
    model constructed with ``compute_stress=True``.
    """

    def __init__(self, model, max_atoms=None):
        self.model = model
        self.max_atoms = max_atoms

    def evaluate(self, structures, properties):
        import torch_sim

        if "stress" in properties and not self.model.compute_stress:
            raise ValueError(
                "Stress was requested from a TorchSim model constructed "
                "without compute_stress=True."
            )
        state = torch_sim.io.atoms_to_state(
            list(structures), device=self.model.device, dtype=self.model.dtype
        )
        output = self.model(state)
        energies = _numpy(output["energy"]).reshape(-1)
        forces = _numpy(output["forces"])
        stresses = _numpy(output["stress"]) if "stress" in properties else None
        results, start = [], 0
        for index, atoms in enumerate(structures):
            stop = start + len(atoms)
            result = {
                "energy": float(energies[index]),
                "forces": forces[start:stop].copy(),
            }
            if stresses is not None:
                result["stress"] = full_3x3_to_voigt_6_stress(stresses[index])
            results.append(result)
            start = stop
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


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _is_torch_sim_model(model):
    # Checked only when TorchSim is already imported, so that ASE-only runs
    # do not import torch.
    interface = sys.modules.get("torch_sim.models.interface")
    return interface is not None and isinstance(model, interface.ModelInterface)


def as_backend(model, max_atoms=None):
    """Return ``model`` as a backend.

    Backends are used as given. Symmetrix calculators and TorchSim models
    are evaluated in batched calls. Other ASE calculators
    are evaluated one structure at a time, and plain functions are wrapped in
    :class:`FunctionBackend`.
    """
    if isinstance(model, Backend):
        if max_atoms is not None:
            model.max_atoms = max_atoms
        return model
    if callable(getattr(model, "calculate_batch", None)) or supports_graph_batching(
        model
    ):
        return SymmetrixBackend(model, max_atoms=max_atoms)
    if _is_torch_sim_model(model):
        return TorchSimBackend(model, max_atoms=max_atoms)
    if callable(getattr(model, "calculate", None)):
        return SequentialBackend(model, max_atoms=max_atoms)
    if callable(model):
        return FunctionBackend(model, max_atoms=max_atoms)
    raise TypeError(f"Cannot use {type(model).__name__} as a model backend.")
