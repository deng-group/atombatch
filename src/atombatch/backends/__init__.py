"""Evaluation backends: map a list of structures to a list of results."""

from .base import SOURCE_KEY, Backend, FunctionBackend, SequentialBackend
from .symmetrix import SymmetrixBackend, supports_graph_batching
from .torchsim import TorchSimBackend, _is_torch_sim_model

__all__ = [
    "SOURCE_KEY",
    "Backend",
    "FunctionBackend",
    "SequentialBackend",
    "SymmetrixBackend",
    "TorchSimBackend",
    "as_backend",
]


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
