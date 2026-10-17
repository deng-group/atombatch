"""Batch energy and force evaluations from many independent ASE tasks."""

from .backends import (
    Backend,
    FunctionBackend,
    SequentialBackend,
    SymmetrixBackend,
    TorchSimBackend,
)
from .scheduler import BatchCalculator, Scheduler, TaskContext, TaskResult
from .workflows import (
    airss,
    displacement_forces,
    equation_of_state,
    hot_airss,
    molecular_dynamics,
    neb,
    relax,
    single_point,
    swap_monte_carlo,
)

__version__ = "0.1.3"

__all__ = [
    "Backend",
    "BatchCalculator",
    "FunctionBackend",
    "Scheduler",
    "SequentialBackend",
    "SymmetrixBackend",
    "TaskContext",
    "TaskResult",
    "TorchSimBackend",
    "airss",
    "displacement_forces",
    "equation_of_state",
    "hot_airss",
    "molecular_dynamics",
    "neb",
    "relax",
    "single_point",
    "swap_monte_carlo",
]
