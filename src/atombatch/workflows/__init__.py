"""Task factories for common workflows.

Each function returns a task for :meth:`atombatch.Scheduler.submit`: a callable
that receives a :class:`atombatch.TaskContext` and runs ordinary ASE code with
batched calculators. Structures are copied, so the caller's objects are
unchanged.
"""

from .airss import airss, hot_airss
from .eos import equation_of_state
from .md import molecular_dynamics
from .monte_carlo import swap_monte_carlo
from .neb import neb
from .phonons import displacement_forces
from .relax import relax, single_point

__all__ = [
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
