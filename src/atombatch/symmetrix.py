"""Kept for compatibility; see :mod:`atombatch.backends.symmetrix`."""

from .backends.symmetrix import (  # noqa: F401
    SOURCE_KEY,
    GraphBatcher,
    edge_forces,
    segment_virials,
    supports_graph_batching,
    uses_ase_neighbor_list,
)
