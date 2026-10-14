"""Edge geometry of a reused Symmetrix neighbor cache.

For a cached neighbor list, every step recomputes the edge vectors
``xyz = (reference_xyz + d[source]) - d[receiver]`` from the atomic
displacements ``d`` since the cache was built, their lengths, and which edges
lie inside the cutoff. NumPy does this in several passes over temporary
arrays. With Numba installed, one compiled loop does it instead.

Both versions perform the same floating-point operations in the same order as
the Symmetrix calculator (``np.linalg.norm`` sums ``(x*x + y*y) + z*z``), so
their results are identical to it bit for bit. Numba does not contract
multiplications and additions into fused multiply-adds unless ``fastmath`` is
enabled, which it is not here.
"""

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover - exercised when numba is absent
    numba = None


def edge_geometry(reference_xyz, sources, receivers, displacements, cutoff, clamp):
    """Return ``(receivers, sources, distances, xyz)`` of the current step.

    With ``clamp``, edges at or beyond ``cutoff`` are kept with their vector
    scaled to the cutoff length, as Symmetrix does for prepared execution
    modes; otherwise they are dropped.
    """
    if numba is not None:
        return _edge_geometry_numba(
            np.ascontiguousarray(reference_xyz, dtype=np.float64),
            np.ascontiguousarray(sources),
            np.ascontiguousarray(receivers),
            np.ascontiguousarray(displacements, dtype=np.float64),
            float(cutoff),
            bool(clamp),
        )
    return edge_geometry_numpy(
        reference_xyz, sources, receivers, displacements, cutoff, clamp
    )


def edge_geometry_numpy(
    reference_xyz, sources, receivers, displacements, cutoff, clamp
):
    """NumPy version, written as in ``Symmetrix._cached_neighbor_geometry``."""
    xyz = reference_xyz + displacements[sources] - displacements[receivers]
    distances = np.linalg.norm(xyz, axis=1)
    if clamp:
        inactive = distances >= cutoff
        if np.any(inactive):
            xyz = np.array(xyz, copy=True)
            distances = np.array(distances, copy=True)
            xyz[inactive] *= (cutoff / distances[inactive])[:, None]
            distances[inactive] = cutoff
        return (
            np.ascontiguousarray(receivers),
            np.ascontiguousarray(sources),
            np.ascontiguousarray(distances),
            np.ascontiguousarray(xyz),
        )
    active = distances < cutoff
    return (
        np.ascontiguousarray(receivers[active]),
        np.ascontiguousarray(sources[active]),
        np.ascontiguousarray(distances[active]),
        np.ascontiguousarray(xyz[active]),
    )


if numba is not None:

    @numba.njit(cache=True)
    def _fill(reference_xyz, sources, receivers, displacements, cutoff, clamp,
              out_receivers, out_sources, out_distances, out_xyz):  # fmt: skip
        count = 0
        for edge in range(sources.shape[0]):
            source = sources[edge]
            receiver = receivers[edge]
            x = (reference_xyz[edge, 0] + displacements[source, 0]) - displacements[
                receiver, 0
            ]
            y = (reference_xyz[edge, 1] + displacements[source, 1]) - displacements[
                receiver, 1
            ]
            z = (reference_xyz[edge, 2] + displacements[source, 2]) - displacements[
                receiver, 2
            ]
            xx = x * x
            yy = y * y
            zz = z * z
            distance = np.sqrt((xx + yy) + zz)
            if distance < cutoff:
                pass
            elif clamp:
                scale = cutoff / distance
                x = x * scale
                y = y * scale
                z = z * scale
                distance = cutoff
            else:
                continue
            out_receivers[count] = receiver
            out_sources[count] = source
            out_distances[count] = distance
            out_xyz[count, 0] = x
            out_xyz[count, 1] = y
            out_xyz[count, 2] = z
            count += 1
        return count

    def _edge_geometry_numba(
        reference_xyz, sources, receivers, displacements, cutoff, clamp
    ):
        size = sources.shape[0]
        out_receivers = np.empty(size, dtype=receivers.dtype)
        out_sources = np.empty(size, dtype=sources.dtype)
        out_distances = np.empty(size, dtype=np.float64)
        out_xyz = np.empty((size, 3), dtype=np.float64)
        count = _fill(
            reference_xyz, sources, receivers, displacements, cutoff, clamp,
            out_receivers, out_sources, out_distances, out_xyz,
        )  # fmt: skip
        return (
            out_receivers[:count],
            out_sources[:count],
            out_distances[:count],
            out_xyz[:count],
        )
