"""Interpolation helpers for travel-time tables."""

import numpy as np
from numba import (
    njit,  # type: ignore[import-untyped]
    prange,
)


@njit(cache=True)
def velocity_from_layers(
    depth: float,
    top_depth: np.ndarray,
    bot_depth: np.ndarray,
    top_velocity: np.ndarray,
    bot_velocity: np.ndarray,
) -> float:
    """Interpolate velocity from layered model parameters."""
    n = len(top_depth)
    layer_index = -1
    for i in range(n):
        if top_depth[i] <= depth and depth <= bot_depth[i]:
            layer_index = i
            break
    if layer_index == -1:
        raise ValueError("Depth is not covered by any model layer.")

    top_z = top_depth[layer_index]
    bot_z = bot_depth[layer_index]
    top_v = top_velocity[layer_index]
    bot_v = bot_velocity[layer_index]
    if bot_z == top_z:
        return top_v
    return top_v + (bot_v - top_v) * ((depth - top_z) / (bot_z - top_z))


@njit(cache=True, inline="always")
def linear_interp(x, x0, x1, y0, y1):
    """Perform linear interpolation between two samples.

    Parameters
    ----------
    x : float
        Target coordinate.
    x0 : float
        Lower sample coordinate.
    x1 : float
        Upper sample coordinate.
    y0 : float
        Sample value at ``x0``.
    y1 : float
        Sample value at ``x1``.

    Returns
    -------
    float
        Interpolated value at ``x``.

    Notes
    -----
    If ``x0 == x1``, the function returns ``y0`` to avoid division by zero.
    """
    if x1 == x0:
        return y0
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


@njit(cache=True, inline="always")
def lagrange_interp_scalars(
    x: float,
    xA: float,
    xB: float,
    xC: float,
    yA: float,
    yB: float,
    yC: float,
) -> float:
    """Evaluate a three-point Lagrange interpolant from scalar coordinates."""
    denomA = (xA - xB) * (xA - xC)
    denomB = (xB - xA) * (xB - xC)
    denomC = (xC - xA) * (xC - xB)
    return (
        yA * (x - xB) * (x - xC) / denomA
        + yB * (x - xA) * (x - xC) / denomB
        + yC * (x - xA) * (x - xB) / denomC
    )


@njit(cache=True, inline="always")
def _grid_upper_index(x: float, x_grid: np.ndarray) -> int:
    """Return ``np.searchsorted(x_grid, x)`` with an O(1) uniform-grid fast path."""
    n = len(x_grid)
    if n >= 2:
        step = (x_grid[n - 1] - x_grid[0]) / (n - 1)
        if step > 0.0:
            idx = int((x - x_grid[0]) / step) + 1
            if idx >= n:
                idx = n - 1
            if idx >= 1 and x_grid[idx - 1] < x <= x_grid[idx]:
                return idx
    return int(np.searchsorted(x_grid, x))


@njit(cache=True, inline="always")
def adaptive_interp_1d(x, x_grid, y_row):
    """Interpolate one-dimensional tabulated data with adaptive stencils.

    Parameters
    ----------
    x : float
        Target coordinate.
    x_grid : numpy.ndarray
        Monotonic grid coordinates for the tabulated samples.
    y_row : numpy.ndarray
        Tabulated values for a single depth slice.

    Returns
    -------
    float
        Interpolated value at ``x``.

    Notes
    -----
    The function prefers three-point Lagrange interpolation when the target is
    safely away from the grid boundaries and falls back to linear
    interpolation near the edges.
    """
    n = len(x_grid)

    if x <= x_grid[0]:
        return y_row[0]
    if x >= x_grid[-1]:
        return linear_interp(x, x_grid[-2], x_grid[-1], y_row[-2], y_row[-1])

    j1 = _grid_upper_index(x, x_grid)
    j0 = j1 - 1

    if j0 < 0:
        j0, j1 = 0, 0
    if j1 >= n:
        j0, j1 = n - 1, n - 1
    if x_grid[j0] == x or j0 == j1:
        return y_row[j0]

    if j0 >= 1 and j1 < n - 1:
        if (x - x_grid[j0]) < (x_grid[j1] - x):
            i0, i1, i2 = j0 - 1, j0, j1
        else:
            i0, i1, i2 = j0, j1, j1 + 1
        return lagrange_interp_scalars(
            x,
            x_grid[i0],
            x_grid[i1],
            x_grid[i2],
            y_row[i0],
            y_row[i1],
            y_row[i2],
        )

    # Fallback to linear interpolation
    return linear_interp(x, x_grid[j0], x_grid[j1], y_row[j0], y_row[j1])


@njit(cache=True, inline="always")
def interp_one_depth(
    distance: float, distance_grid: np.ndarray, depth_slice: np.ndarray
) -> float:
    """Interpolate a single depth slice across distance.

    Parameters
    ----------
    distance : float
        Target epicentral distance.
    distance_grid : numpy.ndarray
        Monotonic distance grid for the table.
    depth_slice : numpy.ndarray
        Travel-time values for one depth level.

    Returns
    -------
    float
        Interpolated travel time at the requested distance.
    """
    return adaptive_interp_1d(distance, distance_grid, depth_slice)


@njit(cache=True)
def interp_quant(
    depth: float,
    distance: float,
    depth_grid: np.ndarray,
    distance_grid: np.ndarray,
    table: np.ndarray,  # shape: (n_depths, n_distances)
) -> float:
    """Interpolate a quantized travel-time table in depth and distance.

    Parameters
    ----------
    depth : float
        Target depth.
    distance : float
        Target epicentral distance.
    depth_grid : numpy.ndarray
        Monotonic depth grid for the table.
    distance_grid : numpy.ndarray
        Monotonic distance grid for the table.
    table : numpy.ndarray
        Travel-time table with shape ``(n_depths, n_distances)``.

    Returns
    -------
    float
        Interpolated travel time.

    Raises
    ------
    ValueError
        If ``depth`` or ``distance`` exceeds the tabulated range.

    Notes
    -----
    Values below the minimum depth and distance are clamped to the first grid
    point so callers can safely query shallow or near-source samples without
    triggering a boundary failure.
    """
    if depth > depth_grid[-1]:
        raise ValueError(
            f"Depth {depth} exceeds max depth {depth_grid[-1]} of travel-time table."
        )
    if distance > distance_grid[-1]:
        raise ValueError(
            "Distance "
            f"{distance} outside range of travel-time table: "
            f"[{distance_grid[0]}, {distance_grid[-1]}]"
        )

    if depth < 0:
        raise ValueError(f"Depth {depth} is negative. Depth must be non-negative.")
    # Clamp depth to minimum grid value
    # TODO: consider raising an error instead
    depth = max(depth, depth_grid[0])
    distance = max(distance, distance_grid[0])

    # Locate depth bounds
    i1 = _grid_upper_index(depth, depth_grid)
    i0 = max(i1 - 1, 0)
    if i1 > len(depth_grid) - 1:
        i1 = len(depth_grid) - 1

    if i0 == i1:
        return interp_one_depth(distance, distance_grid, table[i0])

    val0 = interp_one_depth(distance, distance_grid, table[i0])
    val1 = interp_one_depth(distance, distance_grid, table[i1])
    d0 = depth_grid[i0]
    d1 = depth_grid[i1]

    if d1 == d0:
        return val0
    return linear_interp(depth, d0, d1, val0, val1)


@njit(cache=True, parallel=True)
def interp_quant_batch(
    depths: np.ndarray,
    distances: np.ndarray,
    depth_grid: np.ndarray,
    distance_grid: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """Interpolate many depth-distance pairs on one travel-time table."""
    out = np.empty(depths.size, dtype=np.float64)
    for i in prange(depths.size):
        out[i] = interp_quant(depths[i], distances[i], depth_grid, distance_grid, table)
    return out
