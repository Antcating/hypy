import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def linear_interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


@njit(cache=True)
def lagrange_interp(x, x_vals, y_vals):
    """Three-point Lagrange interpolation"""
    xA, xB, xC = x_vals[0], x_vals[1], x_vals[2]
    yA, yB, yC = y_vals[0], y_vals[1], y_vals[2]
    denomA = (xA - xB) * (xA - xC)
    denomB = (xB - xA) * (xB - xC)
    denomC = (xC - xA) * (xC - xB)
    termA = yA * (x - xB) * (x - xC) / denomA
    termB = yB * (x - xA) * (x - xC) / denomB
    termC = yC * (x - xA) * (x - xB) / denomC
    return termA + termB + termC


@njit(cache=True)
def adaptive_interp_1d(x, x_grid, y_row):
    """
    Interpolate y(x) from grid data using 3-point Lagrange or linear fallback.
    """
    n = len(x_grid)

    if x <= x_grid[0]:
        return y_row[0]
    if x >= x_grid[-1]:
        return linear_interp(x, x_grid[-2], x_grid[-1], y_row[-2], y_row[-1])

    j1 = np.searchsorted(x_grid, x)
    j0 = j1 - 1

    if j0 < 0:
        j0, j1 = 0, 0
    if j1 >= n:
        j0, j1 = n - 1, n - 1
    if x_grid[j0] == x or j0 == j1:
        return y_row[j0]

    if j0 >= 1 and j1 < n - 1:
        if (x - x_grid[j0]) < (x_grid[j1] - x):
            idxs = np.array([j0 - 1, j0, j1])
        else:
            idxs = np.array([j0, j1, j1 + 1])
        x_vals = np.array([x_grid[i] for i in idxs])
        y_vals = np.array([y_row[i] for i in idxs])
        return lagrange_interp(x, x_vals, y_vals)

    # Fallback to linear interpolation
    return linear_interp(x, x_grid[j0], x_grid[j1], y_row[j0], y_row[j1])


@njit(cache=True)
def interp_one_depth(
    distance: float, distance_grid: np.ndarray, depth_slice: np.ndarray
) -> float:
    return adaptive_interp_1d(distance, distance_grid, depth_slice)


@njit(cache=True)
def interp_quant(
    depth: float,
    distance: float,
    depth_grid: np.ndarray,
    distance_grid: np.ndarray,
    table: np.ndarray,  # shape: (n_depths, n_distances)
) -> float:
    """
    Numba-compatible generic interpolation over depth and distance.
    """

    if depth > depth_grid[-1]:
        raise ValueError(
            f"Depth {depth} exceeds max depth {depth_grid[-1]} of travel-time table."
        )
    if distance > distance_grid[-1]:
        raise ValueError(
            f"Distance {distance} outside range of travel-time table: [{distance_grid[0]}, {distance_grid[-1]}]"
        )

    # Clamp depth to minimum grid value
    # TODO: consider raising an error instead
    depth = max(depth, depth_grid[0])
    distance = max(distance, distance_grid[0])

    # Locate depth bounds
    i1 = np.searchsorted(depth_grid, depth)
    i0 = max(i1 - 1, 0)
    i1 = min(i1, len(depth_grid) - 1)

    if i0 == i1:
        return interp_one_depth(distance, distance_grid, table[i0])

    val0 = interp_one_depth(distance, distance_grid, table[i0])
    val1 = interp_one_depth(distance, distance_grid, table[i1])
    d0 = depth_grid[i0]
    d1 = depth_grid[i1]

    if d1 == d0:
        return val0
    return linear_interp(depth, d0, d1, val0, val1)
