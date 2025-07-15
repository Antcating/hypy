import numpy as np
from numba import njit

@njit
def linear_interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)

@njit
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

@njit
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
