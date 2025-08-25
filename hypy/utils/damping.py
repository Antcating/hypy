from typing import Optional

import numpy as np


def update_damping_factor(
    current_factor: float,
    iteration: Optional[int] = None,
    dx: Optional[np.ndarray] = None,
) -> float:
    """
    Simplest damping function that reduces the damping factor by 10% each iteration.
    This function can be replaced with a more sophisticated damping strategy if needed.
    Args:
        current_factor (float): Current damping factor.
        iteration (int): Current iteration number.
        dx (np.ndarray): Update in (lat, lon, depth, time) [degree, degree, km, second] format.
    Returns:
        float: Updated damping factor.
    """
    return 0.9 * current_factor
