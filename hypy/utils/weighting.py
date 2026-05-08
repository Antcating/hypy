"""Weighting helpers for travel-time and arrival-time inversion."""

import numpy as np


def distance_weight(distance, **kwargs):
    """Compute a logistic distance weight.

    Parameters
    ----------
    distance : float or array-like
        Distance in kilometers.
    **kwargs : dict
        Required keyword arguments.
    k : float
        Logistic slope.
    x0 : float
        Midpoint of the logistic curve in kilometers.

    Returns
    -------
    float or numpy.ndarray
        Distance weight(s) in the interval ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``k`` or ``x0`` is missing from ``kwargs``.

    Notes
    -----
    The curve is intentionally steep enough to emphasize nearby stations while
    still keeping far stations in the inversion with reduced influence.
    """
    k = kwargs.get("k", None)
    x0 = kwargs.get("x0", None)

    if k is None or x0 is None:
        raise ValueError("Parameters 'k' and 'x0' must be provided in kwargs.")

    # https://www.desmos.com/calculator/221eudc8kc
    weight = 1.0 / (1.0 + np.exp(k * (distance - x0)))

    return np.sqrt(weight)


def time_error_weight(time_error: float | np.ndarray, **kwargs) -> float | np.ndarray:
    """Compute an inverse-square weight from pick-time error.

    Parameters
    ----------
    time_error : float or array-like
        Time error in seconds.
    **kwargs : dict
        Accepted for API compatibility and ignored.

    Returns
    -------
    float or numpy.ndarray
        Weight(s) proportional to ``1 / (time_error)^2``.
    -----
    """
    time_error = max(
        time_error, 1e-6
    )  # Avoid division by zero or extremely large weights

    return 1 / time_error
