"""Weighting helpers for travel-time and arrival-time inversion."""

from typing import Optional

import numpy as np


def configure_distance_weighting(
    distance_start_km: Optional[float],
    distance_end_km: Optional[float],
    weight_start: Optional[float],
    weight_end: Optional[float],
) -> tuple[bool, Optional[float], Optional[float]]:
    """Configure distance weighting parameters for the location inversion.

    The distance weighting function is defined as:
    w(d) = 1 / (1 + 10^(k (d - x0)))

    where d is the distance, k controls the steepness of the weighting curve,
    and x0 is the distance at which the weight is 0.5.

    Parameters
    ----------
    distance_start_km : Optional[float]
        Distance (in km) at which the weighting starts to decrease from weight_start.
    distance_end_km : Optional[float]
        Distance (in km) at which the weighting reaches weight_end.
    weight_start : Optional[float]
        Weight at distance_start_km. Must be in (0, 1).
    weight_end : Optional[float]
        Weight at distance_end_km. Must be in (0, 1).

    Returns
    -------
    tuple[bool, Optional[float], Optional[float]]
        Returns ``(enabled, k, x0)`` where ``enabled`` is ``True`` only when all
        weighting parameters are provided. If weighting is disabled,
        ``k`` and ``x0`` are ``None``.

    Raises
    ------
    ValueError
        If weight_start or weight_end are not in (0, 1).
    ValueError
        If distance_end_km is not greater than distance_start_km.
    """
    args = (distance_start_km, distance_end_km, weight_start, weight_end)
    if any(a is None for a in args):
        return False, None, None

    d0 = float(distance_start_km)  # type: ignore[arg-type]
    d1 = float(distance_end_km)  # type: ignore[arg-type]
    w0 = float(weight_start)  # type: ignore[arg-type]
    w1 = float(weight_end)  # type: ignore[arg-type]

    if not (0.0 < w0 < 1.0 and 0.0 < w1 < 1.0):
        raise ValueError("weight_start and weight_end must be in (0, 1).")
    if d1 <= d0:
        raise ValueError(
            "distance_weight_end must be greater than distance_weight_start."
        )

    # Solve logistic parameters for: w(d) = 1 / (1 + 10^(k (d - x0)))
    # https://www.desmos.com/calculator/221eudc8kc
    k = (np.log10((1.0 / w1) - 1.0) - np.log10((1.0 / w0) - 1.0)) / (d1 - d0)
    x0 = d0 - (np.log10((1.0 / w0) - 1.0) / k)
    return True, float(k), float(x0)


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
        Square root of the logistic weight(s) in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``k`` or ``x0`` is missing from ``kwargs``.

    Notes
    -----
    The curve is ``w(d) = 1 / (1 + 10^(k (d - x0)))``, matching
    :func:`configure_distance_weighting`. The returned value is ``sqrt(w)``.
    """
    k = kwargs.get("k", None)
    x0 = kwargs.get("x0", None)

    if k is None or x0 is None:
        raise ValueError("Parameters 'k' and 'x0' must be provided in kwargs.")

    # https://www.desmos.com/calculator/221eudc8kc
    # Equivalent to 1 / (1 + 10^(k(d-x0))); use exp so large exponents
    # saturate to inf instead of raising OverflowError.
    weight = 1.0 / (1.0 + np.exp(k * (distance - x0) * np.log(10.0)))

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
    time_error = np.maximum(
        time_error, 1e-6
    )  # Avoid division by zero or extremely large weights

    return 1 / time_error
