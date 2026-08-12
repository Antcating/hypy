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


def configure_residual_weighting(
    start_iteration: Optional[int],
    residual_rms_floor_s: float = 0.1,
    residual_full_weight_factor: float = 1.0,
    residual_zero_weight_factor: float = 3.0,
) -> tuple[bool, Optional[int], float, float, float]:
    """Configure HypoInverse-style residual weighting for a locate run.

    Parameters
    ----------
    start_iteration : Optional[int]
        1-based outer inversion iteration at which residual weighting begins.
        ``None`` disables residual weighting.
    residual_rms_floor_s : float, default=0.1
        Floor (seconds) applied to the current weighted RMS before building
        residual-weight thresholds (HypoInverse ``RMSCUT``).
    residual_full_weight_factor : float, default=1.0
        Full-weight residual threshold as a multiple of the floored RMS
        (HypoInverse ``RMSW1``).
    residual_zero_weight_factor : float, default=3.0
        Zero-weight residual threshold as a multiple of the floored RMS
        (HypoInverse ``RMSW2``). Must be greater than
        ``residual_full_weight_factor``.

    Returns
    -------
    tuple[bool, Optional[int], float, float, float]
        ``(enabled, start_iteration, residual_rms_floor_s,
        residual_full_weight_factor, residual_zero_weight_factor)``.

    Raises
    ------
    ValueError
        If residual weighting is requested with invalid parameters.
    """
    if start_iteration is None:
        return (
            False,
            None,
            float(residual_rms_floor_s),
            float(residual_full_weight_factor),
            float(residual_zero_weight_factor),
        )

    start = int(start_iteration)
    if start < 1:
        raise ValueError("residual_start_iteration must be >= 1 when enabled.")
    if float(residual_rms_floor_s) <= 0.0:
        raise ValueError("residual_rms_floor_s must be > 0.")
    if float(residual_full_weight_factor) < 0.0:
        raise ValueError("residual_full_weight_factor must be >= 0.")
    if float(residual_zero_weight_factor) <= float(residual_full_weight_factor):
        raise ValueError(
            "residual_zero_weight_factor must be greater than "
            "residual_full_weight_factor."
        )

    return (
        True,
        start,
        float(residual_rms_floor_s),
        float(residual_full_weight_factor),
        float(residual_zero_weight_factor),
    )


def residual_weight_factor(
    residuals: np.ndarray,
    base_weights: np.ndarray,
    *,
    residual_rms_floor_s: float = 0.1,
    residual_full_weight_factor: float = 1.0,
    residual_zero_weight_factor: float = 3.0,
) -> np.ndarray:
    """Compute HypoInverse-style cosine residual weights.

    Let ``rms_scale = max(weighted_rms, residual_rms_floor_s)``. Then:

    - ``|residual| <= residual_full_weight_factor * rms_scale`` → weight 1
    - ``|residual| >= residual_zero_weight_factor * rms_scale`` → weight 0
    - in between → cosine taper from 1 to 0

    ``weighted_rms`` uses ``base_weights`` (pick/time and distance weights)
    before residual reweighting.

    Parameters
    ----------
    residuals : array-like
        Arrival-time residuals in seconds for the rows being weighted.
    base_weights : array-like
        Current row scales (e.g. ``distance_W * time_error_W``) used only to
        estimate the weighted RMS that sets the taper thresholds. Must match
        ``residuals`` in shape.
    residual_rms_floor_s, residual_full_weight_factor, residual_zero_weight_factor : float
        Residual-weighting parameters (see :func:`configure_residual_weighting`).

    Returns
    -------
    numpy.ndarray
        Square root of the residual-weight factor in ``[0, 1]``, matching the
        ``distance_weight`` convention so least-squares uses factor * r^2.

    Notes
    -----
    This does not permanently remove picks. Factors are recomputed each
    iteration from the current residuals; a previously zeroed pick can regain
    weight if its residual improves.
    """
    residuals = np.asarray(residuals, dtype=float)
    base_weights = np.asarray(base_weights, dtype=float)
    if residuals.shape != base_weights.shape:
        raise ValueError("residuals and base_weights must have the same shape.")
    if residuals.size == 0:
        return np.ones_like(residuals)

    weight_norm_sq = float(np.sum(base_weights * base_weights))
    if weight_norm_sq <= 0.0:
        return np.ones_like(residuals)

    weighted_rms = float(
        np.sqrt(np.sum((base_weights * residuals) ** 2) / weight_norm_sq)
    )
    rms_scale = max(weighted_rms, float(residual_rms_floor_s))

    full_weight_cutoff = float(residual_full_weight_factor) * rms_scale
    zero_weight_cutoff = float(residual_zero_weight_factor) * rms_scale
    taper_width = zero_weight_cutoff - full_weight_cutoff
    if taper_width <= 0.0:
        raise ValueError(
            "residual_zero_weight_factor must be greater than "
            "residual_full_weight_factor."
        )

    abs_residual = np.abs(residuals)
    weight = np.ones_like(residuals)

    zero_weight = abs_residual >= zero_weight_cutoff
    taper = (abs_residual > full_weight_cutoff) & ~zero_weight

    weight[zero_weight] = 0.0
    # Cosine half-cycle from 1 at the full-weight cutoff to 0 at the zero cutoff.
    taper_fraction = (abs_residual[taper] - full_weight_cutoff) / taper_width
    weight[taper] = 0.5 * (1.0 + np.cos(np.pi * taper_fraction))

    return np.sqrt(weight)
