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
    # No distance bounds means weighting is disabled; weight_start/weight_end
    # have non-None API defaults and therefore must not activate it by themselves.
    if distance_start_km is None and distance_end_km is None:
        return False, None, None
    if (
        distance_start_km is None
        or distance_end_km is None
        or weight_start is None
        or weight_end is None
    ):
        raise ValueError(
            "Distance weighting requires both distance bounds and both endpoint "
            "weights."
        )

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
    # Endpoint weights too close to tell apart in floating point give a flat
    # curve, whose midpoint is undefined and would poison every weight.
    if k == 0.0:
        raise ValueError(
            "weight_start and weight_end are too close to define a distance "
            "weighting curve."
        )
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


def weighted_median(values, weights):
    """
    Weighted median.

    Parameters
    ----------
    values : array-like
    weights : array-like
        Statistical weights, not least-squares row scales.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)

    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)

    return float(values[np.searchsorted(cumulative, cutoff)])


def residual_weight_factor(
    residuals: np.ndarray,
    base_weights: np.ndarray,
    *,
    residual_rms_floor_s: float = 0.1,
    residual_full_weight_factor: float = 1.0,
    residual_zero_weight_factor: float = 3.0,
) -> np.ndarray:
    """
    Robust MAD-based cosine residual weighting.

    Parameters
    ----------
    residuals
        Arrival-time residuals [s].

    base_weights
        Least-squares row scales before residual weighting,
        e.g. distance_W * time_error_W.

    residual_rms_floor_s
        Minimum robust scale [s]. The existing parameter name
        is kept for API compatibility, although the scale is
        now MAD-based.

    residual_full_weight_factor
        Residual deviation <= factor * scale gets full weight.

    residual_zero_weight_factor
        Residual deviation >= factor * scale gets zero weight.

    Returns
    -------
    numpy.ndarray
        Least-squares row multipliers sqrt(weight), in [0, 1].
    """
    residuals = np.asarray(residuals, dtype=float)
    base_weights = np.asarray(base_weights, dtype=float)

    if residuals.shape != base_weights.shape:
        raise ValueError("residuals and base_weights must have the same shape.")

    if residuals.size == 0:
        return np.ones_like(residuals)

    valid = np.isfinite(residuals) & np.isfinite(base_weights) & (base_weights > 0.0)

    if not np.any(valid):
        return np.ones_like(residuals)

    r = residuals[valid]

    # base_weights are LS row multipliers.
    # Their corresponding statistical weights are squared.
    statistical_weights = base_weights[valid] ** 2

    # ---------------------------------------------------------
    # 1. Robust residual center
    # ---------------------------------------------------------

    center = weighted_median(
        r,
        statistical_weights,
    )

    # ---------------------------------------------------------
    # 2. Weighted MAD
    # ---------------------------------------------------------

    absolute_deviation = np.abs(r - center)

    mad = weighted_median(
        absolute_deviation,
        statistical_weights,
    )

    # Gaussian consistency:
    # sigma ~= 1.4826 * MAD
    robust_sigma = 1.4826 * mad

    if not np.isfinite(robust_sigma):
        robust_sigma = 0.0

    scale = max(
        robust_sigma,
        float(residual_rms_floor_s),
    )

    # ---------------------------------------------------------
    # 3. Define rejection thresholds
    # ---------------------------------------------------------

    full_weight_cutoff = float(residual_full_weight_factor) * scale

    zero_weight_cutoff = float(residual_zero_weight_factor) * scale

    if zero_weight_cutoff <= full_weight_cutoff:
        raise ValueError(
            "residual_zero_weight_factor must be greater than "
            "residual_full_weight_factor."
        )

    # Important:
    # determine outliers relative to robust residual center.
    deviation = np.abs(residuals - center)

    # ---------------------------------------------------------
    # 4. Cosine taper
    # ---------------------------------------------------------

    weight = np.ones_like(
        residuals,
        dtype=float,
    )

    # Invalid residuals get zero weight
    weight[~np.isfinite(residuals)] = 0.0

    zero_mask = deviation >= zero_weight_cutoff

    taper_mask = (
        (deviation > full_weight_cutoff)
        & (deviation < zero_weight_cutoff)
        & np.isfinite(deviation)
    )

    weight[zero_mask] = 0.0

    taper_fraction = (deviation[taper_mask] - full_weight_cutoff) / (
        zero_weight_cutoff - full_weight_cutoff
    )

    weight[taper_mask] = 0.5 * (1.0 + np.cos(np.pi * taper_fraction))

    # LS row scale.
    # Squaring this in least-squares gives the intended weight.
    return np.sqrt(weight)
