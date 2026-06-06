"""Main configuration and validation logic for the event locator.

This module defines the LocatorConfig dataclass, which holds all the immutable
settings for the event locator. It also includes validation functions to ensure
that the provided configuration parameters are valid and compatible with each other.

Returns
-------
tuple
    A tuple containing:
    - mode: Mode of operation (ModeLiteral).
    - engine: Ray tracing engine (EngineLiteral).
    - derivative_method: Method for computing derivatives (DerivativeMethodLiteral).

Raises
------
ValueError
    If the provided mode is not supported.
ValueError
    If the provided engine is not supported for the specified mode.
ValueError
    If the provided derivative_method is not supported for the mode and engine.
FileNotFoundError
    If a required file is not found.
ValueError
    If a provided file has an unsupported format.
ValueError
    If an unsupported solver is specified.
ValueError
    If distance weighting parameters are invalid.
ValueError
    If precomputation arguments are invalid.
ValueError
    If weight_start or weight_end are not in (0, 1).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import DerivativeMethodLiteral, SolverLiteral

VALID_SOLVERS = {"lsqr", "svd"}


def validate_solver(solver: str) -> SolverLiteral:
    """Validate that the provided solver is supported, and return it.

    There are specific valid solvers, defined in VALID_SOLVERS:

    * ``lsqr`` uses ``scipy.sparse.linalg.lsmr`` for large sparse systems.
    * ``svd`` uses ``scipy.linalg.svd`` and is generally less scalable.

    Parameters
    ----------
    solver : str
        solver to use for the linear system in the location inversion.

    Returns
    -------
    SolverLiteral
        The validated solver.

    Raises
    ------
    ValueError
        If the provided solver is not supported.
    """
    if solver.lower() not in VALID_SOLVERS:
        raise ValueError(
            f"Unsupported solver: {solver!r}. Use one of: {', '.join(VALID_SOLVERS)}."
        )
    return solver.lower()  # type: ignore[return-value]


def configure_distance_weighting(
    distance_start_km: Optional[float],
    distance_end_km: Optional[float],
    weight_start: Optional[float],
    weight_end: Optional[float],
) -> "tuple[bool, Optional[float], Optional[float]]":
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


@dataclass(frozen=True)
class LocatorConfig:
    """
    Immutable configuration for the event locator.

    Holds all settings that do not change during a locate() run.
    Created once in EventLocator.__init__ and shared (read-only) with all components.
    """

    phases: tuple
    min_picks_per_event: int
    default_time_error: float
    derivative_method: DerivativeMethodLiteral
    fix_depth_km: Optional[float]
    precalculate_velocities: bool
    distance_weighting_enabled: bool
    k: Optional[float]
    x0: Optional[float]
    full_data_covariance: bool = False
    correlation_length_km: float = 1.0
    error_coefficient: float = 1.0
    derivative_delta_km: float = 0.5
    dim: int = 4
    n_phases: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived values and validate post-init constraints."""
        if self.correlation_length_km <= 0.0:
            raise ValueError("correlation_length_km must be > 0.")
        if self.error_coefficient < 0.0:
            raise ValueError("error_coefficient must be >= 0.")
        object.__setattr__(self, "n_phases", len(self.phases))
