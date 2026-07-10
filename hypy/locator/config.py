"""Handling configurations for the event locator."""

import numbers
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .types import (
    VALID_DERIVATIVE_METHODS,
    VALID_DIMENSIONS,
    VALID_SOLVERS,
    DerivativeMethodLiteral,
    SolverLiteral,
)


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
    Immutable, construction-time configuration for the event locator.

    Holds settings that define the *identity* of a locator: the data layout,
    the physical/model assumptions, and the components built from them. These
    do not change between ``locate()`` calls on the same locator. Created once
    in ``EventLocator.__init__`` and shared (read-only) with all components.

    Per-run knobs (weighting, error model, RNG seed) live in :class:`RunConfig`.
    """

    # Phases to use for travel-time calculations.
    # This can be used to limit the inversion to a subset of phases,
    # even if the input catalog contains picks for more phases.
    phases: Iterable[str]
    # Minimum number of picks required per event to attempt location.
    # Events with fewer picks will be marked as invalid
    # and skipped during inversion.
    # The lowest reasonable value is equal to the dimensionality of the problem.
    # For example, if the depth is fixed for all events,
    # the effective dimensionality is reduced by one, so the minimum number
    # of picks can also be reduced by one.
    min_picks_per_event: int
    # If a pick does not have an associated time error,
    # this default error (in seconds) will be used.
    default_time_error: float
    derivative_method: DerivativeMethodLiteral
    # Whether to fix event depths during inversion.
    # This is a common practice when the problem is severely underconstrained,
    # e.g. due to sparse station coverage or lack of depth-sensitive phases.
    # Note: this is a global setting that applies to all events. If you want
    # to fix depth for some events but not others, individual events
    # with operator-assigned depths in the input catalog will also be depth-fixed,
    # regardless of this setting.
    # Note: the solver will still compute depth updates, but they will be ignored
    # when applying the update to the event locations. This means that fixing depth
    # will not reduce the computational cost of the inversion,
    # but it can help stabilize the solution when depth is poorly constrained.
    fix_depth_km: Optional[float]
    precalculate_velocities: bool
    derivative_delta_km: float = 0.5
    # Dimensionality of the inversion problem (x, y, z, t0)
    # Note: for now, only 4D is supported,
    # but this could be made flexible in the future if needed.
    dim: int = 4
    n_phases: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived values and validate post-init constraints."""
        # Convert phases to a lowercase tuple first, then derive n_phases from the
        # materialized tuple.
        object.__setattr__(
            self,
            "phases",
            tuple(phase.lower() for phase in self.phases),
        )
        object.__setattr__(self, "n_phases", len(self.phases))

        if self.dim not in VALID_DIMENSIONS:
            raise ValueError(
                f"dim must be one of: {', '.join(str(d) for d in sorted(VALID_DIMENSIONS))}."  # noqa: E501
            )
        # Validate derivative method
        self.validate_derivative_method()

    def validate_derivative_method(self) -> None:
        """Validate the derivative method and normalize it to lowercase."""
        if not isinstance(self.derivative_method, str):
            raise ValueError(
                f"Derivative method must be a string, got {type(self.derivative_method)}."  # noqa: E501
            )
        normalized = self.derivative_method.lower()
        if normalized not in VALID_DERIVATIVE_METHODS:
            raise ValueError(
                f"Unsupported derivative method: {self.derivative_method!r}. Use one of: {', '.join(sorted(VALID_DERIVATIVE_METHODS))}."  # noqa: E501
            )
        object.__setattr__(self, "derivative_method", normalized)


@dataclass(frozen=True)
class RunConfig:
    """
    Immutable, per-run configuration for a single ``locate()`` call.

    Holds knobs that only take effect while a location run is executing and can
    be varied freely between runs on the same loaded data without rebuilding any
    component: distance weighting, the observation error model, and the RNG seed
    used for location initialization. Built inside ``EventLocator.locate`` from
    its arguments and passed (read-only) to the run-time components.
    """

    distance_weighting_enabled: bool = False
    k: Optional[float] = None
    x0: Optional[float] = None
    full_data_covariance: bool = False
    correlation_length_km: float = 1.0
    error_coefficient: float = 1.0
    seed: Optional[int] = None
    solver: SolverLiteral = "lsqr"

    def __post_init__(self) -> None:
        """Validate post-init constraints."""
        if self.full_data_covariance:
            if (
                not isinstance(self.correlation_length_km, numbers.Real)
                or self.correlation_length_km <= 0.0
            ):
                raise ValueError("correlation_length_km must be > 0.")
            if (
                not isinstance(self.error_coefficient, numbers.Real)
                or self.error_coefficient < 0.0
            ):
                raise ValueError("error_coefficient must be >= 0.")

        if not isinstance(self.solver, str):
            raise ValueError(f"Solver must be a string, got {type(self.solver)}.")
        normalized_solver = self.solver.lower()
        if normalized_solver not in VALID_SOLVERS:
            raise ValueError(
                f"Unsupported solver: {self.solver!r}. Use one of: {', '.join(sorted(VALID_SOLVERS))}."  # noqa: E501
            )
        object.__setattr__(self, "solver", normalized_solver)
