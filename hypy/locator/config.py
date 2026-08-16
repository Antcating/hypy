"""Handling configurations for the event locator."""

import numbers
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .types import (
    VALID_DERIVATIVE_METHODS,
    VALID_DIMENSIONS,
    VALID_SOLVERS,
    DerivativeMethodLiteral,
    SolverLiteral,
)


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
    # Optional geographic box that constrains event epicenters during both
    # location seeding and inversion iterations. Either all four bounds are
    # set, or all are None (then seeding falls back to a station envelope and
    # iterations are unconstrained horizontally).
    search_lat_min: Optional[float] = None
    search_lat_max: Optional[float] = None
    search_lon_min: Optional[float] = None
    search_lon_max: Optional[float] = None
    n_phases: int = field(init=False)

    @property
    def search_bounds(
        self,
    ) -> Optional[tuple[float, float, float, float]]:
        """Return ``(lat_min, lat_max, lon_min, lon_max)`` or ``None``."""
        bounds = (
            self.search_lat_min,
            self.search_lat_max,
            self.search_lon_min,
            self.search_lon_max,
        )
        if all(b is None for b in bounds):
            return None
        assert all(b is not None for b in bounds)
        return (
            float(self.search_lat_min),  # type: ignore[arg-type]
            float(self.search_lat_max),  # type: ignore[arg-type]
            float(self.search_lon_min),  # type: ignore[arg-type]
            float(self.search_lon_max),  # type: ignore[arg-type]
        )

    def clamp_horizontal(self, lat: float, lon: float) -> tuple[float, float]:
        """Clamp an epicenter into the configured search box when set."""
        bounds = self.search_bounds
        if bounds is None:
            return float(lat), float(lon)
        lat_min, lat_max, lon_min, lon_max = bounds
        return (
            min(max(float(lat), lat_min), lat_max),
            min(max(float(lon), lon_min), lon_max),
        )

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

        if self.default_time_error <= 0.0:
            raise ValueError("default_time_error must be > 0.0.")

        if self.derivative_delta_km <= 0.0:
            raise ValueError("derivative_delta_km must be > 0.0.")

        if self.min_picks_per_event <= 0:
            raise ValueError("min_picks_per_event must be > 0.")

        self._validate_search_bounds()

        # Validate derivative method
        self.validate_derivative_method()

    def _validate_search_bounds(self) -> None:
        """Require search lat/lon bounds to be all set or all ``None``."""
        bounds = (
            self.search_lat_min,
            self.search_lat_max,
            self.search_lon_min,
            self.search_lon_max,
        )
        n_set = sum(b is not None for b in bounds)
        if n_set == 0:
            return
        if n_set != 4:
            raise ValueError(
                "search_lat_min/search_lat_max/search_lon_min/search_lon_max "
                "must all be set or all be None."
            )
        if self.search_lat_min >= self.search_lat_max:  # type: ignore[operator]
            raise ValueError("search_lat_min must be < search_lat_max.")
        if self.search_lon_min >= self.search_lon_max:  # type: ignore[operator]
            raise ValueError("search_lon_min must be < search_lon_max.")

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
    component: distance/residual weighting, the observation error model, and the
    RNG seed used for location initialization. Built inside
    ``EventLocator.locate`` from its arguments and passed (read-only) to the
    run-time components.
    """

    distance_weighting_enabled: bool = False
    k: Optional[float] = None
    x0: Optional[float] = None
    residual_weighting_enabled: bool = False
    residual_start_iteration: Optional[int] = None
    residual_rms_floor_s: float = 0.1
    residual_full_weight_factor: float = 1.0
    residual_zero_weight_factor: float = 3.0
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

        if self.residual_weighting_enabled:
            if (
                self.residual_start_iteration is None
                or not isinstance(self.residual_start_iteration, numbers.Integral)
                or int(self.residual_start_iteration) < 1
            ):
                raise ValueError(
                    "residual_start_iteration must be an integer >= 1 when "
                    "residual weighting is enabled."
                )
            object.__setattr__(
                self, "residual_start_iteration", int(self.residual_start_iteration)
            )
            if (
                not isinstance(self.residual_rms_floor_s, numbers.Real)
                or self.residual_rms_floor_s <= 0.0
            ):
                raise ValueError("residual_rms_floor_s must be > 0.")
            if (
                not isinstance(self.residual_full_weight_factor, numbers.Real)
                or self.residual_full_weight_factor < 0.0
            ):
                raise ValueError("residual_full_weight_factor must be >= 0.")
            if (
                not isinstance(self.residual_zero_weight_factor, numbers.Real)
                or self.residual_zero_weight_factor <= self.residual_full_weight_factor
            ):
                raise ValueError(
                    "residual_zero_weight_factor must be greater than "
                    "residual_full_weight_factor."
                )
            object.__setattr__(
                self, "residual_rms_floor_s", float(self.residual_rms_floor_s)
            )
            object.__setattr__(
                self,
                "residual_full_weight_factor",
                float(self.residual_full_weight_factor),
            )
            object.__setattr__(
                self,
                "residual_zero_weight_factor",
                float(self.residual_zero_weight_factor),
            )
        elif self.residual_start_iteration is not None:
            raise ValueError(
                "residual_start_iteration must be None when residual weighting "
                "is disabled."
            )

        if not isinstance(self.solver, str):
            raise ValueError(f"Solver must be a string, got {type(self.solver)}.")
        normalized_solver = self.solver.lower()
        if normalized_solver not in VALID_SOLVERS:
            raise ValueError(
                f"Unsupported solver: {self.solver!r}. Use one of: {', '.join(sorted(VALID_SOLVERS))}."  # noqa: E501
            )
        object.__setattr__(self, "solver", normalized_solver)
