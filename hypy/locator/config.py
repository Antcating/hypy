"""Handling configurations for the event locator."""

import numbers
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..utils.seismic import normalize_phase
from .types import (
    VALID_DERIVATIVE_METHODS,
    VALID_UNRESOLVED_DEPTHS,
    DerivativeMethodLiteral,
    UnresolvedDepthLiteral,
)


@dataclass(frozen=True)
class LocatorConfig:
    """Read-only, construction-time configuration for the event locator.

    Holds settings that define are constant of a locator: the data
    layout, the physical/model parameters.
    These do not change between ``locate()`` calls on the
    same locator.

    Per-run knobs (weighting, error model, RNG seed) live in
    :class:`RunConfig`.
    """

    # Phases to use for travel-time calculations.
    # This can be used to limit the inversion to a subset of phases,
    # even if the input catalog contains picks for more phases.
    phases: Sequence[str]
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
    time_error_floor_s: float
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

    # TODO: Add ability to change delta
    derivative_delta_km: float = 0.5
    # Optional geographic box that constrains event epicenters during both
    # location seeding and inversion iterations. Either all four bounds are
    # set, or all are None (then seeding falls back to a station envelope and
    # iterations are unconstrained horizontally).
    latitude_min_deg: Optional[float] = None
    latitude_max_deg: Optional[float] = None
    longitude_min_deg: Optional[float] = None
    longitude_max_deg: Optional[float] = None
    n_phases: int = field(init=False)

    @property
    def search_bounds(
        self,
    ) -> Optional[tuple[float, float, float, float]]:
        """Return ``(lat_min, lat_max, lon_min, lon_max)`` or ``None``."""
        bounds = (
            self.latitude_min_deg,
            self.latitude_max_deg,
            self.longitude_min_deg,
            self.longitude_max_deg,
        )
        if all(b is None for b in bounds):
            return None
        assert all(b is not None for b in bounds)
        return (
            float(self.latitude_min_deg),  # type: ignore[arg-type]
            float(self.latitude_max_deg),  # type: ignore[arg-type]
            float(self.longitude_min_deg),  # type: ignore[arg-type]
            float(self.longitude_max_deg),  # type: ignore[arg-type]
        )

    def clamp_horizontal(self, lat: float, lon: float) -> tuple[float, float]:
        """Clamp an epicenter into the configured search box when set."""
        bounds = self.search_bounds
        if bounds is None:
            return float(lat), float(lon)
        lat_min, lat_max, lon_min, lon_max = bounds
        return (
            float(np.clip(lat, lat_min, lat_max)),
            float(np.clip(lon, lon_min, lon_max)),
        )

    def __post_init__(self) -> None:
        """Compute derived values and validate post-init constraints."""
        # Normalize phases
        object.__setattr__(
            self,
            "phases",
            tuple(normalize_phase(phase) for phase in self.phases),
        )
        object.__setattr__(self, "n_phases", len(self.phases))

        if self.time_error_floor_s <= 0.0:
            raise ValueError("time_error_floor_s must be > 0.0.")

        if self.derivative_delta_km <= 0.0:
            raise ValueError("derivative_delta_km must be > 0.0.")

        if self.min_picks_per_event <= 0:
            raise ValueError("min_picks_per_event must be > 0.")

        # Validate search bounds
        self._validate_search_bounds()

        # Validate derivative method
        self.validate_derivative_method()

    def _validate_search_bounds(self) -> None:
        """Require search lat/lon bounds to be all set or all ``None``."""
        bounds = (
            self.latitude_min_deg,
            self.latitude_max_deg,
            self.longitude_min_deg,
            self.longitude_max_deg,
        )
        n_set = sum(b is not None for b in bounds)
        if n_set == 0:
            return
        if n_set != 4:
            raise ValueError(
                "latitude_min_deg/latitude_max_deg/longitude_min_deg/longitude_max_deg "
                "must all be set or all be None."
            )
        if self.latitude_min_deg >= self.latitude_max_deg:  # type: ignore[operator]
            raise ValueError("latitude_min_deg must be < latitude_max_deg.")
        if self.longitude_min_deg >= self.longitude_max_deg:  # type: ignore[operator]
            raise ValueError("longitude_min_deg must be < longitude_max_deg.")

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
    """Read-only, per-run configuration for a ``locate()`` call.

    Holds knobs that take effect while a location run is executing and can
    be varied freely between runs on the same loaded data without rebuilding any
    component: distance/residual weighting, the observation error model, the
    RNG seed, and the heuristic ``num_trials`` budget. Built inside
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
    model_uncertainty_s: float = 0.15
    error_coefficient: float = 1.0
    seed: Optional[int] = None
    # Total heuristic seed budget (anchors plus Gaussian samples).
    num_trials: int = 8
    # When True, use each event's last input Origin (lat/lon/depth/time) as an
    # additional seed candidate during location initialization.
    use_origins: bool = False
    unresolved_depth: Optional[UnresolvedDepthLiteral] = "seed"

    def __post_init__(self) -> None:
        """Validate post-init constraints."""
        if isinstance(self.num_trials, bool) or not isinstance(
            self.num_trials, numbers.Integral
        ):
            raise ValueError("num_trials must be an integer >= 1.")
        if int(self.num_trials) < 1:
            raise ValueError("num_trials must be an integer >= 1.")
        object.__setattr__(self, "num_trials", int(self.num_trials))

        if (
            not isinstance(self.error_coefficient, numbers.Real)
            or self.error_coefficient < 0.0
        ):
            raise ValueError("error_coefficient must be >= 0.")
        object.__setattr__(self, "error_coefficient", float(self.error_coefficient))

        if (
            isinstance(self.model_uncertainty_s, bool)
            or not isinstance(self.model_uncertainty_s, numbers.Real)
            or not self.model_uncertainty_s > 0.0
        ):
            raise ValueError("model_uncertainty_s must be > 0.")
        object.__setattr__(self, "model_uncertainty_s", float(self.model_uncertainty_s))

        if self.full_data_covariance:
            if (
                not isinstance(self.correlation_length_km, numbers.Real)
                or self.correlation_length_km <= 0.0
            ):
                raise ValueError("correlation_length_km must be > 0.")

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

        if self.unresolved_depth is not None:
            if not isinstance(self.unresolved_depth, str):
                raise ValueError(
                    "unresolved_depth must be a string or None, "
                    f"got {type(self.unresolved_depth)}."
                )
            normalized_depth = self.unresolved_depth.strip().lower()
            if normalized_depth not in VALID_UNRESOLVED_DEPTHS:
                allowed = ", ".join(sorted(VALID_UNRESOLVED_DEPTHS))
                raise ValueError(
                    f"Unsupported unresolved_depth: {self.unresolved_depth!r}. "
                    f"Use one of: {allowed}, or None."
                )
            object.__setattr__(self, "unresolved_depth", normalized_depth)
