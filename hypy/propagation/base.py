"""Base propagation model contract.

This module defines the common runtime interface expected by locator
code. Concrete propagation backends can implement ray-tracing or
precomputed table strategies while conforming to this API.
"""

from typing import ClassVar, Optional

import numpy as np

from hypy.model import VelocityModel
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.seismic import normalize_phase


class PropagationEngine:
    """Common base for propagation backends used by the locator."""

    feature_flags: ClassVar[frozenset[str]] = frozenset()
    required_model_features: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        model: VelocityModel,
        *,
        mode: str | None = None,
        engine: str | None = None,
        max_depth_km: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        depth_step_km: Optional[float] = None,
        distance_step_km: Optional[float] = None,
        phases: Optional[tuple[str, ...]] = ("p", "s"),
    ):
        self.velocity_model = model
        self.mode = mode
        self.engine = engine
        self.phases = (
            tuple(normalize_phase(phase) for phase in phases)
            if phases is not None
            else ("p", "s")
        )
        self.max_depth_km = (
            max_depth_km if max_depth_km is not None else model.radius_of_planet_km
        )
        self.max_distance_km = max_distance_km
        self.depth_step_km = depth_step_km
        self.distance_step_km = distance_step_km

        self.validate_model(model)

    @property
    def name(self) -> str:
        """Return a human-readable name for this propagation engine."""
        return f"PropagationEngine({self.velocity_model.model_name})"

    @property
    def supported_features(self) -> frozenset[str]:
        """Return propagation feature flags exposed by this backend
        instance.
        """
        inferred: set[str] = set(self.feature_flags)
        if self.mode:
            inferred.add(f"mode.{self.mode}")
        if self.engine:
            inferred.add(f"engine.{self.engine}")
        if self.dim > 0:
            inferred.add(f"dimension.{self.dim}d")
        return frozenset(inferred)

    def supports_feature(self, feature: str) -> bool:
        """Return whether this backend supports a single feature flag."""
        return feature in self.supported_features

    @property
    def dim(self) -> int:
        """Return the propagation dimensionality for this backend."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def _compute_arc_params(
        self,
        source_latitude: float,
        source_longitude: float,
        receiver_latitude: float,
        receiver_longitude: float,
    ) -> tuple[float, float, float]:
        """Return arc distance (km), azimuth, and back-azimuth."""
        dist_m, azimuth, back_azimuth = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.radius_of_planet_km * 1000.0,
        )
        return dist_m / 1000.0, azimuth, back_azimuth

    def get_arrival(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        max_distance_km: Optional[float] = None,
    ) -> tuple[float, Optional[float], Optional[float], float]:
        """Return arc distance, travel time, takeoff angle, and azimuth."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_arrival_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        phase: str,
        max_distance_km: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arc distance, travel time, takeoff angle, and azimuth from batch."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def compute_travel_time_params(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta: Optional[float] = None,
        derivative_method: str = "analytical",
    ) -> tuple[float, float, tuple[float, float, float, float]]:
        """Return distance, travel time, and spatial derivatives."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def compute_travel_time_params_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        phase: str,
        delta: Optional[float] = None,
        derivative_method: str = "analytical",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, travel time, takeoff angle, and azimuth for a batch of source/receiver pairs."""  # noqa: E501
        raise NotImplementedError("This method should be overridden by subclasses")

    def compute_derivatives(self, *args, **kwargs):
        """Return spatial slowness derivatives for a source/receiver pair."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def validate_model(self, model: VelocityModel) -> bool:
        """Check if the provided velocity model is compatible with this
        backend.
        """
        if not self.required_model_features.issubset(model.feature_flags):
            missing = self.required_model_features - model.feature_flags
            raise ValueError(
                f"Velocity model {model.model_name!r} is "
                f"missing required features: {missing}"
            )
        return True
