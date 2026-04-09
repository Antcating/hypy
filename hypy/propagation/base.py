"""Base propagation model contract.

This module defines the common runtime interface expected by locator code.
Concrete propagation backends can implement ray-tracing or precomputed
table strategies while conforming to this API.
"""

from pathlib import Path
from typing import ClassVar, Collection, Optional


class PropagationEngine:
    """Common base for propagation backends used by the locator."""

    default_feature_flags: ClassVar[frozenset[str]] = frozenset()
    default_required_model_features: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        model_path: str,
        *,
        mode: str | None = None,
        engine: str | None = None,
    ):
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.mode = mode
        self.engine = engine
        self.feature_flags: frozenset[str] = self.default_feature_flags
        self.required_model_features: frozenset[str] = (
            self.default_required_model_features
        )

    @property
    def supported_features(self) -> frozenset[str]:
        """Return propagation feature flags exposed by this backend instance."""
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

    def supports_all_features(self, features: Collection[str]) -> bool:
        """Return whether this backend supports all provided feature flags."""
        return set(features).issubset(self.supported_features)

    @property
    def dim(self) -> int:
        """Return the propagation dimensionality for this backend."""
        raise NotImplementedError("This method should be overridden by subclasses")

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

    def compute_derivatives(self, *args, **kwargs):
        """Return spatial slowness derivatives for a source/receiver pair."""
        raise NotImplementedError("This method should be overridden by subclasses")
