"""Base propagation model contract.

This module defines the common runtime interface expected by locator code.
Concrete propagation backends can implement ray-tracing or precomputed
table strategies while conforming to this API.
"""

from pathlib import Path
from typing import Optional


class PropagationEngine:
    """Common base for propagation backends used by the locator."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).stem

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
