"""Layered ray tracing module for seismic wave propagation."""

from collections import OrderedDict
from typing import Any

from hypy.model import VelocityModel
from hypy.propagation.layered import LayeredPropagationEngine


class LayeredRayTracer(LayeredPropagationEngine):
    """
    Ray tracer for layered Earth models.

    Supports multiple backends (e.g., Pyrocko, TauP), but with
    unified interface for computing ray parameters like travel time and takeoff angle
    for given source-receiver configurations and seismic phases.
    """

    def __init__(
        self,
        model: VelocityModel,
        *,
        arrival_cache_size: int = 200_000,
        arrival_cache_precision_km: float = 1.0e-6,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)

        if arrival_cache_size < 0:
            raise ValueError("arrival_cache_size must be non-negative.")
        if arrival_cache_precision_km <= 0.0:
            raise ValueError("arrival_cache_precision_km must be positive.")

        self._arrival_cache_size = arrival_cache_size
        self._arrival_cache_precision_km = arrival_cache_precision_km
        self._arrival_cache: OrderedDict[tuple[str, int, int], tuple[float, float]] = (
            OrderedDict()
        )

    def _arrival_cache_key(
        self,
        phase_key: str,
        distance_km: float,
        depth_km: float,
    ) -> tuple[str, int, int]:
        """Build a stable arrival-cache key from phase and quantized geometry."""
        scale = 1.0 / self._arrival_cache_precision_km
        return (
            phase_key,
            int(round(distance_km * scale)),
            int(round(depth_km * scale)),
        )

    def _arrival_cache_get(
        self,
        key: tuple[str, int, int],
    ) -> tuple[float, float] | None:
        """Return a cached ``(travel_time, takeoff_angle)`` arrival if present."""
        if self._arrival_cache_size == 0:
            return None
        cached = self._arrival_cache.get(key)
        if cached is None:
            return None
        self._arrival_cache.move_to_end(key)
        return cached

    def _arrival_cache_put(
        self,
        key: tuple[str, int, int],
        value: tuple[float, float],
    ) -> None:
        """Store a ``(travel_time, takeoff_angle)`` arrival in the LRU cache."""
        if self._arrival_cache_size == 0:
            return
        self._arrival_cache[key] = value
        self._arrival_cache.move_to_end(key)
        while len(self._arrival_cache) > self._arrival_cache_size:
            self._arrival_cache.popitem(last=False)

    def clear_arrival_cache(self) -> None:
        """Clear cached travel-time/takeoff results."""
        self._arrival_cache.clear()

    def arrival_cache_info(self) -> dict[str, int | float]:
        """Return basic lazy-arrival-cache settings and current size."""
        return {
            "size": len(self._arrival_cache),
            "max_size": self._arrival_cache_size,
            "precision_km": self._arrival_cache_precision_km,
        }
