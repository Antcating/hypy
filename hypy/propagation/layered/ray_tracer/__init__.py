"""Layered ray tracing module for seismic wave propagation."""

import functools
import math
from typing import Any

from hypy.model import VelocityModel
from hypy.propagation.layered import LayeredPropagationEngine
from hypy.utils.seismic import normalize_phase


class LayeredRayTracer(LayeredPropagationEngine):
    """Ray tracer for layered Earth models.

    Supports multiple backends (e.g., Pyrocko, TauP), but with unified
    interface for computing ray parameters like travel time and takeoff
    angle for given source-receiver configurations and seismic phases.

    Subclasses implement :meth:`_compute_arrival_1d`; normalization,
    range checking, and caching are handled here for all of them.
    """

    name: str

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

        self._arrival_cache_precision_km = arrival_cache_precision_km
        # Bound to the instance rather than decorating the method, so the
        # cache dies with the tracer instead of pinning it in a class-level
        # table. Location iterations re-request the same geometries often.
        self._cached_arrival = functools.lru_cache(maxsize=arrival_cache_size)(
            self._compute_quantized_arrival
        )

    def _compute_quantized_arrival(
        self,
        phase_key: str,
        distance_steps: int,
        depth_steps: int,
    ) -> tuple[float, float]:
        """Compute an arrival from a cache key, in units of the precision.

        Working from the quantized geometry rather than the raw query makes
        every hit on a key return the same numbers, independent of which
        query happened to populate it first.
        """
        precision_km = self._arrival_cache_precision_km
        return self._compute_arrival_1d(
            distance_steps * precision_km,
            depth_steps * precision_km,
            phase_key,
        )

    def _compute_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return ``(travel_time_s, takeoff_angle_deg)`` from the backend.

        Arguments are already normalized and range-checked by
        :meth:`get_arrival_1d`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Return the first arrival for one 1-D source/receiver geometry.

        Negative distance is mirrored to positive distance because a 1-D
        calculation depends only on offset magnitude, not azimuth.
        """
        phase_key = normalize_phase(phase)
        if not math.isfinite(arc_distance_km) or not math.isfinite(source_depth_km):
            raise ValueError("Distance and depth must be finite.")
        distance_km = abs(float(arc_distance_km))
        depth_km = float(source_depth_km)

        if depth_km < 0.0:
            raise ValueError("source_depth_km must be non-negative.")
        if self.max_distance_km is not None and distance_km > self.max_distance_km:
            raise ValueError(
                f"Distance {distance_km:.3f} km exceeds configured local "
                f"coverage {self.max_distance_km:.3f} km. Increase max_distance_km."
            )
        if self.max_depth_km is not None and depth_km > self.max_depth_km:
            raise ValueError(
                f"Depth {depth_km:.3f} km exceeds configured local "
                f"coverage {self.max_depth_km:.3f} km. Increase max_depth_km."
            )

        steps_per_km = 1.0 / self._arrival_cache_precision_km
        return self._cached_arrival(
            phase_key,
            int(round(distance_km * steps_per_km)),
            int(round(depth_km * steps_per_km)),
        )

    def clear_arrival_cache(self) -> None:
        """Clear cached travel-time/takeoff results."""
        self._cached_arrival.cache_clear()

    def arrival_cache_info(self) -> dict[str, int | float]:
        """Return basic lazy-arrival-cache settings and current size."""
        info = self._cached_arrival.cache_info()
        return {
            "size": info.currsize,
            "max_size": info.maxsize,
            "precision_km": self._arrival_cache_precision_km,
        }
