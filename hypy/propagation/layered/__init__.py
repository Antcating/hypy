"""Layered propagation model base classes."""

from typing import Optional

import numpy as np

from hypy.logger import logger
from hypy.model.layered.tvel import TvelModel
from hypy.propagation.base import PropagationEngine
from hypy.utils.geo import (
    get_analytical_derivative_1d,
    gps2dist_azimuth_batch_nb,
)
from hypy.utils.math import finite_difference_derivative


class LayeredPropagationEngine(PropagationEngine):
    """Base class for layered 1D propagation backends."""

    @property
    def dim(self) -> int:
        """Return the dimensionality of the propagation model.

        For a layered 1D model, this is typically ``1``.
        """
        return 1

    @property
    def name(self) -> str:
        """Return a human-readable name for this propagation engine."""
        return f"LayeredPropagationEngine({self.velocity_model.model_name})"

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Get the arrival time and takeoff angle for a 1D source/phase
        pair.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def effective_source_depth_km(
        self,
        source_depth_km: float,
        receiver_elevation_m: float | np.ndarray,
    ) -> float | np.ndarray:
        """Fold receiver elevation into the source depth and range-check it.

        A hypocenter above the free surface raises, so the locator sees that
        the trial left the Earth rather than being handed a fabricated
        surface arrival. Only the elevation term may push the result below
        the surface -- a receiver below sea level -- and that collapses onto
        it. Backends decide separately how to handle depths past the bottom
        of their coverage.
        """
        depth_km = float(source_depth_km)
        elevations = np.asarray(receiver_elevation_m, dtype=np.float64)
        if not np.isfinite(depth_km) or not np.all(np.isfinite(elevations)):
            raise ValueError("Source depth and receiver elevation must be finite.")
        if depth_km < 0.0:
            raise ValueError(
                f"Source depth {depth_km:.3f} km is above the free surface."
            )
        depths = np.maximum(0.0, depth_km + elevations / 1000.0)
        return float(depths) if depths.ndim == 0 else depths

    def get_arrival(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        max_distance_km: float | None = None,
    ) -> tuple[float, float | None, float | None, float]:
        """Return geometry and a 1-D arrival for layered ray tracers.

        This implementation is shared by TauP, Pyrocko, and other
        subclasses whose backend-specific work is entirely in
        ``get_arrival_1d``.
        """
        arc_distance_km, azimuth_deg, _ = self._compute_arc_params(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
        )
        # Validate the source before skipping a station, so a hypocenter
        # outside the model is reported even when no station is in reach.
        effective_depth_km = self.effective_source_depth_km(
            source_depth_km, receiver_elevation_m
        )
        if max_distance_km is not None and arc_distance_km > max_distance_km:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        time, takeoff_angle = self.get_arrival_1d(
            arc_distance_km=arc_distance_km,
            source_depth_km=effective_depth_km,
            phase=phase,
        )
        return arc_distance_km, float(time), float(takeoff_angle), azimuth_deg

    def get_arrival_batch(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: np.ndarray,
        receiver_longitude: np.ndarray,
        receiver_elevation_m: np.ndarray,
        phase: str,
        max_distance_km: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized :meth:`get_arrival` for layered ray tracers.

        Geometry is computed in batch; each in-range receiver still calls
        :meth:`get_arrival_1d`. Failed or out-of-range travel times and
        takeoff angles are ``NaN``.
        """
        receiver_latitude = np.asarray(receiver_latitude, dtype=np.float64).reshape(-1)
        receiver_longitude = np.asarray(receiver_longitude, dtype=np.float64).reshape(
            -1
        )
        receiver_elevation_m = np.asarray(
            receiver_elevation_m, dtype=np.float64
        ).reshape(-1)

        distances_m, azimuths, _ = gps2dist_azimuth_batch_nb(
            float(source_latitude),
            float(source_longitude),
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.radius_of_planet_km * 1000.0,
        )
        distances = distances_m / 1000.0
        depths = self.effective_source_depth_km(source_depth_km, receiver_elevation_m)
        travel_times = np.full(distances.shape, np.nan, dtype=np.float64)
        takeoff_angles = np.full(distances.shape, np.nan, dtype=np.float64)

        effective_max_distance = (
            float(max_distance_km)
            if max_distance_km is not None
            else self.max_distance_km
        )
        in_range = np.isfinite(distances) & np.isfinite(depths)
        if effective_max_distance is not None:
            in_range &= distances <= float(effective_max_distance)
        if self.max_depth_km is not None:
            in_range &= depths <= float(self.max_depth_km)

        for index in np.flatnonzero(in_range):
            try:
                time_s, takeoff_deg = self.get_arrival_1d(
                    float(distances[index]),
                    float(depths[index]),
                    phase,
                )
            except ValueError:
                continue
            travel_times[index] = time_s
            takeoff_angles[index] = takeoff_deg

        return distances, travel_times, takeoff_angles, azimuths

    def compute_travel_time_params(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta: float | None = None,
        derivative_method: str = "analytical",
    ) -> tuple[
        float,
        float | None,
        tuple[float | None, float | None, float | None, float | None],
    ]:
        """Return distance, travel time, and derivatives for a source/receiver
        pair.

        When the path exceeds the engine's tabulated/configured distance
        limit, ``travel_time`` and the derivative tuple entries are
        ``None``.
        """
        arc_distance_km, travel_time_s, takeoff_angle_deg, azimuth_deg = (
            self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
                max_distance_km=self.max_distance_km,
            )
        )

        if travel_time_s is None:
            return arc_distance_km, None, (None, None, None, None)

        dt_dr, dt_dx, dt_dy, dt_dz = self.compute_derivatives(
            arc_distance_km=arc_distance_km,
            source_depth_km=self.effective_source_depth_km(
                source_depth_km, receiver_elevation_m
            ),
            az_deg=azimuth_deg,
            phase=phase,
            takeoff_angle_deg=takeoff_angle_deg,
            delta=delta,
            mode=derivative_method,
        )
        return arc_distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

    def _compute_derivatives_analytical(
        self,
        takeoff_angle_deg: float,
        azimuth_deg: float,
        velocity_km_s: float,
    ) -> tuple[float, float, float, float]:
        takeoff_angle_rad = np.deg2rad(takeoff_angle_deg)
        dt_dr, dt_dx, dt_dy, dt_dz = get_analytical_derivative_1d(
            takeoff_angle_rad, azimuth_deg, velocity_km_s
        )
        return dt_dr, dt_dx, dt_dy, dt_dz

    def _compute_derivatives_numerical(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        azimuth_deg: float,
        phase: str,
        delta: float,
        method: str = "central",
    ) -> tuple[float, float, float, float]:
        """Compute spatial slowness derivatives using finite differences.

        Near the physical boundaries (zero epicentral distance and the
        free surface) a centered stencil would sample non-physical
        negative distance or depth, which the 1-D forward models reject.
        In those cases the stencil degrades to a one-sided forward
        difference.
        """
        dist_method = method
        if method == "central" and arc_distance_km - delta < 0.0:
            dist_method = "forward"
        dt_ddist = finite_difference_derivative(
            lambda d: self.get_arrival_1d(d, source_depth_km, phase)[0],
            arc_distance_km,
            delta,
            method=dist_method,
        )

        depth_method = method
        if method == "central" and source_depth_km - delta < 0.0:
            depth_method = "forward"
        dt_ddepth = finite_difference_derivative(
            lambda z: self.get_arrival_1d(arc_distance_km, z, phase)[0],
            source_depth_km,
            delta,
            method=depth_method,
        )
        az_rad = np.deg2rad(azimuth_deg)

        # Convert path-distance derivative to source-coordinate derivatives.
        # Increasing source x/y toward the receiver reduces source-receiver distance,
        # so source-coordinate horizontal derivatives carry a negative sign.
        dt_dr = -dt_ddist
        dt_dx = dt_dr * np.sin(az_rad)
        dt_dy = dt_dr * np.cos(az_rad)
        dt_dz = dt_ddepth

        return dt_dr, dt_dx, dt_dy, dt_dz

    def compute_derivatives(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        az_deg: float,
        phase: str,
        takeoff_angle_deg: Optional[float],
        delta: Optional[float] = 0.01,
        mode: str = "analytical",
    ):
        """Compute spatial slowness derivatives using the configured mode.

        Parameters are interpreted as source/receiver geometry in the
        locator coordinate system.
        """
        if mode == "analytical":
            self.velocity_model: TvelModel  # type hint for velocity lookup
            velocity_km_s = self.velocity_model.get_velocity_1d(source_depth_km, phase)
            if takeoff_angle_deg is None:
                raise ValueError(
                    "Takeoff angle is required for analytical derivative computation."
                )

            return self._compute_derivatives_analytical(
                takeoff_angle_deg=takeoff_angle_deg,
                azimuth_deg=az_deg,
                velocity_km_s=velocity_km_s,
            )
        elif mode == "numerical":
            if delta is None:
                raise ValueError(
                    "Delta must be provided for numerical derivative computation."
                )
            return self._compute_derivatives_numerical(
                arc_distance_km=arc_distance_km,
                source_depth_km=source_depth_km,
                azimuth_deg=az_deg,
                phase=phase,
                delta=delta,
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are analytical and numerical."
            )
