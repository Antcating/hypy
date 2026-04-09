"""Layered propagation model base classes."""

from pathlib import Path
from typing import Optional

import numpy as np

from hypy.model import TvelModel
from hypy.propagation.base import PropagationEngine
from hypy.utils.geo import get_analytical_derivative_1d, gps2dist_azimuth_nb
from hypy.utils.math import finite_difference_derivative


class LayeredPropagationEngine(PropagationEngine):
    """Base class for layered 1D propagation backends."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.velocity_model = TvelModel(Path(model_path))

    @property
    def dim(self) -> int:
        """Return the dimensionality of the propagation model.

        For a layered 1D model, this is typically ``1``.
        """
        return 1

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Get the arrival time and takeoff angle for a 1D source/phase pair."""
        raise NotImplementedError("This method should be implemented by subclasses")

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
    ) -> tuple[float, float, tuple[float, float, float, float]]:
        """Return distance, travel time, and derivatives for a source/receiver pair."""
        arc_distance_km, travel_time_s, takeoff_angle_deg, azimuth_deg = (
            self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )
        )

        dt_dr, dt_dx, dt_dy, dt_dz = self.compute_derivatives(
            arc_distance_km=arc_distance_km,
            source_depth_km=source_depth_km + receiver_elevation_m / 1000.0,
            az_deg=azimuth_deg,
            phase=phase,
            takeoff_angle_deg=takeoff_angle_deg,
            delta=delta,
            mode=derivative_method,
        )
        return arc_distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

    def _compute_arc_params(
        self,
        source_latitude: float,
        source_longitude: float,
        receiver_latitude: float,
        receiver_longitude: float,
    ) -> tuple[float, float, float]:
        """Compute the arc distance between source and receiver in kilometers."""
        dist_m, azimuth, back_azimuth = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.model.model.radius_of_planet * 1000.0,
        )
        return dist_m / 1000.0, azimuth, back_azimuth

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
        """Compute spatial slowness derivatives using finite differences."""
        dt_ddist = finite_difference_derivative(
            lambda d: self.get_arrival_1d(d, source_depth_km, phase)[0],
            arc_distance_km,
            delta,
            method=method,
        )
        dt_ddepth = finite_difference_derivative(
            lambda z: self.get_arrival_1d(arc_distance_km, z, phase)[0],
            source_depth_km,
            delta,
            method=method,
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

        Parameters are interpreted as source/receiver geometry in the locator
        coordinate system.
        """
        if mode == "analytical":
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
            # TODO: Expose numerical derivative stencils
            # (central, forward, backward, five-point, etc.).
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
