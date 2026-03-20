from typing import Optional

from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from pathlib import Path
from hypy.logger import logger
from hypy.propagation import PropagationModel
from hypy.locator.model import VelocityModelHandler
from hypy.utils.geo import gps2dist_azimuth_nb, get_spatial_slowness_1d
from hypy.utils.math import finite_difference_derivative

import numpy as np


class LayeredPropagationModel(PropagationModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.velocity_model = VelocityModelHandler(Path(model_path))

    def get_ray_params(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ) -> tuple[float, float]:
        """Get the arrival time and takeoff angle for a given source depth, distance, and seismic phase."""
        raise NotImplementedError("This method should be implemented by subclasses")

    def _compute_arc_params(
        self,
        source_latitude: float,
        source_longitude: float,
        receiver_latitude: float,
        receiver_longitude: float,
    ) -> tuple[float, float, float]:
        """
        Compute the arc distance between source and receiver in kilometers.
        """
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
        dt_dr, dt_dx, dt_dy, dt_dz = get_spatial_slowness_1d(
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
        """
        Compute spatial slowness derivatives (dt/dx, dt/dy, dt/dz) using numerical differentiation
        of arrival times computed via ray tracing.
        """

        dt_ddist = finite_difference_derivative(
            lambda d: self.get_ray_params(d, source_depth_km, phase)[0],
            arc_distance_km,
            delta,
            method=method,
        )
        dt_ddepth = finite_difference_derivative(
            lambda z: self.get_ray_params(arc_distance_km, z, phase)[0],
            source_depth_km,
            delta,
            method=method,
        )
        az_rad = np.deg2rad(azimuth_deg)

        dt_dr = dt_ddist
        dt_dx = dt_ddist * np.sin(az_rad)
        dt_dy = dt_ddist * np.cos(az_rad)
        dt_dz = dt_ddepth

        return dt_dr, dt_dx, dt_dy, dt_dz

    def get_spatial_slowness(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        az_deg: float,
        phase: str,
        takeoff_angle_deg: Optional[float],
        delta: Optional[float] = 0.01,
        mode: str = "analytical",
    ):
        """
        Compute spatial slowness derivatives (dt/dx, dt/dy, dt/dz) using numerical differentiation
        of arrival times computed via ray tracing.

        Args:
            source_depth: Source depth in km
            distance: Horizontal distance in km
            phase: Seismic phase ('P' or 'S')
            phase_vel: Phase velocity in km/s
            az_deg: Azimuth from source to receiver in degrees (clockwise from North)
            delta: Perturbation for numerical differentiation in km

        Returns:
            Tuple of (dt_dr, dt_dx, dt_dy, dt_dz) in s/km
        """

        if mode == "analytical":
            velocity_km_s = self.velocity_model._get_velocity_1d(source_depth_km, phase)
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
            # TODO: Expose method to computing numerical derivatives (central, forward, backward, five-point, etc.)
            return self._compute_derivatives_numerical(
                arc_distance_km=arc_distance_km,
                source_depth_km=source_depth_km,
                azimuth_deg=az_deg,
                phase=phase,
                delta=delta,
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are 'analytical' and 'numerical'."
            )
