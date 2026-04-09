"""Volumetric propagation backend for 3D travel-time calculations."""

import numpy as np

from hypy.logger import logger
from hypy.propagation.base import PropagationEngine
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.math import finite_difference_derivative


class VolumetricPropagationEngine(PropagationEngine):
    """Propagation backend that computes numerical 3D travel-time derivatives."""

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def _compute_arc_params(
        self,
        source_latitude: float,
        source_longitude: float,
        receiver_latitude: float,
        receiver_longitude: float,
    ) -> tuple[float, float, float]:
        """Compute arc distance and azimuth pair for a source-receiver geometry."""
        dist_m, azimuth, back_azimuth = gps2dist_azimuth_nb(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
            R=self.velocity_model.model.model.radius_of_planet * 1000.0,
        )
        return dist_m / 1000.0, azimuth, back_azimuth

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
        derivative_method: str = "numerical",
    ) -> tuple[
        float,
        float | None,
        tuple[float | None, float | None, float | None, float | None],
    ]:
        """Return travel time and spatial derivatives for a source-receiver pair."""
        arc_distance_km, travel_time, _, _ = self.get_arrival(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            source_depth_km=source_depth_km,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_elevation_m=receiver_elevation_m,
            phase=phase,
        )

        if travel_time is None:
            return arc_distance_km, None, (None, None, None, None)

        if delta is None:
            raise ValueError(
                "Delta must be provided for numerical derivative computation."
            )

        try:
            dt_dr, dt_dx, dt_dy, dt_dz = self._compute_derivatives_numerical(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
                delta_km=delta,
                method="central",
            )
        except ValueError as exc:
            logger.warning(
                "Unable to compute numerical derivatives for source=(%s, %s, %s) "
                "receiver=(%s, %s, %s) phase=%s: %s",
                source_latitude,
                source_longitude,
                source_depth_km,
                receiver_latitude,
                receiver_longitude,
                receiver_elevation_m,
                phase,
                exc,
            )
            return arc_distance_km, travel_time, (None, None, None, None)

        return arc_distance_km, travel_time, (dt_dr, dt_dx, dt_dy, dt_dz)

    def _compute_derivatives_numerical(
        self,
        source_latitude: float,
        source_longitude: float,
        source_depth_km: float,
        receiver_latitude: float,
        receiver_longitude: float,
        receiver_elevation_m: float,
        phase: str,
        delta_km: float,
        method: str = "central",
    ) -> tuple[float, float, float, float]:
        """Compute spatial slowness derivatives with numerical finite differences."""

        def _lat_offset_deg(delta_km: float) -> float:
            return float(delta_km / 111.0)

        def _lon_offset_deg(lat_deg: float, delta_km: float) -> float:
            return float(delta_km / (111.320 * np.cos(np.deg2rad(lat_deg))))

        def _travel_time_north(d_km: float) -> float:
            return self.get_arrival(
                source_latitude=source_latitude + _lat_offset_deg(d_km),
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1]

        def _travel_time_east(d_km: float) -> float:
            return self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude
                + _lon_offset_deg(source_latitude, d_km),
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1]

        dt_dy = finite_difference_derivative(
            _travel_time_north,
            0.0,
            delta_km,
            method=method,
        )

        dt_dx = finite_difference_derivative(
            _travel_time_east,
            0.0,
            delta_km,
            method=method,
        )

        depth_method = method
        if method == "central" and source_depth_km - delta_km < 0.0:
            depth_method = "forward"

        dt_dz = finite_difference_derivative(
            lambda z: self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                source_depth_km=z,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1],
            source_depth_km,
            delta_km,
            method=depth_method,
        )

        dt_dr = 0.0

        return dt_dr, dt_dx, dt_dy, dt_dz
