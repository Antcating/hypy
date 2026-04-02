from pathlib import Path

from obspy.geodetics import kilometers2degrees

from hypy.locator.model import VelocityModelHandler
from hypy.logger import logger
from hypy.propagation import PropagationModel
from hypy.utils.geo import gps2dist_azimuth_nb
from hypy.utils.math import finite_difference_derivative


class VolumetricPropagationModel(PropagationModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.velocity_model = VelocityModelHandler(Path(model_path))

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
    ) -> tuple[float, float, tuple[float, float, float, float]]:
        arc_distance_km, travel_time, _, _ = self.get_arrival(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            source_depth_km=source_depth_km,
            receiver_latitude=receiver_latitude,
            receiver_longitude=receiver_longitude,
            receiver_elevation_m=receiver_elevation_m,
            phase=phase,
        )

        if delta is None:
            raise ValueError(
                "Delta must be provided for numerical derivative computation."
            )

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
        """
        Compute spatial slowness derivatives (dt/dx, dt/dy, dt/dz) using numerical differentiation
        of arrival times computed via ray tracing.
        """

        dt_dy = finite_difference_derivative(
            lambda lat: self.get_arrival(
                source_latitude=lat,
                source_longitude=source_longitude,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1],
            source_latitude,
            kilometers2degrees(delta_km),
            method=method,
        )

        dt_dx = finite_difference_derivative(
            lambda lon: self.get_arrival(
                source_latitude=source_latitude,
                source_longitude=lon,
                source_depth_km=source_depth_km,
                receiver_latitude=receiver_latitude,
                receiver_longitude=receiver_longitude,
                receiver_elevation_m=receiver_elevation_m,
                phase=phase,
            )[1],
            source_longitude,
            kilometers2degrees(delta_km),
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
