from typing import List

import numpy as np
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel
from obspy.taup.tau import Arrival

from hypy.propagation import PropagationModel
from hypy.utils.geo import get_spatial_slowness_1d, gps2dist_azimuth_nb


class TauPRayTracer(PropagationModel):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = TauPyModel(self.model_name)

    def compute_travel_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lon: float,
        receiver_lat: float,
        receiver_elev_m: float,
        phase: str,
    ) -> "tuple[float, float, tuple[float, float, float]]":
        """
        Get arrival parameters for all phases between source and receiver.
        """
        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            source_lat,
            source_lon,
            receiver_lat,
            receiver_lon,
            R=self.model.model.radius_of_planet * 1000.0,
        )
        distance_km = dist_m / 1000.0

        source_depth_km += receiver_elev_m / 1000.0  # account for station elevation

        travel_time, takeoff_angle = self.get_arrival(
            source_depth=source_depth_km, distance=distance_km, phase=phase
        )
        # Local velocity and partial derivatives
        velocity_km_s = self.model.model.s_mod.v_mod.evaluate_below(
            source_depth_km, phase
        )[0]
        takeoff_rad = np.deg2rad(takeoff_angle)

        # dt/dr (ray), dt/dx (east), dt/dy (north), dt/dz (down)
        _, dt_dx, dt_dy, dt_dz = get_spatial_slowness_1d(
            takeoff_rad, azimuth_deg, velocity_km_s
        )

        return distance_km, travel_time, (dt_dx, dt_dy, dt_dz)

    def compute_bulk_arrival_parameters(
        self,
        source_lon: float,
        source_lat: float,
        source_depth_km: float,
        receiver_lons: List[float],
        receiver_lats: List[float],
        receiver_elevs_m: List[float],
        phase: str,
    ):
        """
        Get arrival parameters for all phases between source and multiple receivers.
        """
        results = []
        for receiver_lon, receiver_lat, receiver_elev_m in zip(
            receiver_lons, receiver_lats, receiver_elevs_m
        ):
            result = self.compute_travel_parameters(
                source_lon,
                source_lat,
                source_depth_km,
                receiver_lon,
                receiver_lat,
                receiver_elev_m,
                phase,
            )
            results.append(result)
        return results

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        """
        Get the arrival time for a given source depth, distance, and seismic phase.
        """
        if phase.lower() == "p":
            phases = ["p", "P"]
        elif phase.lower() == "s":
            phases = ["s", "S"]
        else:
            raise ValueError(
                f"Unsupported phase: {phase}. Supported phases are 'P' and 'S'."
            )

        # Convert distance from km to degrees
        try:
            arrivals: List[Arrival] = self.model.get_ray_paths(
                source_depth_in_km=source_depth,
                distance_in_degree=kilometer2degrees(distance),
                phase_list=phases,
            )
        except Exception as e:
            raise ValueError(
                f"Error computing arrivals for phase {phase} at distance {distance} and depth {source_depth}: {e}"
            )
        if len(arrivals) == 0:
            raise ValueError(
                f"No arrivals found for phase {phase} at distance {distance} and depth {source_depth}."
            )

        return arrivals[0].time, arrivals[0].takeoff_angle
