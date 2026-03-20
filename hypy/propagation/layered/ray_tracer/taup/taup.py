from typing import List, Optional

from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.tau import Arrival  # type: ignore[import-untyped]

from hypy.propagation.layered.ray_tracer import LayeredRayTracer
import numpy as np

from hypy.utils.geo import gps2dist_azimuth_nb

from hypy.logger import logger


class TauPRayTracer(LayeredRayTracer):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model: TauPyModel = TauPyModel(self.model_name)

    def get_ray_params(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ):
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
                source_depth_in_km=source_depth_km,
                distance_in_degree=kilometer2degrees(arc_distance_km),
                phase_list=phases,
            )
        except Exception as e:
            raise ValueError(
                f"Error computing arrivals for phase {phase} at distance {arc_distance_km} and depth {source_depth_km}: {e}"
            )
        if len(arrivals) == 0:
            raise ValueError(
                f"No arrivals found for phase {phase} at distance {arc_distance_km} and depth {source_depth_km}."
            )

        return arrivals[0].time, arrivals[0].takeoff_angle

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
    ) -> tuple[float, Optional[float], Optional[float], float]:

        arc_distance_km, azimuth_deg, back_azimuth_deg = self._compute_arc_params(
            source_latitude,
            source_longitude,
            receiver_latitude,
            receiver_longitude,
        )

        eff_source_depth_km = source_depth_km + receiver_elevation_m / 1000.0

        # Check max distance
        # This check is nessesary when using Tables with limited grid coverage
        if max_distance_km is not None and arc_distance_km > max_distance_km:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                arc_distance_km,
                max_distance_km,
            )
            return arc_distance_km, None, None, azimuth_deg

        time, takeoff_angle = self.get_ray_params(
            arc_distance_km=arc_distance_km,
            source_depth_km=eff_source_depth_km,
            phase=phase,
        )

        return arc_distance_km, time, takeoff_angle, azimuth_deg
