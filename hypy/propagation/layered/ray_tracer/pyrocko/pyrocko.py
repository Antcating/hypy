from typing import List

try:
    from pyrocko import cake  # type: ignore[import-untyped]
    from pyrocko.cake import Ray  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "Pyrocko is not installed or not available. Please install Pyrocko to use the pyrocko engine."
    )
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from hypy.propagation.layered.ray_tracer import LayeredRayTracer
from hypy.logger import logger


class PyrockoRayTracer(LayeredRayTracer):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model: cake.LayeredModel = cake.load_model(model_path)

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
            arrivals: List[Ray] = self.model.arrivals(
                zstart=source_depth_km * 1000,
                zstop=0,
                distances=[kilometer2degrees(arc_distance_km)],
                phases=phases,
            )
        except Exception as e:
            raise ValueError(
                f"Error computing arrivals for phase {phase} at distance {arc_distance_km} and depth {source_depth_km}: {e}"
            )
        if not arrivals:
            raise ValueError(
                f"No arrivals found for phase {phase} at distance {arc_distance_km} and depth {source_depth_km}."
            )

        return arrivals[0].t, arrivals[0].takeoff_angle()

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
    ):

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
