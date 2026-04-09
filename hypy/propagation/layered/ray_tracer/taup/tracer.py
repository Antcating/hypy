"""1D TauP Ray Tracer based on obspy's TauP module.

Performs seismic ray tracing using the obspy's TauP toolkit.
This module defines the TauPRayTracer class,
which implements the LayeredRayTracer interface using TauP
for travel time and takeoff angle calculations.

Raises
------
    ValueError: If an unsupported seismic phase is requested.
    ValueError: If no arrivals are found for the given parameters.
    ValueError: If an error occurs during arrival computation.

Returns
-------
    (float, float): The arrival time and takeoff angle.

"""

from typing import List, Optional

from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from obspy.taup.tau import Arrival  # type: ignore[import-untyped]

from hypy.logger import logger
from hypy.propagation.layered.ray_tracer import LayeredRayTracer


class TauPRayTracer(LayeredRayTracer):
    """Seismic Ray Tracer using obspy's TauP toolkit.

    It provides methods to compute ray parametes for seismic phases
    and given source-receiver configurations and a 1D velocity model.

    Parameters
    ----------
    LayeredRayTracer : LayeredPropagationModel
        Base class for layered ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model: TauPyModel = TauPyModel(self.model_name)

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ):
        """Calculate time and takeoff angle for a given depth, distance, and phase.

        For 1D layered models, the arrival time and takeoff angle depend only
        on the source depth and the arc distance between source and receiver,
        not on the specific geographic locations,
        so we can compute these parameters in a 1D context using TauP.

        Parameters
        ----------
        arc_distance_km : float
            Great-circle distance between source and receiver in kilometers.
        source_depth_km : float
            Depth of the seismic source in kilometers.
        phase : str
            Seismic phase for which to compute arrivals.

        Returns
        -------
        (float, float)
            The arrival time and takeoff angle.

        Raises
        ------
        ValueError
            If an unsupported seismic phase is requested.
        ValueError
            If no arrivals are found for the given parameters.
        ValueError
            If an error occurs during arrival computation.
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
                f"Error computing arrivals for phase {phase}"
                f" at distance {arc_distance_km} and depth {source_depth_km}: {e}"
            )
        if len(arrivals) == 0:
            raise ValueError(
                f"No arrivals found for phase {phase} "
                f"at distance {arc_distance_km} and depth {source_depth_km}."
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
        """Get arrival parameters for a given source-receiver configuration.

        The distance is computed between the source and receiver locations, and then
        the arrival time and takeoff angle are computed using TauP for the
        specified seismic phase.

        Parameters
        ----------
        source_latitude : float
            Latitude of the seismic source in degrees.
        source_longitude : float
            Longitude of the seismic source in degrees.
        source_depth_km : float
            Depth of the seismic source in kilometers.
        receiver_latitude : float
            Latitude of the receiver in degrees.
        receiver_longitude : float
            Longitude of the receiver in degrees.
        receiver_elevation_m : float
            Elevation of the receiver in meters.
        phase : str
            Seismic phase for which to compute arrivals.
        max_distance_km : float | None, optional
            Maximum distance for which to compute arrivals, by default None

        Returns
        -------
        tuple[float, Optional[float], Optional[float], float]
            A tuple containing:

            arc_distance_km : float
                Great-circle distance between source and receiver in kilometers.
            arrival_time : float or None
                Arrival time in seconds, or None if (distance >= max_distance_km).
            takeoff_angle : float or None
                Takeoff angle in degrees, or None if (distance >= max_distance_km).
            azimuth_deg : float
                Azimuth from source to receiver in degrees (clockwise from North).
        """
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

        time, takeoff_angle = self.get_arrival_1d(
            arc_distance_km=arc_distance_km,
            source_depth_km=eff_source_depth_km,
            phase=phase,
        )

        return arc_distance_km, time, takeoff_angle, azimuth_deg
