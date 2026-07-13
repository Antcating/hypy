"""1D TauP Ray Tracer based on pyrocko's cake module.

Performs seismic ray tracing using the pyrocko's cake toolkit.
This module defines the PyrockoRayTracer class,
which implements the LayeredRayTracer interface using pyrocko
for travel time and takeoff angle calculations.

Raises
------
    ImportError: If pyrocko is not installed or not available.
    ValueError: If an unsupported seismic phase is requested.
    ValueError: If no arrivals are found for the given parameters.
    ValueError: If an error occurs during arrival computation.

Returns
-------
    (float, float): The arrival time and takeoff angle.
"""

from typing import Any, List

from hypy.model import VelocityModel

try:
    from pyrocko import cake  # type: ignore[import-untyped]
    from pyrocko.cake import Ray  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "Pyrocko is not installed or not available."
        " Please install Pyrocko to use the pyrocko engine."
    )
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from hypy.logger import logger
from hypy.propagation.layered.ray_tracer import LayeredRayTracer
from hypy.utils.seismic import normalize_phase


class PyrockoRayTracer(LayeredRayTracer):
    """Seismic Ray Tracer using pyrocko's cake toolkit.

    It provides methods to compute ray parametes for seismic phases
    and given source-receiver configurations and a 1D velocity model.

    Parameters
    ----------
    LayeredRayTracer : LayeredPropagationModel
        Base class for layered ray tracing.

    Additional keyword arguments are forwarded to ``LayeredRayTracer``.  Use
    ``arrival_cache_size`` and ``arrival_cache_precision_km`` there to configure
    the shared layered-arrival cache
    """

    feature_flags = frozenset(
        {
            "propagation.layered",
            "propagation.ray",
            "derivative.analytical",
            "derivative.numerical",
        }
    )
    required_model_features = frozenset({"model.profile.1d", "model.profile.1d"})

    def __init__(self, model: VelocityModel, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.model: cake.LayeredModel = cake.load_model(model.model_path.as_posix())

    def get_arrival_1d(
        self,
        arc_distance_km: float,
        source_depth_km: float,
        phase: str,
    ):
        """Calculate time and takeoff angle for a given depth, distance, and phase.

        For 1D layered models, the arrival time and takeoff angle depend only
        on the source depth and the arc distance between source and receiver,
        not on their specific geographic locations,
        so we can compute these parameters in a 1D context using pyrocko's cake.

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
        phase_key = normalize_phase(phase)
        phases = [phase_key, phase_key.upper()]

        distance_km = float(abs(arc_distance_km))
        cache_key = self._arrival_cache_key(
            phase_key,
            distance_km,
            float(source_depth_km),
        )
        cached = self._arrival_cache_get(cache_key)
        if cached is not None:
            return cached

        # Convert distance from km to degrees
        try:
            arrivals: List[Ray] = self.model.arrivals(
                zstart=source_depth_km * 1000,
                zstop=0,
                distances=[kilometer2degrees(distance_km)],
                phases=phases,
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

        result = (float(arrivals[0].t), float(arrivals[0].takeoff_angle()))
        self._arrival_cache_put(cache_key, result)
        return result

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
