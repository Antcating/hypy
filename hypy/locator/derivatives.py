from typing import Union

import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..propagation import PropagationModel
from ..utils.weighting import distance_weight
from .config import LocatorConfig
from .data import DataHandler
from .model import VelocityModelHandler


class DerivativeEngine:
    """
    Computes travel-time derivatives and fills the design (Jacobian) matrix.

    Operates on DataHandler arrays in-place but does not own them.
    All dependencies are passed explicitly via the constructor.
    """

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        model_handler: VelocityModelHandler,
        tracer: PropagationModel,
    ) -> None:
        self.config = config
        self.data = data
        self.model_handler = model_handler
        self.tracer = tracer

    def derive_travel_time_param(
        self,
        event_location: "tuple[float, float, float]",  # (lat, lon, depth_km)
        station_location: "tuple[float, float, float]",  # (lat, lon, elev_m)
        phase: str,
    ) -> "tuple[float, Union[float, None], tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]]":
        """
        Compute great-circle distance/azimuth, travel time, and spatial derivatives
        for a given event-station pair and phase.

        Parameters
        ----------
        event_location : tuple of float
            (latitude, longitude, depth_km) of the event.
        station_location : tuple of float
            (latitude, longitude, elevation_m) of the station.
        phase : str
            Seismic phase name (e.g., 'P', 'S').

        Returns
        -------
        (distance_km, travel_time, (dt_dr, dt_dx, dt_dy, dt_dz))
            If the path exceeds max_distance, travel_time and derivatives are None.
        """
        ev_lat, ev_lon, ev_depth_km = event_location
        st_lat, st_lon, st_elev_m = station_location

        arc_distance_km, travel_time_s, takeoff_angle_deg, azimuth_deg = (
            self.tracer.get_arrival(
                source_latitude=ev_lat,
                source_longitude=ev_lon,
                source_depth_km=ev_depth_km,
                receiver_latitude=st_lat,
                receiver_longitude=st_lon,
                receiver_elevation_m=st_elev_m,
                phase=phase,
                max_distance_km=self.config.max_distance_km,
            )
        )
        # TODO: Expose method to compute derivatives (analytical or numerical).
        # dt/dr (ray), dt/dx (east), dt/dy (north), dt/dz (down)
        dt_dr, dt_dx, dt_dy, dt_dz = self.tracer.get_spatial_slowness(
            arc_distance_km=arc_distance_km,
            takeoff_angle_deg=takeoff_angle_deg,
            source_depth_km=ev_depth_km + st_elev_m / 1000.0,
            az_deg=azimuth_deg,
            phase=phase,
            mode=self.config.derivative_method,
            delta=self.config.derivative_delta_km,
        )
        return arc_distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

    def populate_derivative_matrix(self) -> np.ndarray:
        """
        Fill the design (derivative) matrix rows corresponding to all valid
        event-station-phase observations, and compute theoretical arrivals.

        Returns
        -------
        np.ndarray
            1D array of row indices (into the global design matrix) that were populated.
        """
        config = self.config
        data = self.data
        valid_row_indices: list[int] = []

        for event_index in range(data.n_events):
            # Skip invalid events
            if data.invalid_events[event_index]:
                continue

            ev_lat, ev_lon, ev_depth_km = data.locations[event_index]

            for station_index in range(data.n_stations):
                st_lat, st_lon, st_elev_m = data.stations_locations[station_index]

                for phase_index, phase_name in enumerate(config.phases):
                    # Column block start for this event in the global design matrix
                    col0 = event_index * config.dim

                    # Column within the event's observation vector for this station–phase
                    obs_col = config.n_phases * station_index + phase_index

                    # If no observed pick, skip row
                    observed_time_s = data.arrivals[event_index, obs_col]
                    if np.isnan(observed_time_s):
                        continue

                    # Compute travel time and spatial derivatives
                    distance_km, phase_travel_time, (_, dt_dx, dt_dy, dt_dz) = (
                        self.derive_travel_time_param(
                            (ev_lat, ev_lon, ev_depth_km),
                            (st_lat, st_lon, st_elev_m),
                            phase_name,
                        )
                    )
                    # Skip if no travel time (e.g., out of range)
                    if phase_travel_time is None:
                        logger.warning(
                            "No travel time for event %d, station %s, phase %s → skip.",
                            event_index,
                            data.stations[station_index],
                            phase_name,
                        )
                        continue

                    # Store theoretical travel time
                    data.theoretical_travel_times[event_index, obs_col] = (
                        phase_travel_time
                    )

                    # Row index in the global design matrix for this observation
                    row_index = (
                        event_index * (data.n_stations * config.n_phases) + obs_col
                    )

                    # Optional distance-based weighting
                    if config.distance_weighting_enabled:
                        data.distance_W[row_index] = distance_weight(
                            distance_km, k=config.k, x0=config.x0
                        )

                    # Fill design matrix:
                    # follow original convention: [y, x, z, t0] ordering in columns
                    data.derivative_matrix[row_index, col0 + 0] = dt_dy or 0.0
                    data.derivative_matrix[row_index, col0 + 1] = dt_dx or 0.0
                    if not data.fixed_depth_mask[event_index]:
                        data.derivative_matrix[row_index, col0 + 2] = dt_dz or 0.0
                    data.derivative_matrix[row_index, col0 + 3] = 1.0  # dt/dt

                    valid_row_indices.append(row_index)

        return np.array(valid_row_indices, dtype=int)
