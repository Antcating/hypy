"""Derivative assembly for location inversion."""

from typing import Union

import numpy as np

from hypy.logger import logger  # type: ignore[import-not-found]

from ..propagation import PropagationEngine
from ..utils.weighting import distance_weight
from .config import LocatorConfig
from .data import DataHandler


class DerivativesHandler:
    """Compute travel time derivatives for seismic event location."""

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        tracer: PropagationEngine,
    ) -> None:
        self.config = config
        self.data = data
        self.tracer = tracer

    def derive_travel_time_param(
        self,
        event_location: "tuple[float, float, float]",  # (lat, lon, depth_km)
        station_location: "tuple[float, float, float]",  # (lat, lon, elev_m)
        phase: str,
    ) -> "tuple[float, Union[float, None], tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]]":  # noqa: E501
        """
        Compute travel time params for a given event-station pair and phase.

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

        return self.tracer.compute_travel_time_params(
            source_latitude=ev_lat,
            source_longitude=ev_lon,
            source_depth_km=ev_depth_km,
            receiver_latitude=st_lat,
            receiver_longitude=st_lon,
            receiver_elevation_m=st_elev_m,
            phase=phase,
            delta=self.config.derivative_delta_km,
            derivative_method=self.config.derivative_method,
        )

    def build_derivative_matrix(
        self, data: DataHandler, config: LocatorConfig
    ) -> np.ndarray:
        """Build the global derivative matrix for all events, stations, and phases.

        Returns
        -------
        np.ndarray
            Array of row indices corresponding to valid observations used in the matrix.
        """
        # Reset per-iteration inversion state so skipped stations are zeroed out.
        data.jacobian_matrix[:] = 0.0
        data.theoretical_travel_times[:] = 0.0
        if config.distance_weighting_enabled:
            data.distance_W[:] = 0.0
        else:
            data.distance_W[:] = 1.0

        observed_mask = ~np.isnan(data.arrivals)
        valid_row_mask = np.zeros(data.jacobian_matrix.shape[0], dtype=bool)

        for event_index in range(data.n_events):
            # Skip invalid events
            if data.invalid_events[event_index]:
                continue

            # Column block start for this event in the global design matrix
            col0 = event_index * config.dim
            ev_lat, ev_lon, ev_depth_km = data.locations[event_index]

            for station_index in range(data.n_stations):
                st_lat, st_lon, st_elev_m = data.stations_locations[station_index]
                station_obs_start = station_index * config.n_phases
                station_obs_end = station_obs_start + config.n_phases
                observed_for_station = observed_mask[
                    event_index, station_obs_start:station_obs_end
                ]
                if not observed_for_station.any():
                    continue

                for phase_index, phase_name in enumerate(config.phases):
                    if not observed_for_station[phase_index]:
                        continue

                    obs_col = station_obs_start + phase_index
                    row_index = (
                        event_index * (data.n_stations * config.n_phases) + obs_col
                    )

                    distance_km, theoretical_travel_time, (_, dt_dx, dt_dy, dt_dz) = (
                        self.derive_travel_time_param(
                            (ev_lat, ev_lon, ev_depth_km),
                            (st_lat, st_lon, st_elev_m),
                            phase_name,
                        )
                    )
                    if theoretical_travel_time is None:
                        data.theoretical_travel_times[event_index, obs_col] = 0.0
                        data.jacobian_matrix[row_index, :] = 0.0
                        data.distance_W[row_index] = 0.0
                        continue

                    derivatives = (dt_dx, dt_dy, dt_dz)
                    if any(v is None for v in derivatives) or any(
                        np.isnan(v) for v in derivatives if v is not None
                    ):
                        logger.warning(
                            "Skipping event %d, station %s, phase %s: invalid derivatives computed.",  # noqa: E501
                            event_index,
                            data.stations[station_index],
                            phase_name,
                        )
                        data.theoretical_travel_times[event_index, obs_col] = 0.0
                        data.jacobian_matrix[row_index, :] = 0.0
                        data.distance_W[row_index] = 0.0
                        continue

                    data.theoretical_travel_times[event_index, obs_col] = (
                        theoretical_travel_time
                    )

                    if config.distance_weighting_enabled:
                        data.distance_W[row_index] = distance_weight(
                            distance_km, k=config.k, x0=config.x0
                        )

                    data.jacobian_matrix[row_index, col0 + 0] = dt_dy  # type: ignore[index]
                    data.jacobian_matrix[row_index, col0 + 1] = dt_dx
                    if not data.fixed_depth_mask[event_index]:
                        data.jacobian_matrix[row_index, col0 + 2] = dt_dz
                    data.jacobian_matrix[row_index, col0 + 3] = 1.0  # dt/dt

                    valid_row_mask[row_index] = True

        valid_row_indices_arr = np.flatnonzero(valid_row_mask)
        if valid_row_indices_arr.size > 0:
            obs_cols = valid_row_indices_arr % (data.n_stations * config.n_phases)
            station_indices = np.unique(obs_cols // config.n_phases)
            logger.debug(
                "Using %d stations (%d obs) in current derivative build.",
                station_indices.size,
                valid_row_indices_arr.size,
            )

        return valid_row_indices_arr
