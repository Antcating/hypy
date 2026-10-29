"""Derivative assembly for location inversion."""

import numpy as np

from ..logger import logger
from ..propagation import PropagationEngine
from ..utils.weighting import distance_weight
from .config import LocatorConfig, RunConfig
from .data import DataHandler


class DerivativesHandler:
    """Compute travel times and spatial derivatives for event location."""

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        propagation_engine: PropagationEngine,
    ) -> None:
        self.config = config
        self.data = data
        self.propagation_engine = propagation_engine

    def derive_travel_time_param(
        self,
        event_location: tuple[float, float, float],
        station_location: tuple[float, float, float],
        phase: str,
    ) -> tuple[
        float,
        float | None,
        tuple[float | None, float | None, float | None, float | None],
    ]:
        """Return distance, travel time, and derivatives for one observation."""
        ev_lat, ev_lon, ev_depth_km = event_location
        st_lat, st_lon, st_elev_m = station_location

        return self.propagation_engine.compute_travel_time_params(
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

    def _build_table(
        self,
        data: DataHandler,
        config: LocatorConfig,
        run_config: RunConfig,
        event_indices: np.ndarray,
    ) -> np.ndarray:
        observed = ~np.isnan(data.arrivals)
        valid_rows = np.zeros(data.jacobian_matrix.shape[0], dtype=bool)
        n_obs = data.n_stations * config.n_phases

        for event_index in event_indices:
            event_index = int(event_index)
            if data.invalid_events[event_index]:
                continue

            event_location = tuple(data.locations[event_index])
            event_row0 = event_index * n_obs
            skipped = 0

            for obs_col in np.flatnonzero(observed[event_index]):
                station_index, phase_index = divmod(int(obs_col), config.n_phases)
                phase = config.phases[phase_index]
                station_location = tuple(data.stations_locations[station_index])
                row = event_row0 + obs_col

                distance_km, travel_time, (_, dt_dx, dt_dy, dt_dz) = (
                    self.derive_travel_time_param(
                        event_location,
                        station_location,
                        phase,
                    )
                )

                if travel_time is None:
                    skipped += 1
                    data.distance_W[row] = 0.0
                    continue

                if (
                    dt_dx is None
                    or dt_dy is None
                    or dt_dz is None
                    or not np.isfinite(dt_dx)
                    or not np.isfinite(dt_dy)
                    or not np.isfinite(dt_dz)
                ):
                    logger.warning(
                        "Skipping event %d, station %s, phase %s: "
                        "invalid derivatives computed.",
                        event_index,
                        data.stations[station_index],
                        phase,
                    )
                    data.distance_W[row] = 0.0
                    continue

                data.theoretical_travel_times[event_index, obs_col] = travel_time
                data.jacobian_matrix[row, 0] = dt_dy
                data.jacobian_matrix[row, 1] = dt_dx
                if not data.fixed_depth_mask[event_index]:
                    data.jacobian_matrix[row, 2] = dt_dz
                data.jacobian_matrix[row, 3] = 1.0

                if run_config.distance_weighting_enabled:
                    data.distance_W[row] = distance_weight(
                        distance_km,
                        k=run_config.k,
                        x0=run_config.x0,
                    )

                valid_rows[row] = True

            if skipped:
                logger.warning(
                    "Event %d: skipped %d station-phase observation(s) beyond "
                    "the propagation engine range (or with failed travel times).",
                    event_index,
                    skipped,
                )

        return np.flatnonzero(valid_rows)

    def _build_table_batch(
        self,
        data: DataHandler,
        config: LocatorConfig,
        run_config: RunConfig,
        event_indices: np.ndarray,
    ) -> np.ndarray:
        observed = ~np.isnan(data.arrivals)
        valid_rows = np.zeros(data.jacobian_matrix.shape[0], dtype=bool)
        n_obs = data.n_stations * config.n_phases

        station_lats = np.asarray(data.stations_locations[:, 0], dtype=np.float64)
        station_lons = np.asarray(data.stations_locations[:, 1], dtype=np.float64)
        station_elevs = np.asarray(data.stations_locations[:, 2], dtype=np.float64)

        for event_index in event_indices:
            event_index = int(event_index)
            if data.invalid_events[event_index]:
                continue

            event_lat, event_lon, event_depth_km = data.locations[event_index]
            event_row0 = event_index * n_obs

            for phase_index, phase in enumerate(config.phases):
                obs_cols = np.arange(
                    phase_index,
                    n_obs,
                    config.n_phases,
                    dtype=np.int64,
                )
                observed_mask = observed[event_index, obs_cols]
                if not observed_mask.any():
                    continue

                distances, travel_times, _dt_dr, dt_dx, dt_dy, dt_dz = (
                    self.propagation_engine.compute_travel_time_params_batch(
                        source_latitude=float(event_lat),
                        source_longitude=float(event_lon),
                        source_depth_km=float(event_depth_km),
                        receiver_latitude=station_lats,
                        receiver_longitude=station_lons,
                        receiver_elevation_m=station_elevs,
                        phase=phase,
                        delta=float(config.derivative_delta_km),
                        derivative_method=config.derivative_method,
                    )
                )

                finite = (
                    observed_mask
                    & np.isfinite(travel_times)
                    & np.isfinite(dt_dx)
                    & np.isfinite(dt_dy)
                    & np.isfinite(dt_dz)
                )
                if not finite.any():
                    continue

                rows = event_row0 + obs_cols[finite]
                active_obs_cols = obs_cols[finite]
                active_distances = distances[finite]

                data.theoretical_travel_times[event_index, active_obs_cols] = (
                    travel_times[finite]
                )
                data.jacobian_matrix[rows, 0] = dt_dy[finite]
                data.jacobian_matrix[rows, 1] = dt_dx[finite]
                if not data.fixed_depth_mask[event_index]:
                    data.jacobian_matrix[rows, 2] = dt_dz[finite]
                data.jacobian_matrix[rows, 3] = 1.0

                if run_config.distance_weighting_enabled:
                    data.distance_W[rows] = distance_weight(
                        active_distances,
                        k=run_config.k,
                        x0=run_config.x0,
                    )

                valid_rows[rows] = True

        return np.flatnonzero(valid_rows)

    def refresh_ttt(
        self,
        data: DataHandler,
        config: LocatorConfig,
        event_indices: np.ndarray,
    ) -> np.ndarray:
        """Refresh predicted travel times with scalar ``get_arrival``."""
        observed = ~np.isnan(data.arrivals)
        valid_rows = np.zeros(data.jacobian_matrix.shape[0], dtype=bool)
        n_obs = data.n_stations * config.n_phases
        data.theoretical_travel_times[event_indices] = np.nan
        max_distance_km = getattr(self.propagation_engine, "max_distance_km", None)

        for event_index in event_indices:
            event_index = int(event_index)
            if data.invalid_events[event_index]:
                continue

            event_lat, event_lon, event_depth_km = data.locations[event_index]
            event_row0 = event_index * n_obs

            for obs_col in np.flatnonzero(observed[event_index]):
                station_index, phase_index = divmod(int(obs_col), config.n_phases)
                station_lat, station_lon, station_elev_m = data.stations_locations[
                    station_index
                ]
                _distance, travel_time, _takeoff, _azimuth = (
                    self.propagation_engine.get_arrival(
                        source_latitude=float(event_lat),
                        source_longitude=float(event_lon),
                        source_depth_km=float(event_depth_km),
                        receiver_latitude=float(station_lat),
                        receiver_longitude=float(station_lon),
                        receiver_elevation_m=float(station_elev_m),
                        phase=config.phases[phase_index],
                        max_distance_km=max_distance_km,
                    )
                )
                if travel_time is None or not np.isfinite(travel_time):
                    continue
                data.theoretical_travel_times[event_index, obs_col] = travel_time
                valid_rows[event_row0 + obs_col] = True

        return np.flatnonzero(valid_rows)

    def refresh_ttt_batch(
        self,
        data: DataHandler,
        config: LocatorConfig,
        event_indices: np.ndarray,
    ) -> np.ndarray:
        """Refresh predicted travel times with ``get_arrival_batch``."""
        observed = ~np.isnan(data.arrivals)
        valid_rows = np.zeros(data.jacobian_matrix.shape[0], dtype=bool)
        n_obs = data.n_stations * config.n_phases

        station_lats = np.asarray(data.stations_locations[:, 0], dtype=np.float64)
        station_lons = np.asarray(data.stations_locations[:, 1], dtype=np.float64)
        station_elevs = np.asarray(data.stations_locations[:, 2], dtype=np.float64)

        data.theoretical_travel_times[event_indices] = np.nan

        for event_index in event_indices:
            event_index = int(event_index)
            if data.invalid_events[event_index]:
                continue

            event_lat, event_lon, event_depth_km = data.locations[event_index]
            event_row0 = event_index * n_obs

            for phase_index, phase in enumerate(config.phases):
                obs_cols = np.arange(
                    phase_index,
                    n_obs,
                    config.n_phases,
                    dtype=np.int64,
                )
                observed_mask = observed[event_index, obs_cols]
                if not observed_mask.any():
                    continue

                station_indices = np.flatnonzero(observed_mask)
                _distances, travel_times, _takeoffs, _azimuths = (
                    self.propagation_engine.get_arrival_batch(
                        source_latitude=float(event_lat),
                        source_longitude=float(event_lon),
                        source_depth_km=float(event_depth_km),
                        receiver_latitude=station_lats[station_indices],
                        receiver_longitude=station_lons[station_indices],
                        receiver_elevation_m=station_elevs[station_indices],
                        phase=phase,
                    )
                )

                finite = np.isfinite(travel_times)
                if not finite.any():
                    continue

                active_obs_cols = obs_cols[station_indices[finite]]
                data.theoretical_travel_times[event_index, active_obs_cols] = (
                    travel_times[finite]
                )
                valid_rows[event_row0 + active_obs_cols] = True

        return np.flatnonzero(valid_rows)

    def build_derivative_matrix(
        self,
        data: DataHandler,
        config: LocatorConfig,
        run_config: RunConfig,
        event_index: int | None = None,
    ) -> np.ndarray:
        """Fill the Jacobian and theoretical travel times for one or all events."""
        n_obs = data.n_stations * config.n_phases
        w_init = 0.0 if run_config.distance_weighting_enabled else 1.0
        if event_index is None:
            event_indices = np.arange(data.n_events, dtype=np.int64)
            data.jacobian_matrix.fill(0.0)
            data.theoretical_travel_times.fill(0.0)
            data.distance_W.fill(w_init)
        else:
            event_indices = np.array([event_index], dtype=np.int64)
            r0, r1 = event_index * n_obs, (event_index + 1) * n_obs
            data.jacobian_matrix[r0:r1].fill(0.0)
            data.theoretical_travel_times[event_index].fill(0.0)
            data.distance_W[r0:r1].fill(w_init)

        if self.propagation_engine.supports_feature("derivative.batch"):
            valid_rows = self._build_table_batch(
                data, config, run_config, event_indices
            )
        else:
            valid_rows = self._build_table(data, config, run_config, event_indices)

        if valid_rows.size:
            obs_cols = valid_rows % n_obs
            station_indices = np.unique(obs_cols // config.n_phases)
            logger.debug(
                "Using %d stations (%d obs) in current derivative build.",
                station_indices.size,
                valid_rows.size,
            )

        return valid_rows

    def refresh_theoretical_travel_times(
        self,
        data: DataHandler,
        config: LocatorConfig,
        run_config: RunConfig,
        event_index: int | None = None,
    ) -> np.ndarray:
        """Refresh predicted travel times without rebuilding derivatives."""
        event_indices = (
            np.arange(data.n_events, dtype=np.int64)
            if event_index is None
            else np.array([event_index], dtype=np.int64)
        )
        if self.propagation_engine.supports_feature("derivative.batch"):
            return self.refresh_ttt_batch(data, config, event_indices)
        return self.refresh_ttt(data, config, event_indices)
