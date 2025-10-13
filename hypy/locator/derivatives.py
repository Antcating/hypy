import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.geo import get_spatial_slowness_1d, gps2dist_azimuth_nb
from ..utils.weighting import distance_weight
from .types import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import LocatorState


class DerivativesMixin:
    def __init__(self, *args, **kwargs) -> None:
        # per-instance defaults
        self.derivative_matrix = np.empty((0, 0), dtype=float)
        self.theoretical_travel_times = np.empty((0, 0), dtype=float)
        self.time_error_W = np.empty((0,), dtype=float)
        self.distance_W = np.empty((0,), dtype=float)

        self.max_distance: float | None = None  # in km
        self._distance_weighting: bool = False
        self.k: float = 1.0  # distance weighting parameter
        self.x0: float = 1.0  # distance weighting parameter

        super().__init__(*args, **kwargs)  # cooperative MRO

    def _derive_travel_time_param(
        self: "LocatorState",
        event_location: tuple[float, float, float],  # (lat, lon, depth_km)
        station_location: tuple[float, float, float],  # (lat, lon, elev_m)
        phase: str,
    ) -> tuple[
        float,  # distance_km
        float | None,  # travel_time (s) or None if out of range
        tuple[
            float | None, float | None, float | None, float | None
        ],  # (dt_dr, dt_dx, dt_dy, dt_dz)
    ]:
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
            If the path exceeds self.max_distance, travel_time and derivatives are None.
        """
        ev_lat, ev_lon, ev_depth_km = event_location
        st_lat, st_lon, st_elev_m = station_location

        # Spherical distance & azimuth (meters, degrees)
        dist_m, azimuth_deg, _ = gps2dist_azimuth_nb(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0

        # Check max distance
        # This check is nessesary when using Tables with limited grid coverage
        # Also, you may want to limit the distance range for Ray tracing too to
        # prevent runaway locations.
        if self.max_distance is not None and distance_km > self.max_distance:
            logger.info(
                "Station at distance %.2f km exceeds max_distance %.2f km. Skipping.",
                distance_km,
                self.max_distance,
            )
            return distance_km, None, (None, None, None, None)

        # Travel-time and takeoff angle from tracer
        source_depth_km = (
            ev_depth_km + st_elev_m / 1000.0
        )  # account for station elevation
        travel_time_s, takeoff_angle_deg = self.tracer.get_arrival(
            source_depth=source_depth_km, distance=distance_km, phase=phase
        )

        # Local velocity and partial derivatives
        velocity_km_s = self._get_velocity_1d(ev_depth_km, phase)
        takeoff_rad = np.deg2rad(takeoff_angle_deg)

        # dt/dr (ray), dt/dx (east), dt/dy (north), dt/dz (down)
        dt_dr, dt_dx, dt_dy, dt_dz = get_spatial_slowness_1d(
            takeoff_rad, azimuth_deg, velocity_km_s
        )

        return distance_km, travel_time_s, (dt_dr, dt_dx, dt_dy, dt_dz)

    def _populate_derivative_matrix(self: "LocatorState") -> np.ndarray:
        """
        Fill the design (derivative) matrix rows corresponding to all valid
        event-station-phase observations, and compute theoretical arrivals.

        Returns
        -------
        np.ndarray
            1D array of row indices (into the global design matrix) that were populated.
        """
        valid_row_indices: list[int] = []

        for event_index in range(self.n_events):
            # Skip invalid events
            if self.invalid_events[event_index]:
                continue

            ev_lat, ev_lon, ev_depth_km = self.locations[event_index]

            for station_index in range(self.n_stations):
                st_lat, st_lon, st_elev_m = self.stations_locations[station_index]

                for phase_index, phase_name in enumerate(self.phases):
                    # Column block start for this event in the global design matrix
                    col0 = event_index * self.dim

                    # Column within the event’s observation vector for this station–phase
                    obs_col = self.n_phases * station_index + phase_index

                    # If no observed pick, skip row
                    observed_time_s = self.arrivals[event_index, obs_col]
                    if np.isnan(observed_time_s):
                        continue

                    # Compute travel time and spatial derivatives
                    distance_km, phase_travel_time, (_, dt_dx, dt_dy, dt_dz) = (
                        self._derive_travel_time_param(
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
                            self.stations[station_index],
                            phase_name,
                        )
                        continue

                    # Store theoretical travel time
                    self.theoretical_travel_times[event_index, obs_col] = (
                        phase_travel_time
                    )

                    # Row index in the global design matrix for this observation
                    row_index = (
                        event_index * (self.n_stations * self.n_phases) + obs_col
                    )

                    # Optional distance-based weighting
                    if self._distance_weighting:
                        self.distance_W[row_index] = distance_weight(
                            distance_km, k=self.k, x0=self.x0
                        )

                    # Fill design matrix:
                    # follow original convention: [y, x, z, t0] ordering in columns
                    self.derivative_matrix[row_index, col0 + 0] = dt_dy or 0.0
                    self.derivative_matrix[row_index, col0 + 1] = dt_dx or 0.0
                    if not self.fixed_depth_mask[event_index]:
                        self.derivative_matrix[row_index, col0 + 2] = dt_dz or 0.0
                    self.derivative_matrix[row_index, col0 + 3] = 1.0  # dt/dt

                    valid_row_indices.append(row_index)

        return np.array(valid_row_indices, dtype=int)
