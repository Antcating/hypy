from .types import np
from ..utils.geo import compute_spatial_derivatives_1d, gps2dist_azimuth_nb
from ..utils.weighting import distance_weight
from  ..logger import logger  # type: ignore[import-not-found]

class DerivativesMixin:
    def _derive_travel_time_param(self, event_location, station_location, phase):
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location
        dist_m, az_deg, _ = gps2dist_azimuth_nb(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0
        if self.max_distance is not None and distance_km > self.max_distance:
            logger.info(
                f"Station at distance {distance_km:.2f} km exceeds max_distance {self.max_distance} km. Skipping."
            )
            return distance_km, None, (None, None, None, None)

        travel_time, takeoff_angle = self.tracer.get_arrival(
            source_depth=ev_depth + st_elev / 1000, distance=distance_km, phase=phase
        )
        vel = self._get_velocity_1d(ev_depth, phase)
        takeoff_rad = np.deg2rad(takeoff_angle)
        (dt_dr, dt_dx, dt_dy, dt_dz) = compute_spatial_derivatives_1d(
            takeoff_rad, az_deg, vel
        )
        spatial_derivatives = (dt_dr, dt_dx, dt_dy, dt_dz)
        return distance_km, travel_time, spatial_derivatives

    def _populate_derivative_matrix(self):
        valid_rows = []
        for event_idx in range(self.n_events):
            if self.invalid_events[event_idx]:
                continue
            ev_lat, ev_lon, ev_depth = self.locations[event_idx]
            for station_idx in range(self.n_stations):
                sta_lat, sta_lon, sta_elev = self.stations_locations[station_idx]
                for phase_idx, phase in enumerate(self.phases):
                    col_idx = event_idx * self.dim
                    arr_index = self.n_phases * station_idx + phase_idx
                    observed_time = self.arrivals[event_idx, arr_index]
                    if np.isnan(observed_time):
                        continue

                    distance_km, travel_time, spatial_derivatives = (
                        self._derive_travel_time_param(
                            (ev_lat, ev_lon, ev_depth),
                            (sta_lat, sta_lon, sta_elev),
                            phase,
                        )
                    )
                    if travel_time is None:
                        logger.warning(
                            f"No TT for ev {event_idx}, st {self.stations[station_idx]}, phase {phase} → skip."
                        )
                        continue

                    dt_dr, dt_dx, dt_dy, dt_dz = spatial_derivatives
                    self.theoretical_arrivals[event_idx, arr_index] = travel_time
                    row_idx = event_idx * (self.n_stations * self.n_phases) + arr_index

                    if self._distance_weighting:
                        self.distance_W[row_idx] = distance_weight(
                            distance_km, k=self.k, x0=self.x0
                        )

                    # dt/dy, dt/dx follow original convention
                    self.derivative_matrix[row_idx, col_idx + 0] = dt_dy
                    self.derivative_matrix[row_idx, col_idx + 1] = dt_dx
                    if not self.fixed_depth_mask[event_idx]:
                        self.derivative_matrix[row_idx, col_idx + 2] = dt_dz
                    self.derivative_matrix[row_idx, col_idx + 3] = 1.0
                    valid_rows.append(row_idx)
        return np.array(valid_rows, dtype=int)
