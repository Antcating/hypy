"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""

import numpy as np
from obspy.taup import TauPyModel
from obspy.core.event import Event
from scipy.sparse.linalg import lsqr


from typing import List, Tuple
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from .travel_time_tables import TravelTimeTables
from .utils.geo import gps2dist_azimuth_numba, compute_spatial_derivatives


class EventLocator:
    """
    Earthquake event locator using precomputed travel-time tables.
    """

    def __init__(
        self,
        model: str,
        dim: int = 3,
        phases: List[str] = ["P"],
        max_depth: float = 20.0,
        max_distance: float = 100.0,
        depth_step: float = 1.0,
        distance_step: float = 1.0,
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.
        """
        self.model = TauPyModel(model)
        self.model_name = model
        self.dim = dim
        self.phases = phases
        self.n_phases = len(phases)
        self.tables = TravelTimeTables(
            model, phases, max_depth, max_distance, depth_step, distance_step
        )
        self.inv: Inventory
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: List[Tuple[float, float, float]] = []
        self.n_stations: int = 0
        self.events: List[Event] = []
        self.n_events: int = 0
        self.origin_times: np.ndarray
        self.arrivals: np.ndarray
        self.theoretical_arrivals: np.ndarray
        self.derivative_matrix: np.ndarray
        self.locations: np.ndarray
        # Precompute velocity profiles for each phase
        self._precompute_velocity_profiles(max_depth, depth_step)

    def _precompute_velocity_profiles(self, max_depth, depth_step):
        """
        Precompute and store velocity as a function of depth for each phase.
        """
        self.velocity_profiles = {}
        depths = np.arange(0, max_depth + depth_step, depth_step)
        for phase in self.phases:
            phase_lower = phase.lower()
            v_profile = []
            for d in depths:
                try:
                    v = self.model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
                except Exception:
                    v = np.nan
                v_profile.append(v)
            self.velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    def _get_velocity(self, depth, phase):
        """
        Interpolate velocity for a given depth and phase from precomputed profile.
        """
        phase_lower = phase.lower()
        depths, v_profile = self.velocity_profiles[phase_lower]
        return np.interp(depth, depths, v_profile)

    def load_data(self, inv: Inventory, events: List[Event]) -> None:
        """
        Load station inventory (station metadata) and events (with picks) into the locator.
        This must be called before running locate().
        """
        self.inv = inv
        self.networks = [net.code for net in inv]
        self.stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        self.stations_locations = [
            (sta.latitude, sta.longitude, sta.elevation)
            for net in inv
            for sta in net.stations
        ]
        self.n_stations = len(self.stations)
        self.events = events
        self.n_events = len(events)
        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self._fill_data()

    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int) -> None:
        """
        Allocate arrays for origin times, observed arrivals, theoretical arrivals, derivative matrix, and locations.
        """
        self.origin_times = np.zeros(n_events)
        self.arrivals = np.full((n_events, n_phases * n_stations), np.nan)
        self.theoretical_arrivals = np.zeros((n_events, n_phases * n_stations))
        self.derivative_matrix = np.zeros(
            (n_events * n_phases * n_stations, n_events * self.dim)
        )
        self.locations = np.zeros((n_events, self.dim))

    def _fill_data(self) -> None:
        """
        Populate the arrivals matrix with observed pick times from each event.
        """
        for event_idx, event in enumerate(self.events):
            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    pick = next(
                        (
                            pk
                            for pk in event.picks
                            if pk.waveform_id.station_code == station
                            and pk.waveform_id.network_code == network
                            and pk.phase_hint.lower() == phase.lower()
                        ),
                        None,
                    )
                    arrival_time = np.nan if pick is None else pick.time
                    self.arrivals[
                        event_idx, self.n_phases * station_idx + phase_idx
                    ] = arrival_time

    def _interpolate_travel_time(
        self, phase: str, depth: float, distance: float
    ) -> float:
        return self.tables.interpolate_travel_time(phase, depth, distance)

    def _interpolate_takeoff_angle(
        self, phase: str, depth: float, distance: float
    ) -> float:
        return self.tables.interpolate_takeoff_angle(phase, depth, distance)

    def _interpolate_ray_param(
        self, phase: str, depth: float, distance: float
    ) -> float:
        return self.tables.interpolate_ray_param(phase, depth, distance)

    def _derive_travel_time_param(self, event_location, station_location, phase):
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location
        dist_m, az_deg, _ = gps2dist_azimuth_numba(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0
        travel_time = self._interpolate_travel_time(
            phase, ev_depth + st_elev / 1000, distance_km
        )
        takeoff_angle = self._interpolate_takeoff_angle(
            phase, ev_depth + st_elev / 1000, distance_km
        )
        vel = self._get_velocity(ev_depth, phase)
        takeoff_rad = np.deg2rad(takeoff_angle)
        dt_dr, dt_dz, dt_dx, dt_dy = compute_spatial_derivatives(
            takeoff_rad, az_deg, vel
        )

        return travel_time, dt_dx, dt_dy, dt_dz

    def init_locations(
        self, trial_depth: float = 10.0, trial_time: float = 2.0
    ) -> np.ndarray:
        """
        Initialize the event locations and origin times for iterative location.
        """
        for event_idx, event in enumerate(self.events):
            if len(event.picks) == 0:
                continue
            first_pick = min(event.picks, key=lambda pk: pk.time)
            net_code = first_pick.waveform_id.network_code
            sta_code = first_pick.waveform_id.station_code
            station = self.inv.select(network=net_code, station=sta_code)[0][0]
            self.locations[event_idx] = (
                station.latitude,
                station.longitude,
                trial_depth,
            )
            self.origin_times[event_idx] = first_pick.time - trial_time
        return self.locations

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
                    travel_time, dt_dx, dt_dy, dt_dz = self._derive_travel_time_param(
                        (ev_lat, ev_lon, ev_depth),
                        (sta_lat, sta_lon, sta_elev),
                        phase,
                    )
                    self.theoretical_arrivals[event_idx, arr_index] = travel_time
                    row_idx = event_idx * (self.n_stations * self.n_phases) + arr_index
                    if self.dim >= 2:
                        self.derivative_matrix[row_idx, col_idx + 0] = dt_dx
                        self.derivative_matrix[row_idx, col_idx + 1] = dt_dy
                    if self.dim == 3:
                        self.derivative_matrix[row_idx, col_idx + 2] = dt_dz
                    valid_rows.append(row_idx)
        return np.array(valid_rows, dtype=int)

    def _demean_residuals_eventwise(self, residuals, valid_rows):
        factor = self.n_stations * self.n_phases
        event_indices = valid_rows // factor
        residual_mean = np.zeros(self.n_events)
        for event_idx in range(self.n_events):
            mask = event_indices == event_idx
            if np.any(mask):
                mean_residual_event = np.mean(residuals[mask])
                residual_mean[event_idx] = mean_residual_event
                residuals[mask] -= mean_residual_event
        self.origin_times -= residual_mean


    def locate(
            self, 
            max_iterations=10, 
            conv_threshold=1e-3,
            solve_airquakes=False, 
            max_depth=20.0, 
            max_depth_iterations=5,
            trial_depth=10.0
            ):
        
        self.init_locations(trial_depth=trial_depth)
        self.invalid_events = np.zeros(self.n_events, dtype=bool)
        prev_residual_norm = np.inf

        max_damping = 10
        damping = 1.0
        max_update = 0.0

        residual_norms = []

        for it in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()
            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = self.theoretical_arrivals - travel_times_obs
            residuals_flat = residuals.flatten()
            b = residuals_flat[valid_rows]
            A = self.derivative_matrix[valid_rows, :]
            if self.n_events > 0:
                self._demean_residuals_eventwise(residuals_flat, valid_rows)

            dx_solution, istop = lsqr(A, b, damp=damping, atol=1e-6, btol=1e-6, iter_lim=300)[:2]
            dx = dx_solution.reshape((self.n_events, self.dim))

            residual_vector = A @ dx_solution - b
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)

            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - (
                    dx[event_idx, 2] if self.dim == 3 else 0.0
                )
                if (not solve_airquakes and new_depth < 0.0) or (
                    it > max_depth_iterations and new_depth > max_depth
                ):
                    print(
                        f"Event {event_idx} would have depth {new_depth:.2f} km out of bounds. Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    if self.dim == 3:
                        dx[event_idx, 2] = 0.0

            if self.dim >= 2:
                dx[:, 0] = kilometer2degrees(dx[:, 0])
                dx[:, 1] = kilometer2degrees(dx[:, 1])

            # --- Update locations ---
            self.locations -= dx

            if prev_residual_norm == 0 or prev_residual_norm is np.inf:
                prev_residual_norm = res_norm
                continue

            if it % 10 == 0:
                damping *= 0.9
            damping = np.clip(damping, 0, max_damping)
            prev_residual_norm = res_norm


            # --- Early stop if gradient over last 10 residuals is positive ---
            if len(residual_norms) >= 10:
                y = np.array(residual_norms[-10:])
                x = np.arange(it - 9, it + 1)
                coeffs = np.polyfit(x, y, 1)
                grad = coeffs[0]
                if grad > 0:
                    print(f"Stopping early at iteration {it + 1} due to positive residual gradient over last 10 iterations.")
                    break

            # --- Stop if residual norm is below threshold ---
            if res_norm < conv_threshold:
                print(f"Converged at iteration {it + 1} with residual norm {res_norm:.6f}.")
                break

        return self.locations, self.origin_times, max_update


    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
