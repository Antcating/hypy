"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""

import numpy as np
from obspy.taup import TauPyModel
from obspy.core.event import Event
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt


from typing import List, Tuple
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from .travel_time_tables import TravelTimeTables
from .utils.geo import gps2dist_azimuth_numba, compute_spatial_derivatives
from scipy.optimize import least_squares


class EventLocator:
    """
    Earthquake event locator using precomputed travel-time tables.
    """

    def __init__(
        self,
        model: str,
        phases: List[str] = ["P"],
        max_depth: float = 20.0,
        max_distance: float = 100.0,
        depth_step: float = 1.0,
        distance_step: float = 1.0,
        time_error_floor: float = 0.1,
        min_num_phases: int = 4,
        min_num_stations: int = 2,
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.
        """
        self.model = TauPyModel(model)
        self.model_name = model

        # The dimesion is 4 for (x, y, z, t) and is hardcoded to 4 for now.
        # Any other dimension will require changes in the code and is not supported.
        self.dim = 4

        # Minimum number of phases and stations required for a valid location
        self.min_num_phases = min_num_phases
        self.min_num_stations = min_num_stations

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

        # TODO: Replace the list of events with proper Catalog handling
        self.events: List[Event] = []
        self.n_events: int = 0
        self.origin_times: np.ndarray
        self.arrivals: np.ndarray
        self.theoretical_arrivals: np.ndarray
        self.derivative_matrix: np.ndarray
        self.locations: np.ndarray

        # --- Error handling ---
        # If the time error is not provided or is zero, we use a default floor value.
        # If testing with synthetic data with zero time-error,
        # you *must* explicitly set time_error_floor to 0,
        # otherwise pick errors will be overridden to be time_error_floor.
        self.time_error_floor = time_error_floor
        # Array to store arrival time errors for each pick
        self.arrival_time_errors: np.ndarray
        # Weighting matrix W for LSQR
        self.W: np.ndarray

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
        self.arrival_time_errors = np.full((n_events, n_phases * n_stations), np.nan)
        self.theoretical_arrivals = np.zeros((n_events, n_phases * n_stations))
        self.derivative_matrix = np.zeros(
            (n_events * n_phases * n_stations, n_events * self.dim)
        )
        self.W = np.ones(n_events * n_phases * n_stations)
        self.locations = np.zeros((n_events, self.dim - 1))
        self.invalid_events = np.zeros(n_events, dtype=bool)

    def _fill_data(self) -> None:
        """
        Populate the arrivals matrix with observed pick times from each event.
        """
        for event_idx, event in enumerate(self.events):
            if len(event.picks) < self.min_num_phases:
                print(
                    f"Event {event.resource_id} has only {len(event.picks)} picks, skipping."
                )
                self.invalid_events[event_idx] = True
                continue
            # Count unique stations with picks for this event
            picked_stations = set(
                (pk.waveform_id.network_code, pk.waveform_id.station_code)
                for pk in event.picks
            )
            if len(picked_stations) < self.min_num_stations:
                print(
                    f"Event {event.resource_id} has picks at only {len(picked_stations)} unique stations, skipping."
                )
                self.invalid_events[event_idx] = True
                continue

            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    # Find the pick for this station and phase
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

                    # If no pick is found, continue to the next phase
                    if pick is None:
                        continue

                    # Store the arrival time and error
                    arrival_time = pick.time
                    arrival_time_error = (
                        np.nan
                        if pick.time_errors is None
                        else pick.time_errors.uncertainty
                    )

                    # Replace NaN or zero errors with the time error floor
                    if np.isnan(arrival_time_error) or arrival_time_error <= 0.0:
                        print(
                            f"Pick for {network}.{station} phase {phase} has invalid time error. Using floor value: {self.time_error_floor} s."
                        )
                        arrival_time_error = self.time_error_floor

                    self.arrivals[
                        event_idx, self.n_phases * station_idx + phase_idx
                    ] = arrival_time

                    self.arrival_time_errors[
                        event_idx, self.n_phases * station_idx + phase_idx
                    ] = arrival_time_error

                    self.W[
                        event_idx * self.n_phases * self.n_stations
                        + station_idx * self.n_phases
                        + phase_idx
                    ] = 1.0 / (
                        (arrival_time_error + 1) ** 2 if arrival_time_error > 0 else 1.0
                    )

    def _interpolate_travel_time(
        self, phase: str, depth: float, distance: float
    ) -> float:
        return self.tables.interpolate_travel_time(phase, depth, distance)

    def _interpolate_takeoff_angle(
        self, phase: str, depth: float, distance: float
    ) -> float:
        return self.tables.interpolate_takeoff_angle(phase, depth, distance)

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

                    # Populate the derivative matrix
                    self.derivative_matrix[row_idx, col_idx + 0] = dt_dx
                    self.derivative_matrix[row_idx, col_idx + 1] = dt_dy
                    self.derivative_matrix[row_idx, col_idx + 2] = dt_dz
                    self.derivative_matrix[row_idx, col_idx + 3] = 1.0
                    valid_rows.append(row_idx)
        return np.array(valid_rows, dtype=int)

    def compute_location_error(
        self,
        A_raw: np.ndarray,
        b_raw: np.ndarray,
        dx_solution: np.ndarray,
        n_events: int,
        valid_rows: np.ndarray,
        invalid_events: np.ndarray,
        error_coef_scaling: float = 1.0,
        error_floor: float = 0.1,
    ):
        """
        Compute covariance and per-event errors using per-pick arrival_time_errors instead of fixed RDERR.

        Returns:
            full_covariance_matrix (n_params x n_params)
            per_event_errors: [σ1, σ2, ..., σ_dim] per valid event, or None
        """
        A = A_raw[valid_rows, :]
        b = b_raw[valid_rows]
        residuals = A @ dx_solution - b

        n_params = A.shape[1]

        # -------------------------------
        # Get per-pick RDERR from self.arrival_time_errors
        # -------------------------------
        flat_errors = self.arrival_time_errors.flatten()
        pick_errors = flat_errors[valid_rows]

        # Replace missing or zero errors with a floor (e.g., 0.1 s)
        pick_errors = np.where(
            np.isnan(pick_errors) | (pick_errors <= 0.0), error_floor, pick_errors
        )

        # Total pick variance: (reading error)^2 + (ERCOF × residual)^2
        pick_var = pick_errors**2 + (error_coef_scaling * residuals) ** 2
        pick_std = np.sqrt(pick_var)

        # Weighted design matrix and residuals
        A_w = A / pick_std[:, np.newaxis]

        # SVD decomposition
        _, s, VT = np.linalg.svd(A_w, full_matrices=False)

        mean_sigma2 = np.mean(pick_var)
        full_covariance_matrix = np.zeros((n_params, n_params))
        per_event_errors = []

        for i in range(n_events):
            if invalid_events[i]:
                per_event_errors.append([np.nan] * self.dim)
                continue
            idx = slice(i * self.dim, (i + 1) * self.dim)
            V_event = VT.T[idx, :]
            S_inv_sq = np.diag(1.0 / s**2)
            cov_block = V_event @ S_inv_sq @ V_event.T * mean_sigma2
            full_covariance_matrix[idx, idx] = cov_block
            std_errors = np.sqrt(np.diag(cov_block)) * 2  # 95% CI
            per_event_errors.append(std_errors)

        return full_covariance_matrix, np.array(per_event_errors)

    def locate(
        self,
        max_iterations=10,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=20.0,
        max_depth_iterations=5,
        trial_depth=10.0,
        plot_residuals=False,
    ):
        self.init_locations(trial_depth=trial_depth)
        
        max_damping = 10
        damping = 1.0
        max_update = 0.0

        residual_norms = []

        for it in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()
            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = self.theoretical_arrivals - travel_times_obs
            residuals_flat = residuals.flatten()
            b = self.W[valid_rows] * residuals_flat[valid_rows]
            A = (
                self.derivative_matrix[valid_rows, :]
                * self.W[valid_rows][:, np.newaxis]
            )

            def residual_func(params):
                # params is a flat array of shape (n_events * self.dim,)
                return A @ params - b

            x0 = np.zeros(self.n_events * self.dim)
            result = least_squares(residual_func, x0, method='lm', xtol=1e-6, ftol=1e-6)
            dx_solution = result.x
            dx = dx_solution.reshape((self.n_events, self.dim))

            residual_vector = A @ dx_solution - b
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)

            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - dx[event_idx, 2]
                if (not solve_airquakes and new_depth < 0.0) or (
                    it > max_depth_iterations and new_depth > max_depth
                ):
                    print(
                        f"Event {event_idx} would have depth {new_depth:.2f} km out of bounds. Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, 2] = 0.0

            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)

            # --- Update locations ---
            self.locations[:, :3] -= dx[:, :3]
            self.origin_times -= dx[:, 3]

            damping *= 0.9
            damping = np.clip(damping, 0, max_damping)

            # --- Early stop if gradient over last 10 residuals is positive ---
            if len(residual_norms) >= 10:
                y = np.array(residual_norms[-10:])
                x = np.arange(it - 9, it + 1)
                coeffs = np.polyfit(x, y, 1)
                grad = coeffs[0]
                if grad > 0:
                    print(
                        f"Stopping early at iteration {it + 1} due to positive residual gradient over last 10 iterations."
                    )
                    break

            # --- Stop if residual norm is below threshold ---
            if res_norm < conv_threshold:
                print(
                    f"Converged at iteration {it + 1} with residual norm {res_norm:.6f}."
                )
                break

        cov, errors = self.compute_location_error(
            A_raw=self.derivative_matrix,
            b_raw=residuals_flat,
            dx_solution=dx_solution,
            n_events=self.n_events,
            valid_rows=valid_rows,
            invalid_events=self.invalid_events,
        )

        print(errors)

        if plot_residuals:
            plt.figure()
            plt.plot(
                residual_norms, marker="o", markersize=4, linestyle="-", linewidth=1
            )
            plt.xlabel("Iteration")
            plt.ylabel("Residual Norm")
            plt.title("Residual Norm over Iterations")
            plt.yscale("log")
            plt.grid(True)
            plt.show()

        return self.locations, self.origin_times, max_update

    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
