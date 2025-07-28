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


from typing import List, Tuple, Callable
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from .travel_time_tables import TravelTimeTables
from .utils.geo import gps2dist_azimuth_numba, compute_spatial_derivatives
from .utils.damping import update_damping_factor
from scipy.optimize import least_squares


class EventLocator:
    """
    Earthquake event locator using precomputed travel-time tables.
    """

    def __init__(
        self,
        model: TauPyModel,
        phases: List[str] = ["P"],
        max_depth: float = 20.0,
        max_distance: float = 100.0,
        depth_step: float = 1.0,
        distance_step: float = 1.0,
        time_error_floor: float = 0.1,
        min_num_phases: int = 4,
        min_num_stations: int = 2,
        damping_function: Callable = update_damping_factor,
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.
        """
        self.model = model
        self.model_name = self.model.model.s_mod.v_mod.model_name

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

        # Can be replaced with a more sophisticated damping function
        # Function gets called with current damping factor,
        # iteration number and current update in
        # (lat, lon, depth, time) [degree, degree, km, second] format
        self.damping_function = damping_function

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
            result = least_squares(residual_func, x0, method="lm", xtol=1e-6, ftol=1e-6)
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

            damping = self.damping_function(damping, it, dx)
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

    def _plot_locations(
        self,
        locations: np.ndarray,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        if lower_bounds is not None and upper_bounds is not None:
            ax.errorbar(
                locations[:, 0],
                locations[:, 1],
                xerr=[
                    locations[:, 0] - lower_bounds[:, 0],
                    upper_bounds[:, 0] - locations[:, 0],
                ],
                yerr=[
                    locations[:, 1] - lower_bounds[:, 1],
                    upper_bounds[:, 1] - locations[:, 1],
                ],
                fmt="o",
                capsize=5,
                label="Locations with Uncertainty",
            )
        else:
            ax.scatter(
                locations[:, 0], locations[:, 1], label="Locations", color="blue"
            )

        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_title("Event Locations")
        ax.grid(True)
        ax.legend()
        plt.show()

    def compute_location_pdf(
        self,
        samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        n_events: int,
        invalid_events: np.ndarray,
        percentile_bounds: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute marginal posterior PDFs and confidence intervals for each event
        using physical coordinate samples.

        Args:
            samples: tuple of (lat_samples, lon_samples, dep_samples, ot_samples), each of shape (n_samples, n_events)
            n_events: number of events.
            invalid_events: boolean array for invalid events.
            percentile_bounds: confidence bounds, e.g., (2.5, 97.5) for 95% interval.

        Returns:
            pdf_marginals: list of PDFs (1D histograms per dimension per event).
            conf_intervals: list of (lower, upper) per dimension per event in absolute coordinates.
        """
        lat_samples, lon_samples, dep_samples, ot_samples = samples
        pdf_marginals = []
        conf_intervals = []

        for i in range(n_events):
            if invalid_events[i]:
                pdf_marginals.append([None] * self.dim)
                conf_intervals.append([(np.nan, np.nan)] * self.dim)
                continue

            marginals = []
            intervals = []
            for d, coord_samples in enumerate(
                [
                    lat_samples[:, i],
                    lon_samples[:, i],
                    dep_samples[:, i],
                ]
            ):
                hist, bin_edges = np.histogram(coord_samples, bins=100, density=True)
                marginals.append((hist, bin_edges))

                lower = np.percentile(coord_samples, percentile_bounds[0])
                upper = np.percentile(coord_samples, percentile_bounds[1])
                intervals.append((lower, upper))

            pdf_marginals.append(marginals)
            conf_intervals.append(intervals)

        return pdf_marginals, conf_intervals

    def monte_carlo_pdf_solver(
        self,
        max_iterations: int = 500,
        error_coef_scaling: float = 1.0,
        solve_airquakes: bool = False,
        trial_depth: float = 5.0,
        max_depth: float = 20.0,
        max_depth_iterations: int = 5,
    ):
        """
        Monte Carlo-based probabilistic earthquake location that produces posterior PDFs.

        Returns:
            locations_mean, origin_times_mean, pdf_marginals, conf_intervals, location_errors
        """
        self.init_locations(trial_depth=trial_depth)

        dx_samples = []

        for _ in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()
            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = self.theoretical_arrivals - travel_times_obs
            residuals_flat = residuals.flatten()

            # Add Gaussian noise to observed arrivals using estimated pick stddev
            pick_errors = self.arrival_time_errors.flatten()
            pick_errors = np.where(
                np.isnan(pick_errors) | (pick_errors <= 0.0),
                self.time_error_floor,
                pick_errors,
            )
            perturbed_residuals = residuals_flat.copy()
            perturbed_residuals[valid_rows] += np.random.normal(
                0, pick_errors[valid_rows]
            )

            A = self.derivative_matrix[valid_rows, :]
            b = perturbed_residuals[valid_rows]

            result = least_squares(
                lambda x: A @ x - b, np.zeros(self.n_events * self.dim), method="lm"
            )
            dx_solution = result.x.reshape((self.n_events, self.dim))
            dx_samples.append(dx_solution)

        dx_samples = np.stack(
            dx_samples, axis=0
        )  # shape: (max_iterations, n_events, dim)

        # Convert dx_samples to physical coordinates and apply to original locations
        samples_lat = self.locations[:, 0][None, :] - kilometer2degrees(
            dx_samples[:, :, 0], self.model.model.radius_of_planet
        )
        samples_lon = self.locations[:, 1][None, :] - kilometer2degrees(
            dx_samples[:, :, 1], self.model.model.radius_of_planet
        )
        samples_dep = self.locations[:, 2][None, :] - dx_samples[:, :, 2]
        samples_ot = self.origin_times[None, :] - dx_samples[:, :, 3]

        locations_mean = np.stack(
            [
                np.nanmean(samples_lat, axis=0),
                np.nanmean(samples_lon, axis=0),
                np.nanmean(samples_dep, axis=0),
            ],
            axis=1,
        )

        origin_times_mean = np.nanmean(samples_ot, axis=0)

        pdf_marginals, conf_intervals = self.compute_location_pdf(
            (samples_lat, samples_lon, samples_dep, samples_ot),
            self.n_events,
            self.invalid_events,
        )

        self._plot_locations(
            locations_mean,
            lower_bounds=np.array(
                [[bound[0] for bound in event] for event in conf_intervals]
            ),
            upper_bounds=np.array(
                [[bound[1] for bound in event] for event in conf_intervals]
            ),
        )

        return locations_mean, origin_times_mean, pdf_marginals, conf_intervals
