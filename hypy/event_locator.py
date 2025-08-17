"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""

from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from obspy.core.event import Catalog, Event, Origin
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from scipy.sparse.linalg import lsmr, lsqr

from .propagation.ray_tracers import RayTracer
from .propagation.tables import TravelTimeTables
from .utils.geo import compute_spatial_derivatives, gps2dist_azimuth_numba


class EventLocator:
    """
    Earthquake event locator using precomputed travel-time tables.
    """

    def __init__(
        self,
        model_path: str,
        phases: List[str] = ["P"],
        max_depth: float = 30.0,
        max_distance: float = 100.0,
        depth_step: float = 1.0,
        distance_step: float = 1.0,
        time_error_floor: float = 0.1,
        min_picks_per_event: int = 4,
        mode: str = "ray",
        engine: str = "taup",
        solver: str = "lsqr",
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        self.model = TauPyModel(self.model_name)

        # The dimesion is 4 for (x, y, z, t) and is hardcoded to 4 for now.
        # Any other dimension will require changes in the code and is not supported.
        self.dim = 4

        self.phases = phases
        self.n_phases = len(phases)
        self.min_picks_per_event = min_picks_per_event


        if mode == "table":
            self.tracer = TravelTimeTables(
                self.model_name,
                phases,
                max_depth,
                max_distance,
                depth_step,
                distance_step,
            )
        elif mode == "ray":
            if engine not in ["taup", "pyrocko"]:
                raise ValueError(
                    f"Unsupported engine: {engine}. Supported engines for ray mode are 'taup' and 'pyrocko'."
                )
            self.tracer = RayTracer(model_path=self.model_path.as_posix(), engine=engine)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'table' and 'ray'.")
        self.mode = mode
        if solver not in ["lsqr", "svd"]:
            raise ValueError(f"Unsupported solver: {solver}. Supported solvers are 'lsqr' and 'svd'.")
        self.solver = solver
        
        self.inv: Inventory
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: List[Tuple[float, float, float]] = []
        self.n_stations: int = 0

        # TODO: Replace the list of events with proper Catalog handling
        self.events: Union[List[Event], Catalog] = []
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

        # Storage for error products
        self._covariances = None  # (n_events, 4, 4) full parameter covariance (x,y,z,t)
        self._spatial_covariances = None  # (n_events, 3, 3)
        self._principal_std = None  # (n_events, 3)
        self._principal_axes = None  # (n_events, 3, 3) eigenvectors as rows (ex,ey,ez)
        self._erz_erh = None  # (n_events, 2) [ERZ, ERH] @ 1σ

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

    def load_data(self, inv: Inventory, events: Union[List[Event], Catalog]) -> None:
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
        if self.n_phases * self.n_stations < self.min_picks_per_event:
            raise ValueError(
                f"{self.n_stations} stations * {self.n_phases} phases < {self.min_picks_per_event} picks per event. "
                f"Not enough stations and phases to satisfy minimum picks per event."
            )
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
            if len(event.picks) < self.min_picks_per_event:
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
            if len(picked_stations) < self.min_picks_per_event:
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

    def _derive_travel_time_param(self, event_location, station_location, phase):
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location
        dist_m, az_deg, _ = gps2dist_azimuth_numba(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0
        travel_time, takeoff_angle = self.tracer.get_arrival(
            source_depth=ev_depth + st_elev / 1000, distance=distance_km, phase=phase
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

    def _compute_event_errors(self, A_e: np.ndarray, r_e: np.ndarray, pick_errs_e: np.ndarray, damping: float):
        """Compute covariance, principal ellipsoid, ERZ/ERH for a single event.

        Parameters
        ----------
        A_e : (m, 4) array
            Weighted design sub-matrix for this event (columns x,y,z,t in km/s & s).
        r_e : (m,) array
            Unweighted residuals in seconds for this event (theoretical - (obs - t0)).
        pick_errs_e : (m,) array
            Arrival-time uncertainties (seconds) for this event, mapped to the same rows.
        damping : float
            Current Tikhonov damping (λ). Applied as λ^2 I to normal matrix.
        """
        if A_e.size == 0:
            return None

        # Normal matrix with Tikhonov regularization consistent with the inversion above
        N = A_e.T @ A_e
        N_reg = N + (damping ** 2) * np.eye(4)

        # Data variance estimate: include both RMS of residuals and robust pick error scale
        # Hypoinverse scales with RMS and RDERR. Use sigma^2 ≈ RMS^2 + median(pick_err)^2
        rms = float(np.sqrt(np.mean(r_e**2))) if r_e.size else 0.0
        rderr = float(np.nanmedian(pick_errs_e)) if pick_errs_e.size else 0.0
        sigma2 = rms**2 + rderr**2

        # Parameter covariance (approx.)
        try:
            N_reg_inv = np.linalg.inv(N_reg)
        except np.linalg.LinAlgError:
            N_reg_inv = np.linalg.pinv(N_reg)
        C = sigma2 * N_reg_inv  # (4x4)

        # Spatial 3x3 block (x,y,z), rotate to principal axes via eigen-decomposition
        Cxyz = C[:3, :3]
        evals, evecs = np.linalg.eigh(Cxyz)  # evals ascending
        order = np.argsort(evals)[::-1]  # descending: major, intermediate, minor
        evals = evals[order]
        evecs = evecs[:, order]

        # Principal standard errors are sqrt of eigenvalues (km)
        std_axes = np.sqrt(np.clip(evals, 0.0, None))  # (3,)

        # ERZ (vertical) and ERH (horizontal) from projections of principal axes
        # Convention: evec components correspond to (x=East, y=North, z=Down)
        ex, ey, ez = evecs[0, :], evecs[1, :], evecs[2, :]
        # For each principal axis i, projection lengths:
        proj_vert = np.abs(ez) * std_axes
        proj_horiz = np.sqrt(ex**2 + ey**2) * std_axes
        ERZ = float(np.max(proj_vert))
        ERH = float(np.max(proj_horiz))

        return {
            "C_full": C,
            "C_xyz": Cxyz,
            "principal_std_km_1sigma": std_axes,
            "principal_axes_unit_vectors": evecs,  # columns correspond to axes 1..3
            "ERZ_km_1sigma": ERZ,
            "ERH_km_1sigma": ERH,
            "RMS_s": rms,
            "RDERR_s": rderr,
        }

    def locate(
        self,
        max_iterations=10,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=20.0,
        max_depth_iterations=5,
        trial_depth=10.0,
    ):
        self.init_locations(trial_depth=trial_depth)
        self.invalid_events = np.zeros(self.n_events, dtype=bool)

        max_damping = 10
        damping = 1.0
        max_update = 0.0

        residual_norms = []
        A = None  # Initialize A to ensure it is always defined
        valid_rows = np.array([])  # Initialize valid_rows to ensure it is always defined
        residuals_flat = np.array([])  # Initialize residuals_flat to ensure it is always defined

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

            dx_solution = np.zeros(A.shape[1])  # Default initialization
            if self.solver == "lsqr":
                dx_solution, istop = lsmr(A, b, damp=damping, atol=1e-6, btol=1e-6)[:2]
            elif self.solver == "svd":
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                # Apply Tikhonov damping (ridge regularization)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping ** 2)
                dx_solution = Vt.T @ (s_inv * (U.T @ b))
            dx = dx_solution.reshape((self.n_events, self.dim))

            dx[:, 2] = np.clip(dx[:, 2], -2, 2)

            residual_vector = A @ dx_solution - b
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)

            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - dx[event_idx, 2]
                if not solve_airquakes and new_depth < 0.0:
                    print(
                        f"Event {event_idx} will become airquake with depth {new_depth:.2f} km. Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

                if (max_depth != np.nan) and (
                    it > max_depth_iterations and new_depth > max_depth
                ):
                    print(
                        f"Event {event_idx} exceeds max depth of {max_depth} km with depth {new_depth:.2f} km. Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)

            # Clip depth updates
            dx[:, 2] = np.clip(dx[:, 2], -2, 2)

            # --- Update locations ---
            valid_mask = ~self.invalid_events
            self.locations[valid_mask, :3] -= dx[valid_mask, :3]
            self.origin_times[valid_mask] -= dx[valid_mask, 3]

            damping *= 0.9
            damping = np.clip(damping, 0, max_damping)

            # --- Stop if residual norm is below threshold ---
            if res_norm < conv_threshold:
                print(
                    f"Converged at iteration {it + 1} with residual norm {res_norm:.6f}."
                )
                break

        self._A_final = A
        self._valid_rows_final = valid_rows
        self._residuals_flat_final = residuals_flat
        self._lambda_final = damping

        # ============================
        # Error ellipsoid computation
        # ============================
        self._covariances = np.zeros((self.n_events, 4, 4))
        self._spatial_covariances = np.zeros((self.n_events, 3, 3))
        self._principal_std = np.zeros((self.n_events, 3))
        self._principal_axes = np.zeros((self.n_events, 3, 3))
        self._erz_erh = np.zeros((self.n_events, 2))

        # Reuse final weighted A and residuals from the last iteration
        A_all = self._A_final
        valid_rows_all = np.asarray(self._valid_rows_final, dtype=int)
        residuals_flat = self._residuals_flat_final

        # Map from global rows to per-event rows
        rows_per_event = [[] for _ in range(self.n_events)]
        errs_per_event = [[] for _ in range(self.n_events)]
        for row in valid_rows_all:
            ev_idx = row // (self.n_stations * self.n_phases)
            rows_per_event[ev_idx].append(row)
            local_row = row - ev_idx * (self.n_stations * self.n_phases)
            errs_per_event[ev_idx].append(self.arrival_time_errors[ev_idx, local_row])

        # Compute per-event covariances and error summaries using the final A/r/λ
        for ev in range(self.n_events):
            if self.invalid_events[ev]:
                continue
            rows = np.array(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue
            # Extract this event's 4 columns from the final weighted A
            A_e = A_all[:, ev * self.dim : (ev + 1) * self.dim][rows]
            r_e = residuals_flat[rows]
            pick_errs_e = np.array(errs_per_event[ev], dtype=float)

            errinfo = self._compute_event_errors(A_e, r_e, pick_errs_e, damping=0.1)
            if errinfo is None:
                continue

            C_full = errinfo["C_full"]
            C_xyz = errinfo["C_xyz"]
            std_axes = errinfo["principal_std_km_1sigma"]
            evecs = errinfo["principal_axes_unit_vectors"]
            ERZ = errinfo["ERZ_km_1sigma"]
            ERH = errinfo["ERH_km_1sigma"]

            self._covariances[ev] = C_full
            self._spatial_covariances[ev] = C_xyz
            self._principal_std[ev] = std_axes
            self._principal_axes[ev] = evecs.T  # store as rows
            self._erz_erh[ev] = [ERZ, ERH]


        # Write origin times, locations, and attach error products
        for event_idx, event in enumerate(self.events):
            if self.invalid_events[event_idx]:
                print(f"Event {event_idx} is marked as invalid. Skipping update.")
                continue
            origin = Origin()
            origin.latitude = self.locations[event_idx, 0]
            origin.longitude = self.locations[event_idx, 1]
            origin.depth = self.locations[event_idx, 2] * 1000.0  # Convert to meters
            origin.time = self.origin_times[event_idx]

            # Package errors for consumers; include 95% (≈2.4×) scaling
            s1, s2, s3 = self._principal_std[event_idx]
            evecs = self._principal_axes[event_idx]  # rows
            ERZ_1s, ERH_1s = self._erz_erh[event_idx]
            scale95 = 2.4

            origin.extra = getattr(origin, "extra", {})
            origin.extra["errors"] = {
                "covariance_full_4x4": self._covariances[event_idx].tolist(),
                "covariance_xyz_3x3": self._spatial_covariances[event_idx].tolist(),
                "principal_std_km_1sigma": [float(s1), float(s2), float(s3)],
                "principal_std_km_95pct": [float(scale95*s1), float(scale95*s2), float(scale95*s3)],
                "principal_axes_unit_vectors_rows_ex_ey_ez": evecs.tolist(),
                "ERZ_km_1sigma": float(ERZ_1s),
                "ERZ_km_95pct": float(scale95 * ERZ_1s),
                "ERH_km_1sigma": float(ERH_1s),
                "ERH_km_95pct": float(scale95 * ERH_1s),
            }

            event.origins.append(origin)

        return self.events

    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
