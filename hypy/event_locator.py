"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from obspy.core.event import Arrival, Catalog, Event, Origin
from obspy.core.event.base import QuantityError
from obspy.core.event.origin import ConfidenceEllipsoid, OriginUncertainty
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from scipy.sparse.linalg import lsmr

from .propagation.ray_tracers import RayTracer
from .propagation.tables import TravelTimeTables
from .utils.damping import update_damping_factor
from .utils.geo import compute_spatial_derivatives_1d, gps2dist_azimuth_nb
from .utils.distance import compute_distance_weight
from .logger import get_logger

logger = get_logger(__name__)

class EventLocator:
    """
    Seismic event relocator using 1D velocity model and event P/S phase picks
    """

    def __init__(
        self,
        model_tvel: str,
        phases: List[str] = ["P"],
        min_picks_per_event: int = 4,
        time_error_floor: float = 0.1,
        mode: str = "ray",
        engine: str = "taup",
        max_depth: Optional[float] = None,
        max_distance: Optional[float] = None,
        depth_step: Optional[float] = None,
        distance_step: Optional[float] = None,
        
        distance_weight_start: Optional[float] = None,
        distance_weight_end: Optional[float] = None,
        weight_start: Optional[float] = 0.99,
        weight_end: Optional[float] = 0.01,

        _fix_depth: bool = False,
        _precalculate_velocities: bool = False,
    ):
        """
        Initialize the EventLocator

        :param model_tvel: Absolute path to velocity model .tvel file.
        :type model_tvel: str
        :param phases: Phases to solve for. Available options: 'P', 'S'.
        :type phases: list[str]
        :param min_picks_per_event: Number of picks required for event to be solved.
            Absolute minimum required is 4 (example: 4 P-phases per event).
        :type min_picks_per_event: int
        :param time_error_floor: Minimal time-pick error assumed for events with no
            time-pick error provided.
        :type time_error_floor: float
        :param mode: Set the method for direct propagation. Available options are:
            `table` and `ray`. \n
            `table` mode uses precomputed tables and interpolation for teorethical arrival
                time and takeoff angle. This mode is faster than alternatives,
                but the accuracycis highly dependent on resolution of the precalculated table.\n
            `ray` mode uses direct tau-p ray propagation algorithm to allow, in general,
                for higher accuracy than `table`, but at the cost of higher computational
                time
        :type mode: str
        :param engine: Set the engine for direct propagation. Available options are:
            `taup` and `pyrocko`.
        :type engine: str
        :param solver: Set the solver for inversion problem. Available options are:
            `lsqr` and `svd`.
        :type solver: str
        :param max_depth: Set bounding depth for precalculated table in km. Required
            if using `table` mode.
        :type max_depth: Optional[float]
        :param max_distance: Set bounding distance for precalculated table in km. Required
            if using `table` mode.
        :type max_distance: Optional[float]
        :param depth_step: Set depth resolution for precalculated table in km. Required
            if using `table` mode.
        :type depth_step: Optional[float]
        :param distance_step: Set distance resolution for precalculated table in km. Required
            if using `table` mode.
        :type distance_step: Optional[float]
        :param distance_cutoff: Set distance at distance weighting becomes significant.
            If `distance_cutoff` is not set, distance weighting is not performed.
        :type distance_cutoff: Optional[float]
        :param cutoff_width: Width of the distance weight transition from 1 to 0 around
            `distance_cutoff`. If `cutoff_width` is not set, distance weighting
            is not performed.
        :type cutoff_width: Optional[float]
        :param _fix_depth: If True, depth is fixed to initial guess FOR ALL EVENTS.
        :type _fix_depth: bool
        :param _precalculate_velocities: If True, velocities are precalculated for
            each phase at depth intervals defined by `depth_step` up to `max_depth`.
            This is only supported for 1D velocity models and is intended to speed up
            the location process.
        :type _precalculate_velocities: bool
        """
        self.tvel_file_path = Path(model_tvel)
        self.model_name = self.tvel_file_path.stem
        try:
            self.model = TauPyModel(self.model_name)
        except FileNotFoundError:
            build_taup_model(self.tvel_file_path.as_posix())
            self.model = TauPyModel(self.model_name)
        # The dimesion is 4 for (x, y, z, t) and is hardcoded to 4 for now.
        # Any other dimension will require changes in the code and is not supported.
        self.dim = 4

        self._fix_depth = _fix_depth

        self.phases = phases
        self.n_phases = len(phases)
        self.min_picks_per_event = min_picks_per_event

        if mode == "table":
            if any(
                [
                    x is None
                    for x in [max_depth, max_distance, depth_step, distance_step]
                ]
            ):
                raise ValueError(
                    "In table mode these variables are required: "
                    "max_depth, max_distance, depth_step, distance_step"
                )
            self.tracer = TravelTimeTables(
                self.tvel_file_path.as_posix(),
                self.model_name,
                phases,
                max_depth,
                max_distance,
                depth_step,
                distance_step,
                engine
            )
            self.max_distance = max_distance
            self.max_depth = max_depth
        elif mode == "ray":
            if engine not in ["taup", "pyrocko"]:
                raise ValueError(
                    f"Unsupported engine: {engine}. Supported engines for ray mode are 'taup' and 'pyrocko'."
                )
            self.tracer = RayTracer(
                model_path=self.tvel_file_path.as_posix(), engine=engine
            )

        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported modes are 'table' and 'ray'."
            )
        self.mode = mode

        self.inv: Inventory
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: List[Tuple[float, float, float]] = []
        self.n_stations: int = 0

        self.events: Union[List[Event], Catalog] = []
        self.n_events: int = 0
        self.origin_times: np.ndarray
        self.arrivals: np.ndarray
        self.theoretical_arrivals: np.ndarray
        self.derivative_matrix: np.ndarray
        self.locations: np.ndarray
        self.max_distance: Optional[float] = max_distance

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

        # Weights for distance-based damping
        self.distance_W: np.ndarray

        if any(
            [
                x is None
                for x in [
                    distance_weight_start,
                    distance_weight_end,
                    weight_start,
                    weight_end,
                ]
            ]
        ):
            self._distance_weighting = False
            self.k = None
            self.x0 = None
        else:
            self.k = (
                np.log((1.0 / weight_end) - 1.0) - np.log((1.0 / weight_start) - 1.0)
            ) / (distance_weight_end - distance_weight_start)
            self.x0 = distance_weight_start - (
                np.log((1.0 / weight_start) - 1.0) / self.k
            )
            self._distance_weighting = True

        # Storage for error products
        self._covariances = None  # (n_events, 4, 4) full parameter covariance (x,y,z,t)
        self._spatial_covariances = None  # (n_events, 3, 3)
        self._principal_std = None  # (n_events, 3)
        self._principal_axes = None  # (n_events, 3, 3) eigenvectors as rows (ex,ey,ez)
        self._erz_erh = None  # (n_events, 2) [ERZ, ERH] @ 1σ

        self._precalculate_velocities = _precalculate_velocities
        if self._precalculate_velocities and any(
            [x is None for x in [max_depth, depth_step]]
        ):
            raise ValueError(
                "When using precalculated velocities for each depth, "
                "max_depth and depth_step must be provided."
            )
        if self._precalculate_velocities:
            # Currently only 1D velocity models are supported
            self.velocity_profile_dimensionality = 1
            if self.velocity_profile_dimensionality == 1:
                # Precompute velocity profiles for each phase
                self._precompute_velocity_profiles_1d(max_depth, depth_step)

    def _precompute_velocity_profiles_1d(self, max_depth, depth_step):
        """
        Precompute and store velocity as a function of depth for each phase.
        """
        self.velocity_profiles = {}
        depths = np.arange(0, max_depth + depth_step, depth_step)
        for phase in self.phases:
            phase_lower = phase.lower()
            v_profile = []
            for d in depths:
                if self.model.model.s_mod is None:
                    raise ValueError(
                        "The taup velocity model's s_mod attribute is not initialized."
                    )
                v = self.model.model.s_mod.v_mod.evaluate_below(d, phase_lower)[0]
                v_profile.append(v)
            self.velocity_profiles[phase_lower] = (depths, np.array(v_profile))

    def _get_velocity_1d(self, depth, phase):
        """
        Get velocity at a given depth for a specific phase.
        """
        phase_lower = phase.lower()
        if not self._precalculate_velocities:
            return self.model.model.s_mod.v_mod.evaluate_below(depth, phase)[0]
        depths, v_profile = self.velocity_profiles[phase_lower]
        return np.interp(depth, depths, v_profile)

    def load_data(self, inv: Inventory, events: Union[List[Event], Catalog]) -> None:
        """
        Load station inventory (station metadata) and events (with picks) into the locator.
        This must be called before running locate().
        """
        
        # Read inventory data
        self.inv = inv
        self.networks = [net.code for net in inv]
        all_stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        all_stations_locations = [
            (sta.latitude, sta.longitude, sta.elevation)
            for net in inv
            for sta in net.stations
        ]

        
        # Read events and picks
        self.events = events
        self.n_events = len(events)

        # Extract stations from picks in events
        ev_stations = list(
            set(
                [
                    (pk.waveform_id.network_code, pk.waveform_id.station_code)
                    for ev in events
                    for pk in ev.picks
                ]
            )
        )

        # Check for stations in picks that are not in inventory
        missing_stations = [sta for sta in ev_stations if sta not in all_stations]
        if len(missing_stations) > 0:
            logger.warning(
                f"The following stations are present in event picks but missing in inventory: {missing_stations}"
            )

        # Mask stations to only those present in both inventory and picks
        stations = list(set(all_stations) & set(ev_stations))

        # Load stations data
        self.stations = stations
        self.n_stations = len(self.stations)
        self.stations_locations = [
            all_stations_locations[all_stations.index(sta)] for sta in stations
        ]

        # Validate data
        if self.n_phases * self.n_stations < self.min_picks_per_event:
            raise ValueError(
                f"{self.n_stations} stations * {self.n_phases} phases < {self.min_picks_per_event} picks per event. "
                f"Not enough stations and phases to satisfy minimum picks per event."
            )
        
        # Create fixed depth mask for events
        self.fixed_depth_mask = [
            self._fix_depth
            or (
                len(ev.origins) > 0 and ev.origins[-1].depth_type == "operator assigned"
            )
            for ev in events
        ]
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
        self.distance_W = np.ones(n_events * n_phases * n_stations)
        self.locations = np.zeros((n_events, self.dim - 1))
        self.invalid_events = np.zeros(n_events, dtype=bool)

    def _fill_data(self) -> None:
        """
        Populate the arrivals matrix with observed pick times from each event.
        """
        for event_idx, event in enumerate(self.events):
            if len(event.picks) < self.min_picks_per_event:
                logger.warning(
                    f"Event {event.resource_id} has only {len(event.picks)} picks, skipping."
                )
                self.invalid_events[event_idx] = True
                continue
            # Count unique stations with picks for this event
            picked_stations = set(
                (pk.waveform_id.network_code, pk.waveform_id.station_code)
                for pk in event.picks
            )
            if len(picked_stations) * self.n_phases < self.min_picks_per_event:
                logger.warning(
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
                        logger.warning(
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
        dist_m, az_deg, _ = gps2dist_azimuth_nb(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0
        if self.max_distance is not None and distance_km > self.max_distance:
            return distance_km, None, None, None, None
        travel_time, takeoff_angle = self.tracer.get_arrival(
            source_depth=ev_depth + st_elev / 1000, distance=distance_km, phase=phase
        )
        vel = self._get_velocity_1d(ev_depth, phase)
        takeoff_rad = np.deg2rad(takeoff_angle)
        dt_dr, dt_dz, dt_dx, dt_dy = compute_spatial_derivatives_1d(
            takeoff_rad, az_deg, vel
        )

        return distance_km, travel_time, dt_dx, dt_dy, dt_dz

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
                    distance_km, travel_time, dt_dx, dt_dy, dt_dz = (
                        self._derive_travel_time_param(
                            (ev_lat, ev_lon, ev_depth),
                            (sta_lat, sta_lon, sta_elev),
                            phase,
                        )
                    )
                    if travel_time is None:
                        logger.warning(
                            f"No travel time found for event {event_idx}, station {self.stations[station_idx]}, phase {phase}. Skipping this arrival."
                        )
                        continue

                    self.theoretical_arrivals[event_idx, arr_index] = travel_time
                    row_idx = event_idx * (self.n_stations * self.n_phases) + arr_index

                    if self._distance_weighting:
                        # Weighting for distance-based damping
                        self.distance_W[row_idx] = compute_distance_weight(
                            distance_km, k=self.k, x0=self.x0
                        )

                    # Populate the derivative matrix
                    self.derivative_matrix[row_idx, col_idx + 0] = dt_dx
                    self.derivative_matrix[row_idx, col_idx + 1] = dt_dy
                    # if not self.fix_depth:
                    if not self.fixed_depth_mask[event_idx]:
                        self.derivative_matrix[row_idx, col_idx + 2] = dt_dz
                    self.derivative_matrix[row_idx, col_idx + 3] = 1.0
                    valid_rows.append(row_idx)
        return np.array(valid_rows, dtype=int)

    def _compute_event_errors(
        self, A_e: np.ndarray, r_e: np.ndarray, pick_errs_e: np.ndarray, damping: float
    ):
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
        N_reg = N + (damping**2) * np.eye(4)

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
        max_iterations=1000,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=20.0,
        max_depth_iterations=5,
        trial_depth=10.0,
        solver: str = "lsqr",
    ):
        if solver not in ["lsqr", "svd"]:
            raise ValueError(
                f"Unsupported solver: {solver}. Supported solvers are 'lsqr' and 'svd'."
            )
        self.solver = solver

        self.init_locations(trial_depth=trial_depth)
        self._run_location_iterations(
            max_iterations,
            conv_threshold,
            solve_airquakes,
            max_depth,
            max_depth_iterations,
        )
        self._compute_error_ellipsoids()
        return self._finalize_origins()

    def _run_location_iterations(
        self,
        max_iterations,
        conv_threshold,
        solve_airquakes,
        max_depth,
        max_depth_iterations,
    ):
        logger.info(f"Locating {self.n_events} events with {self.n_stations} stations.")
        logger.info(f"Using phases: {self.phases}")
        logger.info(f"Using {self.solver} solver.")
        logger.info(f"Using {self.mode} mode for travel-time calculations.")
        logger.info(f"Distance weighting: {self._distance_weighting}")
        logger.info(f"Fix depth: {self._fix_depth}")
        logger.info(f"Precalculate velocities: {self._precalculate_velocities}")

        damping = 1.0
        max_damping = 10
        A = None
        residual_norms = []
        residuals_flat = np.array([])
        valid_rows = np.array([], dtype=int)
        for it in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()
            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = self.theoretical_arrivals - travel_times_obs
            residuals_flat = residuals.flatten()
            b = (
                self.distance_W[valid_rows]
                * self.W[valid_rows]
                * residuals_flat[valid_rows]
            )
            A = (
                self.derivative_matrix[valid_rows, :]
                * self.W[valid_rows][:, np.newaxis]
                * self.distance_W[valid_rows][:, np.newaxis]
            )

            dx_solution = np.zeros(A.shape[1])
            if self.solver == "lsqr":
                dx_solution = lsmr(A, b, damp=damping, atol=1e-6, btol=1e-6)[0]
            elif self.solver == "svd":
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
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
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0
                elif it > max_depth_iterations and new_depth > max_depth:
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)
            valid_mask = ~self.invalid_events
            self.locations[valid_mask, :3] -= dx[valid_mask, :3]
            self.origin_times[valid_mask] -= dx[valid_mask, 3]
            damping = np.clip(update_damping_factor(damping), 0, max_damping)
            if res_norm < conv_threshold:
                break

        self._A_final = A
        self._valid_rows_final = valid_rows
        self._residuals_flat_final = residuals_flat
        self._lambda_final = damping

    def _compute_error_ellipsoids(self):
        self._covariances = np.zeros((self.n_events, 4, 4))
        self._spatial_covariances = np.zeros((self.n_events, 3, 3))
        self._principal_std = np.zeros((self.n_events, 3))
        self._principal_axes = np.zeros((self.n_events, 3, 3))
        self._erz_erh = np.zeros((self.n_events, 2))

        A_all = self._A_final
        valid_rows_all = np.asarray(self._valid_rows_final, dtype=int)
        residuals_flat = self._residuals_flat_final
        rows_per_event = [[] for _ in range(self.n_events)]
        errs_per_event = [[] for _ in range(self.n_events)]

        for row in valid_rows_all:
            ev_idx = row // (self.n_stations * self.n_phases)
            rows_per_event[ev_idx].append(row)
            local_row = row - ev_idx * (self.n_stations * self.n_phases)
            errs_per_event[ev_idx].append(self.arrival_time_errors[ev_idx, local_row])

        for ev in range(self.n_events):
            if self.invalid_events[ev]:
                continue
            rows = np.array(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue
            # A_e = A_all[:, ev * self.dim : (ev + 1) * self.dim][rows]
            A_e = A_all
            r_e = residuals_flat[rows]
            pick_errs_e = np.array(errs_per_event[ev], dtype=float)
            errinfo = self._compute_event_errors(A_e, r_e, pick_errs_e, damping=0.1)
            if errinfo is None:
                continue
            self._covariances[ev] = errinfo["C_full"]
            self._spatial_covariances[ev] = errinfo["C_xyz"]
            self._principal_std[ev] = errinfo["principal_std_km_1sigma"]
            self._principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
            self._erz_erh[ev] = [errinfo["ERZ_km_1sigma"], errinfo["ERH_km_1sigma"]]

    def _finalize_origins(self):
        scale95 = 2.4
        for event_idx, event in enumerate(self.events):
            if self.invalid_events[event_idx]:
                continue
            origin = Origin()
            origin.latitude = self.locations[event_idx, 0]
            origin.longitude = self.locations[event_idx, 1]
            origin.depth = self.locations[event_idx, 2] * 1000.0
            origin.time = self.origin_times[event_idx]
            s1, s2, s3 = self._principal_std[event_idx]
            evecs = self._principal_axes[event_idx]
            ERZ_1s, ERH_1s = self._erz_erh[event_idx]

            v = evecs[:, 0]
            v = v / np.linalg.norm(v)
            plunge = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
            azimuth = float((np.rad2deg(np.arctan2(v[0], v[1])) + 360) % 360)

            confidence_ellipsoid = ConfidenceEllipsoid(
                semi_major_axis_length=s1 * 1000.0,
                semi_minor_axis_length=s3 * 1000.0,
                semi_intermediate_axis_length=s2 * 1000.0,
                major_axis_plunge=plunge,
                major_axis_azimuth=azimuth,
            )

            origin_uncertainty = OriginUncertainty(
                confidence_ellipsoid=confidence_ellipsoid,
                horizontal_uncertainty=ERH_1s * scale95 * 1000.0,
                preferred_description="confidence ellipsoid",
                confidence_level=95.0,
            )
            origin.origin_uncertainty = origin_uncertainty

            depth_errors = QuantityError(
                uncertainty=ERZ_1s * scale95 * 1000.0,
                confidence_level=95.0,
            )
            origin.depth_errors = depth_errors
            origin.depth_type = (
                "operator assigned"
                if self.fixed_depth_mask[event_idx]
                else "from location"
            )

            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    arr_index = self.n_phases * station_idx + phase_idx
                    observed_time = self.arrivals[event_idx, arr_index]
                    if np.isnan(observed_time):
                        continue
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
                    travel_time = self.theoretical_arrivals[event_idx, arr_index]
                    residual = travel_time - (
                        observed_time - self.origin_times[event_idx]
                    )
                    arrival = Arrival(
                        pick_id=pick.resource_id, phase=phase, time_residual=residual
                    )
                    origin.arrivals.append(arrival)
            event.origins.append(origin)
        return self.events

    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
