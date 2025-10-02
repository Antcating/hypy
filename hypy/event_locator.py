"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal, Sequence

import numpy as np
from obspy.core.event import Arrival, Catalog, Event, Origin
from obspy.core.event.base import QuantityError
from obspy.core.event.origin import ConfidenceEllipsoid, OriginUncertainty
from obspy.core.inventory import Inventory
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from scipy.sparse.linalg import lsmr, lsqr
from scipy.linalg import lstsq

from .propagation.ray_tracers import RayTracer
from .propagation.tables import TravelTimeTables
from .utils.damping import update_damping_factor
from .utils.geo import compute_spatial_derivatives_1d, gps2dist_azimuth_nb
from .utils.weighting import distance_weight, time_error_weight
from .logger import logger


class EventLocator:
    """
    Seismic event relocator using 1D velocity model and event P/S phase picks
    """

    def __init__(
        self,
        model_tvel: str,
        phases: Sequence[str] = ["P"],
        min_picks_per_event: int = 4,
        time_error_floor: float = 0.1,
        mode: Literal["table", "ray"] = "table",
        engine: Literal["taup", "pyrocko"] = "taup",
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
        Initialize the EventLocator.

        Parameters
        ----------
        model_tvel : str
            Absolute path to the velocity-model `.tvel` file (TauP).
        phases : Sequence[str], optional
            Phases to solve for (e.g., ("P",), ("P","S")). Default: ("P",).
        min_picks_per_event : int
            Minimum number of picks per event to attempt a solution (>=4).
        time_error_floor : float
            Fallback pick-time error (s) when a pick has no uncertainty.
        mode : {"table","ray"}
            Travel-time engine mode. "table" uses precomputed TT/takeoff tables
            with interpolation; "ray" uses direct ray tracing.
        engine : {"taup","pyrocko"}
            Backend engine used by the tracer.
        max_depth, max_distance, depth_step, distance_step : float, optional
            Table bounds/resolution (km) required when mode="table".
        distance_weight_start, distance_weight_end : float, optional
            Distance (km) over which the logistic distance weighting transitions.
        weight_start, weight_end : float, optional
            Logistic weighting values at the corresponding start/end distances.
        _fix_depth : bool
            If True, fixes event depths to their initial values (all events).
        _precalculate_velocities : bool
            If True, precomputes per-phase velocities for a 1D model at
            [0, max_depth] with spacing `depth_step` (for speed).
        """
        # --- Basic config / validation
        self.dim: int = 4  # Solve for (x, y, z, t0); hardcoded invariant.
        self._fix_depth: bool = _fix_depth

        self.tvel_file_path: Path = Path(model_tvel).expanduser().resolve()
        self.model_name: str = self.tvel_file_path.stem

        self.phases: Tuple[str, ...] = tuple(phases or ("P",))
        self.n_phases: int = len(self.phases)
        self.min_picks_per_event: int = int(min_picks_per_event)
        self.time_error_floor: float = float(time_error_floor)

        self.mode: Literal["table", "ray"] = self._validate_mode(mode)
        self.engine: Literal["taup", "pyrocko"] = self._validate_engine(engine)

        # --- Velocity model & tracer setup
        self.model = self._init_velocity_model(self.model_name, self.tvel_file_path)
        self.tracer: Union[TravelTimeTables, RayTracer]
        self.tracer, self.max_depth = self._init_tracer(
            mode=self.mode,
            engine=self.engine,
            phases=self.phases,
            tvel_path=self.tvel_file_path,
            model_name=self.model_name,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
        self.max_distance: Optional[float] = max_distance

        # --- Station / event containers (filled later by your loaders)
        self.inv: Inventory
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: np.ndarray  # shape (S,3): lat,lon,elev(m)

        self.n_stations: int = 0

        self.events: Union[List[Event], Catalog]
        self.n_events: int = 0

        # --- Data & model matrices (allocated later)
        self.origin_times: np.ndarray  # (E,)
        self.arrivals: np.ndarray  # (E, S*P)
        self.theoretical_arrivals: np.ndarray  # (E, S*P)
        self.derivative_matrix: np.ndarray  # (E*(S*P), E*dim)
        self.locations: np.ndarray  # (E,3) lat,lon,depth(km)

        # --- Error/weights
        self.arrival_time_errors: np.ndarray  # (E, S*P)
        self.time_error_W: np.ndarray  # (E*(S*P),)
        self.distance_W: np.ndarray  # (E*(S*P),)

        (self._distance_weighting, self.k, self.x0) = (
            self._configure_distance_weighting(
                distance_weight_start, distance_weight_end, weight_start, weight_end
            )
        )

        # --- Uncertainty/covariance products (computed post-inversion)
        self._covariances: np.ndarray  # (E,4,4)
        self._spatial_covariances: np.ndarray  # (E,3,3)
        self._principal_std: np.ndarray  # (E,3)
        self._principal_axes: np.ndarray  # (E,3,3)
        self._erz_erh: np.ndarray  # (E,2) -> [ERZ, ERH] @ 1σ

        # --- Optional velocity precomputation (1D only)
        self._precalculate_velocities: bool = bool(_precalculate_velocities)
        if self._precalculate_velocities:
            self._validate_precompute_args(max_depth, depth_step)
            # Currently only 1D velocity models are supported for this feature
            self.velocity_profile_dimensionality: int = 1
            self._precompute_velocity_profiles_1d(max_depth, depth_step)

    @staticmethod
    def _validate_mode(mode: str) -> Literal["table", "ray"]:
        if mode not in ("table", "ray"):
            raise ValueError(f"Unsupported mode: {mode!r}. Use 'table' or 'ray'.")
        return mode  # type: ignore[return-value]

    @staticmethod
    def _validate_engine(engine: str) -> Literal["taup", "pyrocko"]:
        if engine not in ("taup", "pyrocko"):
            raise ValueError(
                f"Unsupported engine: {engine!r}. Use 'taup' or 'pyrocko'."
            )
        return engine  # type: ignore[return-value]

    @staticmethod
    def _init_velocity_model(model_name: str, tvel_path: Path):
        try:
            return TauPyModel(model_name)
        except FileNotFoundError:
            # Build TauP model from .tvel if missing, then load
            build_taup_model(tvel_path.as_posix())
            return TauPyModel(model_name)

    @staticmethod
    def _configure_distance_weighting(
        distance_weight_start: Optional[float],
        distance_weight_end: Optional[float],
        weight_start: Optional[float],
        weight_end: Optional[float],
    ) -> tuple[bool, Optional[float], Optional[float]]:
        """
        Compute logistic weighting parameters (k, x0) if all inputs are provided.
        Returns: (enabled, k, x0).
        """
        args = (distance_weight_start, distance_weight_end, weight_start, weight_end)
        if any(a is None for a in args):
            return False, None, None

        d0 = float(distance_weight_start)  # km
        d1 = float(distance_weight_end)  # km
        w0 = float(weight_start)
        w1 = float(weight_end)

        if not (0.0 < w0 < 1.0 and 0.0 < w1 < 1.0):
            raise ValueError("weight_start and weight_end must be in (0, 1).")
        if d1 <= d0:
            raise ValueError(
                "distance_weight_end must be greater than distance_weight_start."
            )

        # Solve logistic parameters for: w(d) = 1 / (1 + exp(-k (d - x0)))
        k = (np.log((1.0 / w1) - 1.0) - np.log((1.0 / w0) - 1.0)) / (d1 - d0)
        x0 = d0 - (np.log((1.0 / w0) - 1.0) / k)
        return True, float(k), float(x0)

    @staticmethod
    def _validate_precompute_args(
        max_depth: Optional[float], depth_step: Optional[float]
    ) -> None:
        if max_depth is None or depth_step is None:
            raise ValueError(
                "When using precalculated velocities, 'max_depth' and 'depth_step' must be provided."
            )

    def _init_tracer(
        self,
        *,
        mode: Literal["table", "ray"],
        engine: Literal["taup", "pyrocko"],
        phases: Tuple[str, ...],
        tvel_path: Path,
        model_name: str,
        max_depth: Optional[float],
        max_distance: Optional[float],
        depth_step: Optional[float],
        distance_step: Optional[float],
    ):
        tracer: Union[TravelTimeTables, RayTracer]
        if mode == "table":
            # Validate required table params
            missing = [
                name
                for name, val in dict(
                    max_depth=max_depth,
                    max_distance=max_distance,
                    depth_step=depth_step,
                    distance_step=distance_step,
                ).items()
                if val is None
            ]
            if missing:
                raise ValueError(
                    "In table mode these variables are required: "
                    "max_depth, max_distance, depth_step, distance_step "
                    f"(missing: {', '.join(missing)})"
                )
            tracer = TravelTimeTables(
                tvel_path.as_posix(),
                model_name,
                list(phases),
                float(max_depth),
                float(max_distance),
                float(depth_step),
                float(distance_step),
                engine,
            )
            return tracer, float(max_depth)
        else:  # mode == "ray"
            tracer = RayTracer(model_path=tvel_path.as_posix(), engine=engine)
            return tracer, max_depth  # max_depth unused in ray mode

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
        self.time_error_W = np.ones(n_events * n_phases * n_stations)
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
                    if arrival_time_error is None or arrival_time_error <= 0.0:
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

                    self.time_error_W[
                        event_idx * self.n_phases * self.n_stations
                        + station_idx * self.n_phases
                        + phase_idx
                    ] = time_error_weight(arrival_time_error)

    def _derive_travel_time_param(self, event_location, station_location, phase):
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location
        dist_m, az_deg, _ = gps2dist_azimuth_nb(
            ev_lat, ev_lon, st_lat, st_lon, R=self.model.model.radius_of_planet * 1000.0
        )
        distance_km = dist_m / 1000.0
        if self.max_distance is not None and distance_km > self.max_distance:
            logger.info(
                f"Station at distance {distance_km:.2f} km exceeds max_distance {self.max_distance} km. Skipping this arrival."
            )
            return distance_km, None, (None, None, None, None)
        # if distance_km == 0.0:
        #     logger.debug(
        #         "Station is co-located with event epicenter; Ignoring arrival."
        #     )
        #     return distance_km, None, (None, None, None, None)
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

    def init_locations(
        self, trial_depth: float = 10.0, trial_time: float = 2.0
    ) -> np.ndarray:
        """
        Initialize event locations and origin times using a multi-seed search:
        - epicenter seeds: station centroid, componentwise medians, earliest-pick station, closest station
        - shallow depth grid around `trial_depth`
        - t0 via weighted least squares on (t_obs - t_theory)
        Falls back to the legacy heuristic if scoring fails.
        """

        def _usable_arrivals_for_event(ev_idx: int):
            """Return list of (station_idx, phase_idx, t_obs, sigma)."""
            usable = []
            for s in range(self.n_stations):
                for p in range(self.n_phases):
                    j = p + s * self.n_phases
                    t_obs = self.arrivals[ev_idx, j]
                    if not np.isnan(t_obs):
                        sigma = self.arrival_time_errors[ev_idx, j]
                        if not (sigma is None or np.isnan(sigma) or sigma <= 0.0):
                            usable.append((s, p, float(t_obs), float(sigma)))
            return usable

        def _estimate_t0_and_score(
            ev_idx: int, lat: float, lon: float, depth_km: float
        ):
            """Given a candidate hypocenter, compute weighted t0 and return (t0, wrms, n)."""
            items = _usable_arrivals_for_event(ev_idx)
            if len(items) < max(3, self.min_picks_per_event // 2):
                return None
            obs_minus_tt, w = [], []
            for s, p, t_obs, sigma in items:
                sta_lat, sta_lon, sta_elev = self.stations_locations[s]
                phase = self.phases[p]
                _, t_theory, _ = self._derive_travel_time_param(
                    (lat, lon, depth_km), (sta_lat, sta_lon, sta_elev), phase
                )
                if t_theory is None or not np.isfinite(t_theory):
                    continue
                obs_minus_tt.append(t_obs - float(t_theory))
                w.append(1.0 / (sigma * sigma))
            if len(obs_minus_tt) < max(3, self.min_picks_per_event // 2):
                return None
            obs_minus_tt = np.asarray(obs_minus_tt, dtype=float)
            w = np.asarray(w, dtype=float)
            t0 = float(np.sum(w * obs_minus_tt) / np.sum(w))  # weighted mean
            wrms = float(np.sqrt(np.sum(w * (obs_minus_tt - t0) ** 2) / np.sum(w)))
            return t0, wrms, len(obs_minus_tt)

        def _seed_candidates_for_event(event_idx: int, trial_depth_km: float):
            """Build a small set of epicentral seeds + a depth list."""
            # Stations with picks for this event
            picked = []
            t_first = np.inf
            earliest_idx = None
            for pk in self.events[event_idx].picks:
                key = (pk.waveform_id.network_code, pk.waveform_id.station_code)
                if key in self.stations:
                    sidx = self.stations.index(key)
                    picked.append(sidx)
                    if pk.time < t_first:
                        t_first = pk.time
                        earliest_idx = sidx
            picked = sorted(set(picked))
            if not picked:
                return [], []

            # Centroid & componentwise median of picked stations
            pts = np.asarray([self.stations_locations[i][:2] for i in picked], float)
            lat_c, lon_c = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            lat_m, lon_m = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
            seeds = [(lat_c, lon_c), (lat_m, lon_m)]
            if earliest_idx is not None:
                e_lat, e_lon = self.stations_locations[earliest_idx][:2]
                seeds.append((float(e_lat), float(e_lon)))

            # Add the station with the earliest pick time to the seeds
            earliest_pick_idx = min(
                picked,
                key=lambda idx: min(
                    pk.time
                    for pk in self.events[event_idx].picks
                    if (pk.waveform_id.network_code, pk.waveform_id.station_code)
                    == self.stations[idx]
                ),
            )
            earliest_lat, earliest_lon = self.stations_locations[earliest_pick_idx][:2]
            seeds.append((float(earliest_lat), float(earliest_lon)))

            # Deduplicate (numeric)
            uniq, seen = [], set()
            for lat, lon in seeds:
                key = (round(lat, 4), round(lon, 4))
                if key not in seen:
                    uniq.append((lat, lon))
                    seen.add(key)

            # Depth grid around trial_depth, bounded by self.max_depth if set
            d0 = max(0.0, float(trial_depth_km))
            dmax = float(self.max_depth) if self.max_depth is not None else d0 + 20.0
            grid = {round(x, 2) for x in [d0 * 0.5, d0, d0 * 1.5, d0 + 5.0, d0 + 10.0]}
            depths = sorted([d for d in grid if 0.0 <= d <= dmax]) or [d0]
            return uniq, depths

        # ---- main body ----
        for event_idx, event in enumerate(self.events):
            if len(event.picks) == 0:
                continue
            # keep only picks from known stations (as before)
            event.picks = [
                pick
                for pick in event.picks
                if (pick.waveform_id.network_code, pick.waveform_id.station_code)
                in self.stations
            ]

            seeds, depths = _seed_candidates_for_event(event_idx, trial_depth)
            best = None  # (wrms, lat, lon, depth_km, t0)

            for lat, lon in seeds:
                for depth_km in depths:
                    scored = _estimate_t0_and_score(event_idx, lat, lon, depth_km)
                    if scored is None:
                        continue
                    t0, wrms, _ = scored
                    if np.isfinite(wrms) and (best is None or wrms < best[0]):
                        best = (wrms, lat, lon, depth_km, t0)

            if best is not None:
                _, lat, lon, depth_km, t0 = best
                self.locations[event_idx] = (lat, lon, depth_km)
                self.origin_times[event_idx] = t0
                logger.debug(
                    f"[init] ev#{event_idx}: lat={lat:.4f} lon={lon:.4f} z={depth_km:.2f} km, t0={t0:.3f}"
                )
            else:
                # ---- fallback to legacy heuristic ----
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
                logger.debug(
                    f"[init:fallback] ev#{event_idx}: lat={station.latitude:.4f} "
                    f"lon={station.longitude:.4f} z={trial_depth:.2f} km, t0≈first_pick−{trial_time:.2f}s"
                )

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
                    distance_km, travel_time, spatial_derivatives = (
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
                    dt_dr, dt_dx, dt_dy, dt_dz = spatial_derivatives
                    logger.debug(
                        f"Event {event_idx}, Station {self.stations[station_idx]}, Phase {phase}: "
                        f"Distance {distance_km:.2f} km, Travel time {travel_time}, "
                        f"Derivatives dt/dx {dt_dx}, dt/dy {dt_dy}, dt/dz {dt_dz}"
                    )

                    self.theoretical_arrivals[event_idx, arr_index] = travel_time
                    row_idx = event_idx * (self.n_stations * self.n_phases) + arr_index

                    if self._distance_weighting:
                        # Weighting for distance-based damping
                        self.distance_W[row_idx] = distance_weight(
                            distance_km, k=self.k, x0=self.x0
                        )

                    # Populate the derivative matrix
                    self.derivative_matrix[row_idx, col_idx + 0] = dt_dy
                    self.derivative_matrix[row_idx, col_idx + 1] = dt_dx
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
        N_reg = N + (damping**2) * np.eye(self.dim)

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
        max_depth=50.0,
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
        logger.info("Event location completed.")
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
            residuals = travel_times_obs - self.theoretical_arrivals
            residuals_flat = residuals.flatten()
            b = (
                self.distance_W[valid_rows]
                * self.time_error_W[valid_rows]
                * residuals_flat[valid_rows]
            )
            A = (
                self.derivative_matrix[valid_rows, :]
                * self.time_error_W[valid_rows][:, np.newaxis]
                * self.distance_W[valid_rows][:, np.newaxis]
            )

            dx_solution = np.zeros(A.shape[1])
            if self.solver == "lsqr":
                dx_solution = lsqr(A, b, damp=0)[0]
            elif self.solver == "svd":
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
                dx_solution = Vt.T @ (s_inv * (U.T @ b))

            dx = dx_solution.reshape((self.n_events, self.dim))
            dx[:, 2] = np.clip(dx[:, 2], -1, 1)
            dx[:, 0:2] = np.clip(dx[:, 0:2], -0.5, 0.5)
            dx[:, 3] = np.clip(dx[:, 3], -1, 1)
            residual_vector = A @ dx_solution - b
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)

            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - dx[event_idx, 2]
                if not solve_airquakes and new_depth < 0.0:
                    logger.warning(
                        f"Event {event_idx} depth would become negative ({new_depth} km). Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0
                elif it > max_depth_iterations and new_depth > max_depth:
                    logger.warning(
                        f"Event {event_idx} depth would exceed max depth ({new_depth} km > {max_depth} km). Marking as invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)
            valid_mask = ~self.invalid_events
            self.locations[valid_mask, :3] -= dx[valid_mask, :3]
            self.origin_times[valid_mask] += dx[valid_mask, 3]
            damping = np.clip(update_damping_factor(damping), 0, max_damping)
            if res_norm < conv_threshold:
                break

        self._A_final = A
        self._valid_rows_final = valid_rows
        self._residuals_flat_final = residuals_flat
        self._lambda_final = damping
        logger.info(f"Location iterations completed in {it + 1} iterations.")
        logger.info(
            f"Located {np.sum(~self.invalid_events)} events, {np.sum(self.invalid_events)} events invalid."
        )

    def _compute_error_ellipsoids(self):
        self._covariances = np.zeros((self.n_events, self.dim, self.dim))
        self._spatial_covariances = np.zeros(
            (self.n_events, self.dim - 1, self.dim - 1)
        )
        self._principal_std = np.zeros((self.n_events, self.dim - 1))
        self._principal_axes = np.zeros((self.n_events, self.dim - 1, self.dim - 1))
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
            if not np.isfinite(v[2]):
                logger.warning(
                    "Invalid value for v[2], setting plunge and azimuth to 0."
                )
                plunge = 0.0
                azimuth = 0.0
            else:
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
