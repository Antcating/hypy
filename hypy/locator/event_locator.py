"""Event location public implementation.

This module intentionally avoids mixins/multiple inheritance.
The public API remains `EventLocator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..propagation import PropagationModel
from .config import (
    configure_distance_weighting,
    validate_engine,
    validate_mode,
    validate_precompute_args,
)
from .types import (
    Catalog,
    Event,
    Inventory,
    SolverLiteral,
    EngineLiteral,
    ModeLiteral,
    TauPyModel,
)

from . import data as _data
from . import derivatives as _derivatives
from . import errors as _errors
from . import finalize as _finalize
from . import init_locations as _init_locations
from . import inversion as _inversion
from . import model as _model
from . import tracer_setup as _tracer_setup


class EventLocator:
    """
    Absolute seismic event relocator using 1D velocity model and event P/S phase picks.

    Parameters
    ----------
    model_tvel : str
        Path to the 1D velocity model file in .tvel format.
    phases : Sequence[str], optional
        List of seismic phases to use for travel-time calculations. Default is ("P",).
        Alowed phases, for now, are "P" and "S".
    min_picks_per_event : int, optional
        Minimum number of picks required per event to attempt location. Default is 4.
    time_error_floor : float, optional
        Minimum pick time error (in seconds). Used if pick error is not given. Default is 0.1 s.
    mode : Literal["table", "ray"], optional
        Travel-time calculation mode. "table" uses precomputed travel-time tables (faster,
        less accurate), "ray" computes travel-times on-the-fly using ray-tracing (slower,
        more accurate). Default is "table".

        Note: "table" mode has to calculate travel-times for the entire distance/depth
        grid on the first call, which can take a while for large grids. Subsequent calls
        are fast.
    engine : Literal["taup", "pyrocko"], optional
        Travel-time calculation engine. "taup" uses Obspy's TauPyModel, "pyrocko" uses
        Pyrocko's travel-time calculator. Default is "taup".
    max_depth : Optional[float], optional
        Maximum depth (in km) for travel-time tables. Required if mode is "table" or if
        using precomputed velocity profiles. Optional if mode is "ray", as ray-tracing
        can handle arbitrary depths.
    max_distance : Optional[float], optional
        Maximum epicentral distance (in degrees) for travel-time tables. Required if
        mode is "table" or if using precomputed velocity profiles.
        Optional if mode is "ray", as ray-tracing can handle arbitrary
        distances.
    depth_step : Optional[float], optional
        Depth sampling interval (in km) for travel-time tables. Required if mode is
        "table". Optional if mode is "ray".
    distance_step : Optional[float], optional
        Distance sampling interval (in degrees) for travel-time tables. Required if mode
        is "table". Optional if mode is "ray".
    distance_weight_start : Optional[float], optional
        Distance (in km) at which distance-based weighting starts to be applied. If None,
        no distance weighting is applied. Default is None.
    distance_weight_end : Optional[float], optional
        Distance (in km) at which distance-based weighting reaches its minimum. If None,
        no distance weighting is applied. Default is None.
    weight_start : float, optional
        Weight factor at distance_weight_start. Default is 0.99.
    weight_end : float, optional
        Weight factor at distance_weight_end. Default is 0.01.
    _fix_depth : bool, optional
        If True, event depths are fixed during inversion. Default is False.
    _precalculate_velocities : bool, optional
        If True, precomputes velocity profiles for all events to speed up travel-time
        calculations. Requires max_depth and depth_step to be set. Default is False.
    """

    # --- core config
    dim: int
    _fix_depth: bool
    phases: Tuple[str, ...]
    n_phases: int
    min_picks_per_event: int
    time_error_floor: float
    mode: str
    engine: str
    max_depth: Optional[float]
    max_distance: Optional[float]
    _precalculate_velocities: bool
    _distance_weighting: bool
    k: Optional[float]
    x0: Optional[float]
    tvel_file_path: Path
    model_name: str

    # --- velocity model / tracer
    model: TauPyModel
    tracer: PropagationModel
    solver: SolverLiteral
    velocity_profile_dimensionality: int
    velocity_profiles: dict[str, Tuple[np.ndarray, np.ndarray]]

    # --- stations / events containers
    inv: Optional[Inventory]
    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    n_stations: int
    events: Union[List[Event], Catalog]
    n_events: int
    fixed_depth_mask: List[bool]

    # --- data & matrices
    origin_times: np.ndarray
    time_refs: np.ndarray
    arrivals: np.ndarray
    theoretical_travel_times: np.ndarray
    derivative_matrix: np.ndarray
    locations: np.ndarray

    # --- errors / weights
    arrival_time_errors: np.ndarray
    time_error_W: np.ndarray
    distance_W: np.ndarray
    invalid_events: np.ndarray

    # --- bookkeeping for errors/finalization
    _A_final: np.ndarray
    _valid_rows_final: np.ndarray
    _residuals_flat_final: np.ndarray
    _lambda_final: float

    # --- outputs
    _covariances: np.ndarray
    _spatial_covariances: np.ndarray
    _principal_std: np.ndarray
    _principal_axes: np.ndarray
    _erz_erh: np.ndarray

    def __init__(
        self,
        model_tvel: str,
        phases: Sequence[str] = ("P",),
        min_picks_per_event: int = 4,
        time_error_floor: float = 0.1,
        mode: str = "table",
        engine: str = "pyrocko",
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
        # --- defaults previously provided by mixin __init__ methods
        self.velocity_profile_dimensionality = 1
        self.velocity_profiles = {}

        self.inv: Optional[Inventory] = None  # set by load_data()
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations = np.empty((0, 3), dtype=float)
        self.n_stations = 0

        self.events: Union[List[Event], Catalog] = []
        self.n_events = 0

        self.fixed_depth_mask = []
        self.invalid_events = np.zeros(0, dtype=bool)

        self.arrivals = np.empty((0, 0))
        self.arrival_time_errors = np.empty((0, 0))
        self.theoretical_travel_times = np.empty((0, 0))
        self.derivative_matrix = np.empty((0, 0))
        self.time_error_W = np.empty((0,), dtype=float)
        self.distance_W = np.empty((0,), dtype=float)
        self.locations = np.empty((0, 3), dtype=float)
        self.origin_times = np.empty((0,), dtype=float)
        self.time_refs = np.empty((0,), dtype=float)

        self._covariances = np.empty((0, 0, 0), dtype=float)
        self._spatial_covariances = np.empty((0, 0, 0), dtype=float)
        self._principal_std = np.empty((0, 0), dtype=float)
        self._principal_axes = np.empty((0, 0, 0), dtype=float)
        self._erz_erh = np.empty((0, 0), dtype=float)
        self._residuals_flat_final = np.empty((0,), dtype=float)

        # inversion bookkeeping (populated by _run_location_iterations)
        self._A_final = np.empty((0, 0), dtype=float)
        self._valid_rows_final = np.empty((0,), dtype=int)
        self._lambda_final = 0.0

        # --- Basic config / validation
        self.dim: int = 4  # Solve for (x, y, z, t0); hardcoded invariant.
        self._fix_depth: bool = _fix_depth

        self.tvel_file_path: Path = Path(model_tvel).expanduser().resolve()
        self.model_name: str = self.tvel_file_path.stem

        self.phases: Tuple[str, ...] = tuple(phases or ("P",))
        self.n_phases: int = len(self.phases)
        self.min_picks_per_event: int = int(min_picks_per_event)
        self.time_error_floor: float = float(time_error_floor)

        self.mode: str = validate_mode(mode)
        self.engine: str = validate_engine(engine)
        self.solver: SolverLiteral = "lsqr"

        # --- Velocity model & tracer setup
        self.model: TauPyModel = self._init_velocity_model(
            self.model_name, self.tvel_file_path
        )
        self.tracer, self.max_depth = self._init_tracer(
            mode=self.mode,
            engine=self.engine,
            phases=self.phases,
            tvel_file_path=self.tvel_file_path,
            model_name=self.model_name,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
        self.max_distance: Optional[float] = max_distance

        (self._distance_weighting, self.k, self.x0) = configure_distance_weighting(
            distance_weight_start, distance_weight_end, weight_start, weight_end
        )

        # --- Optional velocity precomputation
        self._precalculate_velocities: bool = bool(_precalculate_velocities)
        if self._precalculate_velocities:
            validate_precompute_args(max_depth, depth_step)
            assert max_depth is not None
            assert depth_step is not None
            self._precompute_velocity_profiles_1d(float(max_depth), float(depth_step))

    def _compute_event_errors(self, A_e, r_e, pick_errs_e, damping: float):
        return _errors.compute_event_errors(
            dim=self.dim,
            A_e=A_e,
            r_e=r_e,
            pick_errs_e=pick_errs_e,
            damping=damping,
        )

    def _derive_travel_time_param(self, event_location, station_location, phase: str):
        ev_lat, ev_lon, ev_depth_km = event_location
        st_lat, st_lon, st_elev_m = station_location
        distance_km, travel_time, (dt_dx, dt_dy, dt_dz) = (
            self.tracer.calculate_arrival_parameters(
                ev_lon,
                ev_lat,
                ev_depth_km,
                st_lon,
                st_lat,
                st_elev_m,
                phase,
            )
        )
        return distance_km, travel_time, (None, dt_dx, dt_dy, dt_dz)

    # ---- public API methods (delegating to implementation functions)

    def load_data(
        self, inventory: Inventory, events: Union[List[Event], Catalog]
    ) -> None:
        result = _data.parse_data(
            inventory=inventory,
            events=events,
            phases=self.phases,
            min_picks_per_event=self.min_picks_per_event,
            time_error_floor=self.time_error_floor,
            fix_depth=self._fix_depth,
            dim=self.dim,
        )

        self.inv = inventory
        self.events = events
        self.n_events = int(result["n_events"])
        self.stations = result["stations"]
        self.stations_locations = result["stations_locations"]
        self.n_stations = int(result["n_stations"])
        self.fixed_depth_mask = result["fixed_depth_mask"]

        self.arrivals = result["arrivals"]
        self.arrival_time_errors = result["arrival_time_errors"]
        self.theoretical_travel_times = result["theoretical_travel_times"]
        self.derivative_matrix = result["derivative_matrix"]
        self.time_error_W = result["time_error_W"]
        self.distance_W = result["distance_W"]
        self.locations = result["locations"]
        self.origin_times = result["origin_times"]
        self.time_refs = result["time_refs"]
        self.invalid_events = result["invalid_events"]

        return None

    def init_locations(self, trial_depth: float = 10.0, trial_time: float = 2.0):
        assert self.inv is not None
        result = _init_locations.init_locations(
            inventory=self.inv,
            events=list(self.events),
            stations=self.stations,
            stations_locations=self.stations_locations,
            phases=self.phases,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            min_picks_per_event=self.min_picks_per_event,
            max_depth=self.max_depth,
            tracer=self.tracer,
            trial_depth=trial_depth,
            trial_time=trial_time,
            arrivals=self.arrivals,
            arrival_time_errors=self.arrival_time_errors,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
        )
        self.events = result["events"]
        self.locations = result["locations"]
        self.origin_times = result["origin_times"]
        return self.locations

    def locate(
        self,
        max_iterations=1000,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=50.0,
        safe_iteration_limit=20,
        trial_depth=10.0,
        solver: SolverLiteral = "lsqr",
    ):
        if solver not in ["lsqr", "svd"]:
            raise ValueError("Unsupported solver: {solver}. Supported: 'lsqr', 'svd'.")
        self.solver = solver

        self.init_locations(trial_depth=trial_depth)
        self._run_location_iterations(
            max_iterations,
            conv_threshold,
            solve_airquakes,
            max_depth,
            safe_iteration_limit,
        )
        self._compute_error_ellipsoids()
        return self._finalize_origins(), self.report_invalid_events()

    # ---- internal methods required by the locator pipeline

    def _populate_derivative_matrix(self) -> np.ndarray:
        result = _derivatives.populate_derivative_matrix(
            tracer=self.tracer,
            locations=self.locations,
            stations_locations=self.stations_locations,
            stations=self.stations,
            phases=self.phases,
            n_events=self.n_events,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            dim=self.dim,
            arrivals=self.arrivals,
            theoretical_travel_times=self.theoretical_travel_times,
            derivative_matrix=self.derivative_matrix,
            distance_W=self.distance_W,
            time_error_W=self.time_error_W,
            invalid_events=self.invalid_events,
            fixed_depth_mask=self.fixed_depth_mask,
            distance_weighting=self._distance_weighting,
            k=self.k,
            x0=self.x0,
        )
        self.theoretical_travel_times = result["theoretical_travel_times"]
        self.derivative_matrix = result["derivative_matrix"]
        self.distance_W = result["distance_W"]
        return result["valid_rows"]

    def _run_location_iterations(
        self,
        max_iterations,
        conv_threshold,
        solve_airquakes,
        max_depth,
        init_phase_iteration_max,
    ):
        result = _inversion.run_location_iterations(
            solver=self.solver,
            max_iterations=int(max_iterations),
            conv_threshold=float(conv_threshold),
            solve_airquakes=bool(solve_airquakes),
            max_depth=float(max_depth),
            init_phase_iteration_max=int(init_phase_iteration_max),
            n_events=self.n_events,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            dim=self.dim,
            mode=self.mode,
            tracer=self.tracer,
            stations=self.stations,
            stations_locations=self.stations_locations,
            phases=self.phases,
            fixed_depth_mask=self.fixed_depth_mask,
            distance_weighting=bool(self._distance_weighting),
            k=self.k,
            x0=self.x0,
            fix_depth=bool(self._fix_depth),
            precalculate_velocities=bool(self._precalculate_velocities),
            model_radius_of_planet_km=float(self.model.model.radius_of_planet),
            arrivals=self.arrivals,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
            theoretical_travel_times=self.theoretical_travel_times,
            derivative_matrix=self.derivative_matrix,
            time_error_W=self.time_error_W,
            distance_W=self.distance_W,
            invalid_events=self.invalid_events,
        )

        self.locations = result["locations"]
        self.origin_times = result["origin_times"]
        self.invalid_events = result["invalid_events"]
        self.theoretical_travel_times = result["theoretical_travel_times"]
        self.derivative_matrix = result["derivative_matrix"]
        self.distance_W = result["distance_W"]
        self._A_final = result["A_final"]
        self._valid_rows_final = result["valid_rows_final"]
        self._residuals_flat_final = result["residuals_flat_final"]
        self._lambda_final = float(result["lambda_final"])
        return None

    def _compute_error_ellipsoids(self):
        result = _errors.compute_error_ellipsoids(
            dim=self.dim,
            n_events=self.n_events,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            A_final=self._A_final,
            valid_rows_final=self._valid_rows_final,
            residuals_flat_final=self._residuals_flat_final,
            arrival_time_errors=self.arrival_time_errors,
            invalid_events=self.invalid_events,
            damping=0.1,
        )
        self._covariances = result["covariances"]
        self._spatial_covariances = result["spatial_covariances"]
        self._principal_std = result["principal_std"]
        self._principal_axes = result["principal_axes"]
        self._erz_erh = result["erz_erh"]
        return None

    def _finalize_origins(self):
        self.events = _finalize.finalize_origins(
            events=self.events,
            invalid_events=self.invalid_events,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
            principal_std=self._principal_std,
            principal_axes=self._principal_axes,
            erz_erh=self._erz_erh,
            fixed_depth_mask=self.fixed_depth_mask,
            stations=self.stations,
            phases=self.phases,
            n_phases=self.n_phases,
            arrivals=self.arrivals,
            theoretical_travel_times=self.theoretical_travel_times,
        )
        return self.events

    def report_invalid_events(self) -> np.ndarray:
        return _finalize.report_invalid_events(invalid_events=self.invalid_events)

    # ---- tracer/model helpers

    @staticmethod
    def _init_velocity_model(model_name: str, tvel_file_path: Path) -> TauPyModel:
        return _model._init_velocity_model(model_name, tvel_file_path)

    def _precompute_velocity_profiles_1d(self, max_depth, depth_step):
        self.velocity_profiles = _model.precompute_velocity_profiles_1d(
            model=self.model,
            phases=self.phases,
            max_depth=float(max_depth),
            depth_step=float(depth_step),
        )
        return None

    def _get_velocity_1d(self, depth, phase):
        return _model.get_velocity_1d(
            model=self.model,
            precalculated=bool(self._precalculate_velocities),
            velocity_profiles=self.velocity_profiles,
            depth_km=float(depth),
            phase=str(phase),
        )

    def _init_tracer(
        self,
        *,
        mode: ModeLiteral,
        engine: EngineLiteral,
        phases: tuple[str, ...],
        tvel_file_path,
        model_name: str,
        max_depth: Optional[float],
        max_distance: Optional[float],
        depth_step: Optional[float],
        distance_step: Optional[float],
    ):
        return _tracer_setup._init_tracer(
            mode=mode,
            engine=engine,
            phases=phases,
            tvel_file_path=tvel_file_path,
            model_name=model_name,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )
