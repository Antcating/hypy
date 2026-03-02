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
    calculate_distance_weighting_params,
    validate_engine,
    validate_mode,
    validate_precompute_args,
)
from .types import (
    Catalog,
    Event,
    Inventory,
    SolverLiteral,
    TauPyModel,
)

from . import types as _types

from . import data as _data
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
        self.model: TauPyModel = _model._init_velocity_model(
            self.model_name, self.tvel_file_path
        )
        self.tracer, self.max_depth = _tracer_setup._init_tracer(
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

        (self._distance_weighting, self.k, self.x0) = (
            calculate_distance_weighting_params(
                distance_weight_start, distance_weight_end, weight_start, weight_end
            )
        )

        # --- Optional velocity precomputation
        self._precalculate_velocities: bool = bool(_precalculate_velocities)
        if self._precalculate_velocities:
            validate_precompute_args(max_depth, depth_step)
            assert max_depth is not None
            assert depth_step is not None
            self.velocity_profiles = _model.precompute_velocity_profiles_1d(
                model=self.model,
                phases=self.phases,
                max_depth=float(max_depth),
                depth_step=float(depth_step),
            )

    # ---- public API methods

    def load(self, inventory: Inventory, events: Union[List[Event], Catalog]) -> None:
        parsed_data = _data.parse_data(
            inventory=inventory,
            events=events,
            phases=self.phases,
            min_picks_per_event=self.min_picks_per_event,
            time_error_floor=self.time_error_floor,
            fix_depth=self._fix_depth,
            dim=self.dim,
        )
        # parse_data returns a typed tuple: (inv, event_meta, stations_info, matrices, weights)
        inv, event_meta, stations_info, matrices, weights = parsed_data

        self.inv = inv
        self.events = event_meta.events
        self.n_events = int(event_meta.n_events)
        self.stations = stations_info.stations
        self.stations_locations = stations_info.stations_locations
        self.n_stations = int(stations_info.n_stations)
        self.fixed_depth_mask = event_meta.fixed_depth_mask

        self.arrivals = matrices.arrivals
        self.arrival_time_errors = matrices.arrival_time_errors
        self.theoretical_travel_times = matrices.theoretical_travel_times
        self.derivative_matrix = matrices.derivative_matrix
        self.time_error_W = weights.time_error_W
        self.distance_W = weights.distance_W
        self.locations = matrices.locations
        self.origin_times = matrices.origin_times
        self.time_refs = matrices.time_refs
        self.invalid_events = matrices.invalid_events

        return None

    def init_locations(self, trial_depth: float = 10.0, trial_time: float = 2.0):
        assert self.inv is not None
        cfg = _types.InitConfig(
            stations=self.stations,
            stations_locations=self.stations_locations,
            phases=self.phases,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            min_picks_per_event=self.min_picks_per_event,
            max_depth=self.max_depth,
            trial_depth=float(trial_depth),
            trial_time=float(trial_time),
        )

        matrices = _types.Matrices(
            arrivals=self.arrivals,
            arrival_time_errors=self.arrival_time_errors,
            theoretical_travel_times=self.theoretical_travel_times,
            derivative_matrix=self.derivative_matrix,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
            invalid_events=self.invalid_events,
        )
        result = _init_locations.init_locations(
            inventory=self.inv,
            events=list(self.events),
            tracer=self.tracer,
            cfg=cfg,
            matrices=matrices,
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

    def _run_location_iterations(
        self,
        max_iterations,
        conv_threshold,
        solve_airquakes,
        max_depth,
        init_phase_iteration_max,
    ):
        cfg = _types.IterationConfig(
            solver=self.solver,
            max_iterations=int(max_iterations),
            conv_threshold=float(conv_threshold),
            solve_airquakes=bool(solve_airquakes),
            max_depth=float(max_depth),
            init_phase_iteration_max=int(init_phase_iteration_max),
            mode=str(self.mode),
            fix_depth=bool(self._fix_depth),
            precalculate_velocities=bool(self._precalculate_velocities),
            model_radius_of_planet_km=float(self.model.model.radius_of_planet),
        )

        ctx = _types.DerivativeContext(
            tracer=self.tracer,
            stations=self.stations,
            stations_locations=self.stations_locations,
            phases=self.phases,
            n_events=self.n_events,
            n_stations=self.n_stations,
            n_phases=self.n_phases,
            dim=self.dim,
            fixed_depth_mask=self.fixed_depth_mask,
            distance_weighting=bool(self._distance_weighting),
            k=float(self.k) if self.k is not None else None,
            x0=float(self.x0) if self.x0 is not None else None,
        )

        matrices = _types.Matrices(
            arrivals=self.arrivals,
            arrival_time_errors=self.arrival_time_errors,
            theoretical_travel_times=self.theoretical_travel_times,
            derivative_matrix=self.derivative_matrix,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
            invalid_events=self.invalid_events,
        )
        weights = _types.Weights(
            time_error_W=self.time_error_W,
            distance_W=self.distance_W,
        )

        solution = _inversion.run_location_iterations(
            cfg=cfg,
            ctx=ctx,
            matrices=matrices,
            weights=weights,
        )

        self.locations = solution["locations"]
        self.origin_times = solution["origin_times"]
        self.invalid_events = solution["invalid_events"]
        self.theoretical_travel_times = solution["theoretical_travel_times"]
        self.derivative_matrix = solution["derivative_matrix"]
        self.distance_W = solution["distance_W"]
        self._A_final = solution["A_final"]
        self._valid_rows_final = solution["valid_rows_final"]
        self._residuals_flat_final = solution["residuals_flat_final"]
        self._lambda_final = float(solution["lambda_final"])
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
        matrices = _types.Matrices(
            arrivals=self.arrivals,
            arrival_time_errors=self.arrival_time_errors,
            theoretical_travel_times=self.theoretical_travel_times,
            derivative_matrix=self.derivative_matrix,
            locations=self.locations,
            origin_times=self.origin_times,
            time_refs=self.time_refs,
            invalid_events=self.invalid_events,
        )
        ctx = _types.FinalizeContext(
            stations=self.stations,
            phases=self.phases,
            n_phases=self.n_phases,
            fixed_depth_mask=self.fixed_depth_mask,
        )
        unc = _types.FinalizeUncertainty(
            principal_std=self._principal_std,
            principal_axes=self._principal_axes,
            erz_erh=self._erz_erh,
        )

        self.events = _finalize.finalize_origins(
            events=self.events,
            matrices=matrices,
            ctx=ctx,
            unc=unc,
        )
        return self.events

    def report_invalid_events(self) -> np.ndarray:
        return _finalize.report_invalid_events(invalid_events=self.invalid_events)
