# hypy/locator/state.py
from __future__ import annotations

import numpy as np

from ..propagation import PropagationModel
from .types import (
    Any,
    Catalog,
    Dict,
    Event,
    Inventory,
    List,
    Optional,
    Protocol,
    SolverLiteral,
    TauPyModel,
    Tuple,
    Union,
    Path,
    ModeLiteral,
    EngineLiteral,
)


class LocatorState(Protocol):
    # core config
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

    # velocity model / tracer
    model: TauPyModel
    tracer: PropagationModel
    solver: SolverLiteral
    velocity_profile_dimensionality: int
    velocity_profiles: Dict[
        str, Tuple[np.ndarray, np.ndarray]
    ]  # phase → (depths, velocities)

    # stations / events containers
    inv: Inventory
    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    n_stations: int
    events: Union[List[Event], Catalog]
    n_events: int
    fixed_depth_mask: List[bool]

    # data & matrices
    origin_times: np.ndarray  # per-event t0 in RELATIVE seconds (to time_refs)
    time_refs: (
        np.ndarray
    )  # per-event absolute reference time (mean observed pick), seconds
    arrivals: np.ndarray  # arrival times RELATIVE to time_refs
    theoretical_travel_times: np.ndarray
    derivative_matrix: np.ndarray
    locations: np.ndarray

    # errors / weights
    arrival_time_errors: np.ndarray
    time_error_W: np.ndarray
    distance_W: np.ndarray
    invalid_events: np.ndarray

    # bookkeeping for errors/finalization
    _A_final: np.ndarray
    _valid_rows_final: np.ndarray
    _residuals_flat_final: np.ndarray
    _lambda_final: float

    # outputs
    _covariances: np.ndarray
    _spatial_covariances: np.ndarray
    _principal_std: np.ndarray
    _principal_axes: np.ndarray
    _erz_erh: np.ndarray

    # --- DerivativesMixin ---
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
        Returns (distance_km, travel_time, (dt_dr, dt_dx, dt_dy, dt_dz)).
        If the path exceeds self.max_distance, travel_time and derivatives are None.
        """
        ...

    def _populate_derivative_matrix(self) -> np.ndarray:
        """
        Fill the design (derivative) matrix rows corresponding to all valid
        event-station-phase observations, and compute theoretical arrivals.
        Returns 1D array of row indices (into the global design matrix) that were populated.
        """
        ...

    # --- ModelMixin ---
    def _get_velocity_1d(self, depth: float, phase: str) -> float:
        """
        Get velocity at a given depth for a specific phase.
        """
        ...

    def _precompute_velocity_profiles_1d(
        self, max_depth: float, depth_step: float
    ) -> None:
        """
        Precompute and store velocity as a function of depth for each phase.
        """
        ...

    def _init_velocity_model(self, model_name: str, tvel_file_path: Path) -> TauPyModel:
        """
        Initialize and return the velocity model (TauPyModel).
        """
        ...

    # --- TracerMixin ---
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
    ) -> Tuple[PropagationModel, Optional[float]]:
        """
        Initialize and return the propagation model (tracer) and its max depth.
        """
        ...

    # --- DataMixin ---
    def load_data(
        self,
        inventory: Inventory,
        events: Union[List[Event], Catalog],
    ) -> None:
        """
        Load inventory and events, extract the set of stations present in picks,
        validate counts, allocate arrays, and populate arrival/time-error matrices.
        """
        ...

    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int) -> None:
        """
        Allocate all working arrays used during event location.
        """
        ...

    def _fill_data(self) -> None:
        """
        Populate arrival and error matrices from loaded event and station data.
        """
        ...

    # --- InitLocationsMixin ---
    def init_locations(
        self, trial_depth: float = ..., trial_time: float = ...
    ) -> np.ndarray:
        """
        Smart multi-seed initialization; falls back to legacy heuristic.
        Returns initial locations array.
        """
        ...

    # --- InversionMixin ---
    def locate(
        self,
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        max_depth: float = 50.0,
        safe_iteration_limit: int = 20,
        trial_depth: float = 10.0,
        solver: SolverLiteral = "lsqr",
    ) -> Tuple[Any, np.ndarray]:
        """
        Main entry point for event location. Runs initialization, inversion, error analysis, and finalization.
        Returns (finalized origins, invalid event indices).
        """
        ...

    def _run_location_iterations(
        self,
        max_iterations: int,
        conv_threshold: float,
        solve_airquakes: bool,
        max_depth: float,
        init_phase_iteration_max: int,
    ) -> None:
        """
        Run iterative location updates for all events until convergence or max iterations.
        Updates locations, origin times, and residuals.
        """
        ...

    # --- ErrorAnalysisMixin ---
    def _compute_event_errors(
        self,
        A_e: np.ndarray,
        r_e: np.ndarray,
        pick_errs_e: np.ndarray,
        damping: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute full and spatial covariance matrices, principal axes, and error metrics for a single event.
        Returns a dictionary of error/covariance results.
        """
        ...

    def _compute_error_ellipsoids(self) -> None:
        """
        Compute and store error ellipsoids and principal axes for all events.
        """
        ...

    # --- FinalizeMixin ---
    def _finalize_origins(self) -> Union[List[Event], Catalog]:
        """
        Finalize and attach computed origins and uncertainties to each event.
        Returns the updated list or catalog of events.
        """
        ...

    def report_invalid_events(self) -> np.ndarray:
        """
        Return indices of events marked as invalid after location.
        """
        ...
