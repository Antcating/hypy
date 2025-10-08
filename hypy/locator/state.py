# hypy/locator/state.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from obspy.core.event import Catalog, Event  # type: ignore[import-untyped]
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

from ..propagation import PropagationModel


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

    # velocity model / tracer
    model: TauPyModel
    tracer: PropagationModel
    solver: str

    # stations / events containers
    inv: Inventory
    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    n_stations: int
    events: Union[List[Event], Catalog]
    n_events: int
    fixed_depth_mask: List[bool]

    # data & matrices
    origin_times: np.ndarray
    arrivals: np.ndarray
    theoretical_arrivals: np.ndarray
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

    # --- cross-mixin methods used by others (add these) ---
    def _derive_travel_time_param(
        self,
        event_location: Tuple[float, float, float],
        station_location: Tuple[float, float, float],
        phase: str,
    ) -> Tuple[
        float,  # distance_km
        Optional[float],  # travel_time (s) or None
        Tuple[
            Optional[float], Optional[float], Optional[float], Optional[float]
        ],  # (dt_dr, dt_dx, dt_dy, dt_dz)
    ]: ...

    def _get_velocity_1d(self, depth: float, phase: str) -> float: ...

    def _populate_derivative_matrix(self) -> np.ndarray: ...

    def init_locations(
        self, trial_depth: float = ..., trial_time: float = ...
    ) -> np.ndarray: ...

    def _compute_error_ellipsoids(self) -> None: ...

    def _finalize_origins(self): ...

    def _run_location_iterations(
        self,
        max_iterations: int,
        conv_threshold: float,
        solve_airquakes: bool,
        max_depth: float,
        init_phase_iteration_max: int,
    ) -> None: ...

    def _compute_event_errors(
        self,
        A_e: np.ndarray,  # (m, 4)
        r_e: np.ndarray,  # (m,)
        pick_errs_e: np.ndarray,  # (m,)
        damping: float,
    ) -> Optional[Dict[str, Any]]: ...

    # ---- Data I/O helpers used by DataIOMixin ----
    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int) -> None: ...
    def _fill_data(self) -> None: ...