"""
Public API for hypy.locator

Exposes EventLocator composed from internal mixins.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from typing_extensions import Literal

import numpy as np

from .config import (
    configure_distance_weighting,
    validate_engine,
    validate_mode,
    validate_precompute_args,
)
from .data import DataMixin
from .derivatives import DerivativesMixin
from .errors import ErrorAnalysisMixin
from .finalize import FinalizeMixin
from .init_locations import InitLocationsMixin
from .inversion import InversionMixin
from .model import ModelMixin
from .tracer_setup import TracerSetupMixin
from .types import (
    Catalog,
    Event,
    Inventory,
    SolverLiteral,
    EngineLiteral,
    ModeLiteral,
    TauPyModel,
)


if TYPE_CHECKING:
    from .state import LocatorState
else:

    class LocatorState:
        pass


class EventLocator(
    ModelMixin,
    TracerSetupMixin,
    DataMixin,
    DerivativesMixin,
    InitLocationsMixin,
    InversionMixin,
    ErrorAnalysisMixin,
    FinalizeMixin,
):
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

    def __init__(
        self: "LocatorState",
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
        super().__init__()
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

        # --- Station / event containers (filled later)
        self.inv: Inventory
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: np.ndarray
        self.n_stations: int = 0
        self.events: Union[List[Event], Catalog]
        self.n_events: int = 0

        # --- Data & model matrices (allocated later)
        self.origin_times: np.ndarray
        self.arrivals: np.ndarray
        self.theoretical_travel_times: np.ndarray
        self.derivative_matrix: np.ndarray
        self.locations: np.ndarray

        # --- Error/weights
        self.arrival_time_errors: np.ndarray
        self.time_error_W: np.ndarray
        self.distance_W: np.ndarray

        (self._distance_weighting, self.k, self.x0) = configure_distance_weighting(
            distance_weight_start, distance_weight_end, weight_start, weight_end
        )

        # --- Uncertainty/covariance products (computed later)
        self._covariances: np.ndarray
        self._spatial_covariances: np.ndarray
        self._principal_std: np.ndarray
        self._principal_axes: np.ndarray
        self._erz_erh: np.ndarray

        # --- Optional velocity precomputation
        self._precalculate_velocities: bool = bool(_precalculate_velocities)
        if self._precalculate_velocities:
            validate_precompute_args(max_depth, depth_step)
            self.velocity_profile_dimensionality: int = 1
            self._precompute_velocity_profiles_1d(max_depth, depth_step)  # type: ignore[arg-type]


__all__ = ["EventLocator"]
