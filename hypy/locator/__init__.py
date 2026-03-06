"""
Public API for hypy.locator

Exposes EventLocator composed from internal mixins.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from hypy.locator.config import EngineLiteral, ModeLiteral

from ..propagation import PropagationModel
from .config import (
    configure_distance_weighting,
    validate_engine,
    validate_mode,
    validate_precompute_args,
)
from .data import DataHandler
from .derivatives import DerivativesMixin
from .errors import ErrorAnalysisMixin
from .finalize import FinalizeMixin
from .init_locations import InitLocationsMixin
from .inversion import InversionMixin
from .model import ModelMixin
from .tracer_setup import init_tracer
from .types import (
    Catalog,
    EngineLiteral,
    Event,
    Inventory,
    ModeLiteral,
    SolverLiteral,
    TauPyModel,
)

if TYPE_CHECKING:
    from .state import LocatorState
else:

    class LocatorState:
        pass


class EventLocator(
    ModelMixin,
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
        self,
        model_tvel_path: str,
        phases: Tuple[str, ...] = ("P", "S"),
        min_picks_per_event: int = 4,
        default_time_error: float = 0.1,
        mode: ModeLiteral = "table",
        engine: EngineLiteral = "taup",
        max_depth: Optional[float] = None,
        max_distance: Optional[float] = None,
        depth_step: Optional[float] = None,
        distance_step: Optional[float] = None,
        distance_weight_start: Optional[float] = None,
        distance_weight_end: Optional[float] = None,
        weight_start: Optional[float] = 0.99,
        weight_end: Optional[float] = 0.01,
        fix_depth: bool = False,
        _precalculate_velocities: bool = False,
    ):
        super().__init__()

        # Path to the velocity model file as there is no way to infer if the model is
        # already loaded in obspy's TauP or not.
        # The model will be loaded during the first call to locate().
        self.tvel_file_path: Path = Path(model_tvel_path).expanduser().resolve()
        self.model_name: str = self.tvel_file_path.stem

        # Validate mode and engine
        self.mode: ModeLiteral = validate_mode(mode)
        self.engine: EngineLiteral = validate_engine(engine)

        # Initiate TauPyModel for the given velocity model file.
        # TODO: Evaluate if this can be engine specific in the future
        # to remove hard dependency on obspy's TauPyModel when using other engines.
        self.model: TauPyModel = self._init_velocity_model(
            self.model_name, self.tvel_file_path
        )

        # Boundaries and sampling for travel-time tables (if using table mode)
        # TODO: Evaluate if these can be moved to the solver as they are
        # solution-specific parameters rather than locator-specific.
        self.max_depth = max_depth
        self.max_distance = max_distance
        self.depth_step = depth_step
        self.distance_step = distance_step

        # Distance-based weighting parameters
        # for optional distance weighting of residuals during inversion
        self.distance_weight_start = distance_weight_start
        self.distance_weight_end = distance_weight_end
        self.weight_start = weight_start
        self.weight_end = weight_end
        # Calculate parameters for distance weighting function based on the provided configuration
        (self._distance_weighting, self.k, self.x0) = configure_distance_weighting(
            distance_weight_start, distance_weight_end, weight_start, weight_end
        )

        # Whether to precompute velocity profiles for all events.
        # This can speed up travel-time calculations,
        # but introduces additional overhead during initialization.
        # Also, it introduces resolution to velocity values no matter
        # what travel-time calculation engine is used,
        # which can impact the accuracy of the solution in ray-tracing mode.
        self._precalculate_velocities: bool = False

        # Data handling
        self.dh: DataHandler = DataHandler(
            phases=phases,
            min_picks_per_event=min_picks_per_event,
            default_time_error=default_time_error,
            dim=4,
            fix_depth=fix_depth,
        )

        # Initialize the travel-time calculator (tracer) based on the selected mode and engine.
        self.tracer: PropagationModel = init_tracer(
            mode=self.mode,
            engine=self.engine,
            phases=self.dh.phases,
            tvel_file_path=self.tvel_file_path,
            max_depth=max_depth,
            max_distance=max_distance,
            depth_step=depth_step,
            distance_step=distance_step,
        )

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

    def load_data(self, inv: Inventory, catalog: Union[List[Event], Catalog]) -> None:
        """
        Load station and event data from ObsPy Inventory and Catalog objects.

        Parameters
        ----------
        inv : Inventory
            ObsPy Inventory object containing station metadata.
        catalog : Union[List[Event], Catalog]
            ObsPy Catalog or list of Event objects containing event and pick data.
        """

        self.dh.load_data(inv, catalog)


__all__ = ["EventLocator"]
