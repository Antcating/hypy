"""
Public API for hypy.locator

Exposes EventLocator as a composition-based orchestrator.
Internal modules provide independent component classes:
- LocatorConfig: immutable settings
- DataHandler: mutable runtime arrays and event/station data
- VelocityModelHandler: velocity model access
- DerivativeEngine: travel-time derivative computation
- InversionEngine: iterative least-squares relocation
- ErrorAnalyzer: uncertainty estimation
- OriginFinalizer: ObsPy Origin construction
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..propagation import PropagationModel
from .config import (
    LocatorConfig,
    configure_distance_weighting,
    validate_engine,
    validate_mode,
    validate_precompute_args,
    validate_solver,
)
from .data import DataHandler
from .derivatives import DerivativeEngine
from .errors import ErrorAnalyzer
from .finalize import OriginFinalizer
from .inversion import InversionEngine
from .locations_setup import init_locations
from .model import VelocityModelHandler
from .tracer_setup import init_tracer
from .types import (
    Catalog,
    EngineLiteral,
    Event,
    Inventory,
    ModeLiteral,
    SolverLiteral,
)


class EventLocator:
    """
    Absolute seismic event relocator using 1D velocity model and event P/S phase picks.

    Parameters
    ----------
    model_tvel_path : str
        Path to the 1D velocity model file in .tvel format.
    phases : Sequence[str], optional
        List of seismic phases to use for travel-time calculations. Default is ("P",).
        Alowed phases, for now, are "P" and "S".
    min_picks_per_event : int, optional
        Minimum number of picks required per event to attempt location. Default is 4.
    default_time_error : float, optional
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
    max_depth_km : Optional[float], optional
        Maximum depth (in km) for travel-time tables. Required if mode is "table" or if
        using precomputed velocity profiles. Optional if mode is "ray", as ray-tracing
        can handle arbitrary depths.
    max_distance_km : Optional[float], optional
        Maximum epicentral distance (in degrees) for travel-time tables. Required if
        mode is "table" or if using precomputed velocity profiles.
        Optional if mode is "ray", as ray-tracing can handle arbitrary
        distances.
    depth_step_km : Optional[float], optional
        Depth sampling interval (in km) for travel-time tables. Required if mode is
        "table". Optional if mode is "ray".
    distance_step_km : Optional[float], optional
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
    fix_depth : bool, optional
        If True, event depths are fixed during inversion. Default is False.
    _precalculate_velocities : bool, optional
        If True, precomputes velocity profiles for all events to speed up travel-time
        calculations. Requires max_depth_km and depth_step_km to be set. Default is False.
    """

    def __init__(
        self,
        model_tvel_path: str,
        phases: Tuple[str, ...] = ("P", "S"),
        min_picks_per_event: int = 4,
        default_time_error: float = 0.1,
        mode: ModeLiteral = "table",
        engine: EngineLiteral = "taup",
        max_depth_km: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        depth_step_km: Optional[float] = None,
        distance_step_km: Optional[float] = None,
        distance_weight_start: Optional[float] = None,
        distance_weight_end: Optional[float] = None,
        weight_start: Optional[float] = 0.99,
        weight_end: Optional[float] = 0.01,
        fix_depth: bool = False,
        _precalculate_velocities: bool = False,
    ):
        # Validate mode and engine
        validated_mode: ModeLiteral = validate_mode(mode)
        validated_engine: EngineLiteral = validate_engine(engine)

        # Resolve model path
        tvel_file_path = Path(model_tvel_path).expanduser().resolve()

        # Calculate distance weighting parameters
        distance_weighting_enabled, k, x0 = configure_distance_weighting(
            distance_weight_start, distance_weight_end, weight_start, weight_end
        )

        # Build immutable configuration
        self.config = LocatorConfig(
            tvel_file_path=tvel_file_path,
            model_name=tvel_file_path.stem,
            phases=tuple(phases),
            min_picks_per_event=min_picks_per_event,
            default_time_error=default_time_error,
            mode=validated_mode,
            engine=validated_engine,
            max_depth_km=max_depth_km,
            max_distance_km=max_distance_km,
            depth_step_km=depth_step_km,
            distance_step_km=distance_step_km,
            fix_depth=fix_depth,
            precalculate_velocities=bool(_precalculate_velocities),
            distance_weighting_enabled=distance_weighting_enabled,
            k=k,
            x0=x0,
        )

        # Initiate TauPyModel for the given velocity model file.
        # TODO: Evaluate if this can be engine specific in the future
        # to remove hard dependency on obspy's TauPyModel when using other engines.
        self.model_handler = VelocityModelHandler(tvel_file_path)

        # Initialize Data handling
        self.data = DataHandler(
            phases=self.config.phases,
            min_picks_per_event=self.config.min_picks_per_event,
            default_time_error=self.config.default_time_error,
            dim=self.config.dim,
            fix_depth=self.config.fix_depth,
        )

        # Initialize the travel-time calculator (tracer) based on the selected mode and engine.
        self.tracer: PropagationModel = init_tracer(
            mode=self.config.mode,
            engine=self.config.engine,
            phases=self.config.phases,
            tvel_file_path=self.config.tvel_file_path,
            max_depth=max_depth_km,
            max_distance=max_distance_km,
            depth_step=depth_step_km,
            distance_step=distance_step_km,
        )

        # Derivative engine — bridges config, data, model, and tracer
        self.derivative_engine = DerivativeEngine(
            config=self.config,
            data=self.data,
            model_handler=self.model_handler,
            tracer=self.tracer,
        )

        # Whether to precompute velocity profiles for all events.
        # This can speed up travel-time calculations,
        # but introduces additional overhead during initialization.
        # Also, it introduces resolution to velocity values no matter
        # what travel-time calculation engine is used,
        # which can impact the accuracy of the solution in ray-tracing mode.
        if self.config.precalculate_velocities:
            validate_precompute_args(max_depth_km, depth_step_km)
            self.model_handler._precompute_velocity_profiles_1d(
                self.config.phases, max_depth_km, depth_step_km
            )  # type: ignore[arg-type]

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
        self.data.load_data(inv, catalog)

    # Alias for backward compatibility
    load = load_data

    def locate(
        self,
        solver: SolverLiteral = "lsqr",
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        safe_iteration_limit: int = 20,
        trial_depth_km: float = 10.0,
    ) -> Tuple[Union[List[Event], Catalog], np.ndarray]:
        """
        Run the full location pipeline: initialization, iterative inversion,
        error analysis, and finalization.

        Parameters
        ----------
        solver : Literal["lsqr", "svd"], optional
            Solver method. Default is "lsqr".
        max_iterations : int, optional
            Maximum number of inversion iterations. Default is 1000.
        conv_threshold : float, optional
            RMS convergence threshold (seconds). Default is 1e-3.
        solve_airquakes : bool, optional
            If True, allows negative depths. Default is False.
        safe_iteration_limit : int, optional
            Iteration number after which depth constraints become strict. Default is 20.
        trial_depth_km : float, optional
            Initial trial depth (km) for seed generation. Default is 10.0.

        Returns
        -------
        (events, invalid_indices)
            events: List[Event] or Catalog with appended Origin objects.
            invalid_indices: np.ndarray of event indices that could not be located.
        """
        solver = validate_solver(solver)

        init_locations(
            self.data,
            self.derivative_engine,
            trial_depth_km,
            self.config.max_depth_km,
        )

        inversion_engine = InversionEngine(
            config=self.config,
            data=self.data,
            model_handler=self.model_handler,
            derivative_engine=self.derivative_engine,
        )
        inversion_results = inversion_engine.run_iterations(
            solver=solver,
            max_iterations=max_iterations,
            conv_threshold=conv_threshold,
            solve_airquakes=solve_airquakes,
            max_depth=self.config.max_depth_km or 50.0,
            safe_iteration_limit=safe_iteration_limit,
        )

        error_analyzer = ErrorAnalyzer(config=self.config, data=self.data)
        error_results = error_analyzer.compute_error_ellipsoids(inversion_results)

        finalizer = OriginFinalizer(config=self.config, data=self.data)
        events = finalizer.finalize_origins(error_results)
        invalid_indices = OriginFinalizer.report_invalid_events(self.data)

        return events, invalid_indices


__all__ = ["EventLocator"]
