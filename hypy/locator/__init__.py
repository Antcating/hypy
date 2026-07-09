"""
Public API for hypy.locator.

Exposes EventLocator as a composition-based orchestrator.
"""

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from ..logger import get_logger
from ..propagation import PropagationEngine
from .config import (
    LocatorConfig,
    RunConfig,
    configure_distance_weighting,
)
from .data import DataHandler
from .derivatives import DerivativesHandler
from .errors import compute_error_ellipsoids
from .export import OriginExporter
from .inversion import InversionEngine
from .locations_setup import init_locations
from .types import (
    INFO,
    Catalog,
    DerivativeMethodLiteral,
    Event,
    Inventory,
    SolverLiteral,
)


class EventLocator:
    """
    Absolute seismic event relocator using 1D velocity model and event P/S phase picks.

    Parameters
    ----------
    propagation_engine : PropagationEngine
        Travel-time calculation engine. This is a required argument that specifies the
        backend to use for travel-time calculations and derivative assembly. The provided
        engine must be compatible with the specified derivative_method and support the
        required model features.
    phases : Sequence[str], optional
        List of seismic phases to use for travel-time calculations. Default is ("P",).
        Alowed phases, for now, are "P" and "S".
    min_picks_per_event : int, optional
        Minimum number of picks required per event to attempt location. Default is 4.
    default_time_error : float, optional
        Minimum pick time error (in seconds). Used if pick error is not given. Default is 0.1 s.
    fix_depth_km : Optional[float], optional
        If not None, event depths are fixed during inversion. Default is None.
    _precalculate_velocities : bool, optional
        If True, precomputes velocity profiles for all events to speed up travel-time
        calculations. Requires max_depth_km and depth_step_km to be set. Default is False.

    Notes
    -----
    Per-run knobs (distance weighting, observation error model, and the RNG seed)
    are not set here; they are passed to :meth:`locate` so they can be varied
    between runs on the same loaded data without rebuilding the locator.
    """  # noqa: E501

    def __init__(
        self,
        propagation_engine: PropagationEngine,
        phases: Iterable[str] = ("p", "s"),
        derivative_method: DerivativeMethodLiteral = "numerical",
        fix_depth_km: Optional[float] = None,
        min_picks_per_event: int = 4,
        default_time_error: float = 0.1,
        log_level: int = INFO,
        log_file: Optional[str] = None,
        _precalculate_velocities: bool = False,
    ):
        get_logger("hypy", level=log_level, log_file=log_file)

        if not isinstance(phases, Iterable) or not all(
            isinstance(phase, str) for phase in phases
        ):
            raise ValueError(
                f"Phases must be an iterable of strings, got {type(phases)}."  # noqa: E501
            )
        phases = tuple(phase.lower() for phase in phases)
        if any(phase not in propagation_engine.phases for phase in phases):
            raise ValueError(
                f"Propagation engine phases {propagation_engine.phases} do not match provided phases {phases}."  # noqa: E501
            )

        # Build immutable, construction-time configuration
        self.config = LocatorConfig(
            phases=phases,
            min_picks_per_event=min_picks_per_event,
            default_time_error=default_time_error,
            derivative_method=derivative_method,
            fix_depth_km=fix_depth_km,
            precalculate_velocities=bool(_precalculate_velocities),
        )

        # Initialize Data handling
        self.data = DataHandler(
            phases=self.config.phases,
            min_picks_per_event=self.config.min_picks_per_event,
            default_time_error=self.config.default_time_error,
            dim=self.config.dim,
            fix_depth_km=self.config.fix_depth_km,
        )

        backend_feature_flags = propagation_engine.feature_flags
        # backend_required_model_features = forward_engine.required_model_features

        if f"derivative.{self.config.derivative_method}" not in backend_feature_flags:
            raise ValueError(
                "Unsupported derivative method: "
                f"{self.config.derivative_method!r} "
                f"for {propagation_engine.__class__.__name__!r} backend."
            )

        self.propagation_backend = propagation_engine

        # Derivative engine - bridges config, data, model, and propagation backend
        self.derivative_engine = DerivativesHandler(
            config=self.config,
            data=self.data,
            tracer=self.propagation_backend,
        )

        # Whether to precompute velocity profiles for all events.
        # This can speed up travel-time calculations,
        # but introduces additional overhead during initialization.
        # Also, it introduces resolution to velocity values no matter
        # what travel-time calculation engine is used,
        # which can impact the accuracy of the solution in ray-tracing mode.
        if self.config.precalculate_velocities:
            raise NotImplementedError(
                "Precomputing velocity profiles is not yet implemented. "
                "This feature is planned for a future release."
            )
            # validate_precompute_args(max_depth_km, depth_step_km)
            # precompute_velocity_profiles = getattr(
            #     self.velocity_model,
            #     "_precompute_velocity_profiles_1d",
            #     None,
            # )
            # if callable(precompute_velocity_profiles):
            #     precompute_velocity_profiles(
            #         self.config.phases,
            #         max_depth_km,
            #         depth_step_km,
            #     )

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

    # Dummy alias for better discoverability in the API
    load = load_data

    def locate(
        self,
        solver: SolverLiteral = "lsqr",
        max_iterations: int = 100,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        safe_iteration_limit: int = 20,
        trial_depth_km: float = 10.0,
        distance_weight_start_km: Optional[float] = None,
        distance_weight_end_km: Optional[float] = None,
        weight_start: Optional[float] = 0.99,
        weight_end: Optional[float] = 0.01,
        full_data_covariance: bool = False,
        correlation_length_km: float = 1.0,
        error_coefficient: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[Union[List[Event], Catalog], np.ndarray]:
        """
        Run the full location pipeline.

        Performs iterative least-squares inversion to locate events
        based on the loaded data and configuration. The process includes:
        1. Initial location setup (seed generation)
        2. Iterative inversion to refine event locations
        3. Error analysis to compute uncertainty ellipsoids
        4. Finalization of Origin objects and reporting of invalid picks/events

        Parameters
        ----------
        solver : SolverLiteral, optional
            Solver method (see :data:`SolverLiteral`). Default is "lsqr".
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
        distance_weight_start_km : Optional[float], optional
            Distance (in km) at which distance-based weighting starts to be applied.
            If None, no distance weighting is applied. Default is None.
        distance_weight_end_km : Optional[float], optional
            Distance (in km) at which distance-based weighting reaches its minimum.
            If None, no distance weighting is applied. Default is None.
        weight_start : float, optional
            Weight factor at distance_weight_start_km. Default is 0.99.
        weight_end : float, optional
            Weight factor at distance_weight_end_km. Default is 0.01.
        full_data_covariance : bool, optional
            If True, compute event uncertainty using a correlated observation
            covariance model instead of diagonal row weighting. Default is False.
        correlation_length_km : float, optional
            Correlation length in km used in the exponential observation covariance
            model when full_data_covariance is enabled. Default is 1.0.
        error_coefficient : float, optional
            HYPOINVERSE-style coefficient applied to RMS in the error variance
            estimate ``RDERR^2 + (error_coefficient * RMS)^2``. Default is 1.0.
        seed : Optional[int], optional
            Seed for the random number generator used during location
            initialization. Set this to an integer to make this location run
            reproducible. If None, results are non-deterministic. Default is None.

        Returns
        -------
        (events, invalid_indices)
            events: List[Event] or Catalog with appended Origin objects.
            invalid_indices: np.ndarray of event indices that could not be located.
        """
        # Calculate distance weighting parameters
        distance_weighting_enabled, k, x0 = configure_distance_weighting(
            distance_weight_start_km, distance_weight_end_km, weight_start, weight_end
        )

        # Build immutable, per-run configuration
        run_config = RunConfig(
            distance_weighting_enabled=distance_weighting_enabled,
            k=k,
            x0=x0,
            full_data_covariance=bool(full_data_covariance),
            correlation_length_km=float(correlation_length_km),
            error_coefficient=float(error_coefficient),
            solver=solver,
            seed=seed,
        )

        init_locations(
            self.data,
            self.config,
            run_config,
            self.derivative_engine,
            trial_depth_km,
            self.propagation_backend.distance_step_km,
            self.propagation_backend.max_depth_km,
        )

        inversion_engine = InversionEngine(
            config=self.config,
            data=self.data,
            derivative_engine=self.derivative_engine,
            run_config=run_config,
        )
        inversion_results = inversion_engine.run_iterations(
            solver=run_config.solver,
            max_iterations=max_iterations,
            conv_threshold=conv_threshold,
            solve_airquakes=solve_airquakes,
            max_depth=self.propagation_backend.max_depth_km,
            safe_iteration_limit=safe_iteration_limit,
        )

        error_results = compute_error_ellipsoids(
            self.config,
            self.data,
            inversion_results,
            run_config,
        )

        exporter = OriginExporter(config=self.config, data=self.data)
        events = exporter.finalize_origins(error_results, as_catalog=True)
        invalid_indices = OriginExporter.report_invalid_events(self.data)

        return events, invalid_indices


__all__ = ["EventLocator"]
