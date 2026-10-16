"""EventLocator orchestrator for hypy."""

from logging import INFO
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from obspy.core.event import Catalog, Event  # type: ignore[import-untyped]
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]

from ..logger import get_logger
from ..propagation import PropagationEngine
from ..utils.seismic import normalize_phase
from ..utils.weighting import configure_distance_weighting
from .config import (
    LocatorConfig,
    RunConfig,
)
from .data import DataHandler
from .derivatives import DerivativesHandler
from .errors import compute_error_ellipsoids
from .export import finalize_origins
from .inversion import InversionEngine
from .locations_setup import seed_locations
from .types import (
    DerivativeMethodLiteral,
    UnresolvedDepthLiteral,
)


class EventLocator:
    """Iterative seismic event relocator.

    Parameters
    ----------
    propagation_engine : PropagationEngine
        Ray parameters calculation engine. The provided
        engine must be compatible with the specified derivative_method
        and support the required model features.
    initial_propagation_engine : PropagationEngine, optional
        Optional forward model used during location seeding.
        When set, seeding scores a lat/lon/depth grid. When ``None``,
        heuristic epicenter seeds are scored with ``propagation_engine``.
        Default is ``None``.
    phases : Sequence[str], optional
        List of seismic phases to use for travel-time calculations. Default is ("P",).
        Allowed phases are "P" and "S".
    min_picks_per_event : int, optional
        Minimum number of picks required per event to attempt location. Default is 4.
    time_error_floor_s : float, optional
        Minimum pick time error (in seconds). Used if pick error is not given.
        Default is 0.1 s.
    fix_depth_km : Optional[float], optional
        If not None, event depths are fixed during inversion. Default is None.
    latitude_min_deg, latitude_max_deg, longitude_min_deg, longitude_max_deg : float, optional
        Optional geographic box that constrains event epicenters during seeding
        and inversion iterations. All four must be set together, or all
        omitted. When set, they override the station-envelope horizontal
        extent for grid-search seeding, clamp heuristic seeds, and keep
        iterative updates inside the box. Default is ``None``.

    Notes
    -----
    Per-run knobs (distance/residual weighting, observation error model, and the
    RNG seed) are not set here; they are passed to :meth:`locate` so they can be
    varied between runs on the same loaded data without rebuilding the locator.

    Prefer a precomputed travel-time table with coarse ``depth_step_km`` / ``distance_step_km``
    when using ``initial_propagation_engine`` so a 3D grid search stays cheap.
    """  # noqa: E501

    def __init__(
        self,
        propagation_engine: PropagationEngine,
        initial_propagation_engine: Optional[PropagationEngine] = None,
        phases: Sequence[str] = ("p", "s"),
        derivative_method: DerivativeMethodLiteral = "numerical",
        fix_depth_km: Optional[float] = None,
        min_picks_per_event: int = 4,
        time_error_floor_s: float = 0.1,
        latitude_min_deg: Optional[float] = None,
        latitude_max_deg: Optional[float] = None,
        longitude_min_deg: Optional[float] = None,
        longitude_max_deg: Optional[float] = None,
        log_level: int = INFO,
        log_file: Optional[str] = None,
    ):
        # Configure logger
        get_logger("hypy", level=log_level, log_file=log_file)

        # Read phases
        if not isinstance(phases, Sequence) or not all(
            isinstance(phase, str) for phase in phases
        ):
            raise ValueError(
                f"Phases must be an Sequence of strings, got {type(phases)}."
            )
        phases = tuple(normalize_phase(phase) for phase in phases)

        # Check if any phases provided in the input are not supported
        # by propagation engine.
        # Example: precalculated table have tables only for P and not for S.
        if any(phase not in propagation_engine.phases for phase in phases):
            raise ValueError(
                "Propagation engine phases ",
                f"{propagation_engine.phases} do not match provided phases {phases}.",
            )

        if initial_propagation_engine is not None and any(
            phase not in initial_propagation_engine.phases for phase in phases
        ):
            raise ValueError(
                "Initial propagation engine phases "
                f"{initial_propagation_engine.phases} do not match "
                f"provided phases {phases}."
            )

        # Global config
        self.config = LocatorConfig(
            phases=phases,
            min_picks_per_event=min_picks_per_event,
            time_error_floor_s=time_error_floor_s,
            derivative_method=derivative_method,
            fix_depth_km=fix_depth_km,
            latitude_min_deg=latitude_min_deg,
            latitude_max_deg=latitude_max_deg,
            longitude_min_deg=longitude_min_deg,
            longitude_max_deg=longitude_max_deg,
        )

        self.data = DataHandler(config=self.config)

        # Compare feature flags between derivative method and forward engine
        forward_engine_feature_flags = propagation_engine.feature_flags
        if (
            f"derivative.{self.config.derivative_method}"
            not in forward_engine_feature_flags
        ):
            raise ValueError(
                "Unsupported derivative method: "
                f"{self.config.derivative_method!r} "
                f"for {propagation_engine.__class__.__name__!r} backend."
            )

        self.propagation_backend = propagation_engine
        # Optional table for grid-search seeding; None => heuristic seeds.
        self.initial_propagation_backend = initial_propagation_engine

        # Derivative engine
        self.derivative_engine = DerivativesHandler(
            config=self.config,
            data=self.data,
            propagation_engine=self.propagation_backend,
        )

    def load_data(
        self,
        inv: Inventory,
        catalog: Union[List[Event], Catalog],
        station_time_corrections: Optional[pd.DataFrame] = None,
        fixed_depth_mask: Optional[Sequence[bool]] = None,
    ) -> None:
        """Load station and event data.

        Parameters
        ----------
        inv : Inventory
            ObsPy Inventory object containing station metadata.
        catalog : Union[List[Event], Catalog]
            ObsPy Catalog or list of Event objects containing event and pick data.
        station_time_corrections : mapping or table, optional
            Known station delays in seconds. A table (e.g. pandas DataFrame)
            must have columns ``network``, ``station``, ``delay_s`` and may
            have ``phase``. A mapping uses keys ``(network, station)`` or
            ``(network, station, phase)``. Positive delay means the observed
            arrival is late. Default is no corrections.
        fixed_depth_mask : sequence of bool, optional
            Per-event flags selecting input depths to hold fixed. Must contain
            one flag per catalog event. Default is no fixed event depths.
        """
        self.data.load_data(
            inv,
            catalog,
            station_time_corrections=station_time_corrections,
            fixed_depth_mask=fixed_depth_mask,
        )

    def locate(
        self,
        max_iterations: int = 100,
        conv_threshold: float = 1e-3,
        unresolved_depth: Optional[UnresolvedDepthLiteral] = "seed",
        depth_elimination_iteration: int = 20,
        trial_depth_km: float = 10.0,
        distance_weight_start_km: Optional[float] = None,
        distance_weight_end_km: Optional[float] = None,
        weight_start: Optional[float] = 0.99,
        weight_end: Optional[float] = 0.01,
        residual_weighting_iteration: Optional[int] = None,
        residual_rms_floor_s: float = 0.1,
        residual_full_weight_factor: float = 1.0,
        residual_zero_weight_factor: float = 3.0,
        full_data_covariance: bool = False,
        correlation_length_km: float = 1.0,
        model_uncertainty_s: float = 0.15,
        seed: Optional[int] = None,
        num_trials: int = 8,
        use_origins: bool = False,
    ) -> Tuple[Union[List[Event], Catalog], np.ndarray]:
        """Run the full location pipeline.

        Performs iterative least-squares inversion to locate events
        based on the loaded data and configuration. The process includes:
        1. Initial location setup (seed generation)
        2. Iterative inversion to refine event locations
        3. Error analysis to compute uncertainty ellipsoids
        4. Finalization of Origin objects and reporting of invalid picks/events

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of inversion iterations. Default is 1000.
        conv_threshold : float, optional
            WRMS convergence threshold (seconds). Default is 1e-3.
        unresolved_depth : {'current', 'seed', None}, optional
            Where to hold depth after an out-of-bounds step.
            ``current`` freezes the hypocenter in place; ``seed``
            (default) freezes at the initial location depth. ``None``
            skips that freeze and only eliminates depth after
            ``depth_elimination_iteration``.
        depth_elimination_iteration : int, optional
            Used only when ``unresolved_depth`` is ``None``: iteration
            after which a step below the model ``max_depth`` freezes
            depth in place. Default is 20.
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
        residual_weighting_iteration : Optional[int], optional
            1-based outer iteration at which HypoInverse-style residual weighting
            begins. ``None`` disables residual weighting. Default is None.
            Matching HypoInverse ``RMS 4 0.1 1 3`` uses
            ``residual_weighting_iteration=4``, ``residual_rms_floor_s=0.1``,
            ``residual_full_weight_factor=1``, ``residual_zero_weight_factor=3``.
            Residual weights are recomputed each iteration and do not
            permanently remove picks.
        residual_rms_floor_s : float, optional
            Floor (seconds) on the weighted RMS used to set residual-weight
            thresholds (HypoInverse ``RMSCUT``). Default is 0.1.
        residual_full_weight_factor : float, optional
            Full-weight residual threshold as a multiple of the floored RMS
            (HypoInverse ``RMSW1``). Default is 1.0.
        residual_zero_weight_factor : float, optional
            Zero-weight residual threshold as a multiple of the floored RMS
            (HypoInverse ``RMSW2``). Must be greater than
            ``residual_full_weight_factor``. Default is 3.0.
        full_data_covariance : bool, optional
            If True, compute event uncertainty using a correlated observation
            covariance model instead of diagonal row weighting. Default is False.
        correlation_length_km : float, optional
            Correlation length in km used in the exponential observation covariance
            model when full_data_covariance is enabled. Default is 1.0.
        model_uncertainty_s : float, optional
            Travel-time model uncertainty in seconds (HypoInverse ``RDERR``,
            set by its ``ERR`` command). It enters the data variance as
            ``SIGMA^2 = RDERR^2 + ERCOF * WRMS^2`` and sets the floor on the
            reported uncertainties: an event whose residuals fit perfectly
            still carries the error implied by an imperfect velocity model.
            This is a property of the model, not of the picks, so pick
            uncertainties only set the relative row weighting. Default is 0.15,
            the HypoInverse default; use a value comparable to the residual WRMS
            a well-located event achieves with your model. Must be > 0.
        seed : Optional[int], optional
            Seed for the random number generator used during location
            initialization. Set this to an integer to make this location run
            reproducible. If None, results are non-deterministic. Default is None.
        num_trials : int, optional
            Total number of heuristic seed candidates to evaluate, including
            the catalog Origin when ``use_origins`` is True, then the
            network-mean and early-P anchors. Extra trials are Gaussian
            samples around those anchors. ``num_trials=1`` with
            ``use_origins=True`` keeps only the Origin seed. Must be >= 1.
            Default is 8.
        use_origins : bool, optional
            When True, events that already contain an Origin use ``origins[-1]``
            as the first seed candidate. The origin's latitude, longitude,
            depth, and time (when present) participate in seeding; missing depth
            or time fall back to ``trial_depth_km`` and the earliest pick,
            respectively. Default is False. For catalog relocation over a
            regional network, set this True — heuristic seeds plus a finite
            ``max_distance_km`` on the tracer will otherwise drop distant
            stations and smear poorly constrained depths.

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
        # Build immutable, per-run configuration. RunConfig validates the
        # residual-weighting parameters whenever they are in effect.
        run_config = RunConfig(
            distance_weighting_enabled=distance_weighting_enabled,
            k=k,
            x0=x0,
            residual_weighting_enabled=residual_weighting_iteration is not None,
            residual_start_iteration=residual_weighting_iteration,
            residual_rms_floor_s=residual_rms_floor_s,
            residual_full_weight_factor=residual_full_weight_factor,
            residual_zero_weight_factor=residual_zero_weight_factor,
            full_data_covariance=bool(full_data_covariance),
            correlation_length_km=float(correlation_length_km),
            model_uncertainty_s=model_uncertainty_s,
            unresolved_depth=unresolved_depth,
            seed=seed,
            num_trials=num_trials,
            use_origins=bool(use_origins),
        )

        init_backend = self.initial_propagation_backend
        if init_backend is not None:
            init_depth_step_km = init_backend.depth_step_km
            init_max_depth_km = init_backend.max_depth_km
        else:
            init_depth_step_km = self.propagation_backend.depth_step_km
            init_max_depth_km = self.propagation_backend.max_depth_km
        seed_locations(
            config=self.config,
            data=self.data,
            run_config=run_config,
            trial_depth_km=trial_depth_km,
            depth_step_km=init_depth_step_km,
            max_depth_km=init_max_depth_km,
            propagation_engine=init_backend,
            scoring_engine=self.propagation_backend,
        )

        inversion_engine = InversionEngine(
            config=self.config,
            data=self.data,
            derivative_engine=self.derivative_engine,
            run_config=run_config,
        )
        inversion_results = inversion_engine.run_iterations(
            max_iterations=max_iterations,
            conv_threshold=conv_threshold,
            depth_elimination_iteration=depth_elimination_iteration,
        )

        error_results = compute_error_ellipsoids(
            self.config,
            self.data,
            inversion_results,
            run_config,
        )

        events = finalize_origins(self.config, self.data, error_results)
        invalid_indices = np.flatnonzero(self.data.invalid_events)

        return events, invalid_indices


__all__ = ["EventLocator"]
