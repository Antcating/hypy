from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from obspy.core.event import (  # type: ignore[import-untyped]
    Arrival,
    Catalog,
    Event,
    Origin,
)
from obspy.core.event.base import QuantityError  # type: ignore[import-untyped]
from obspy.core.event.origin import (  # type: ignore[import-untyped]
    ConfidenceEllipsoid,
    OriginUncertainty,
)
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from typing_extensions import Literal

DimLiteral = Literal[4]
ModeLiteral = Literal["table", "ray"]
EngineLiteral = Literal["taup", "pyrocko"]
SolverLiteral = Literal["lsqr", "svd"]


@dataclass(slots=True)
class StationsInfo:
    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    n_stations: int


@dataclass(slots=True)
class EventMeta:
    events: List[Event]
    n_events: int
    fixed_depth_mask: List[bool]


@dataclass(slots=True)
class Matrices:
    arrivals: np.ndarray
    arrival_time_errors: np.ndarray
    theoretical_travel_times: np.ndarray
    derivative_matrix: np.ndarray
    locations: np.ndarray
    origin_times: np.ndarray
    time_refs: np.ndarray
    invalid_events: np.ndarray


@dataclass(slots=True)
class Weights:
    time_error_W: np.ndarray
    distance_W: np.ndarray


# ---- Small DTOs (data transfer objects)


@dataclass(slots=True)
class DerivativeContext:
    """Parameters required to compute travel-time derivatives.

    Keeping this as a small dataclass makes internal APIs stable and keeps
    function signatures short.
    """

    tracer: object
    stations: list
    stations_locations: np.ndarray
    phases: Sequence[str]
    n_events: int
    n_stations: int
    n_phases: int
    dim: int
    fixed_depth_mask: list[bool]
    distance_weighting: bool
    k: Optional[float]
    x0: Optional[float]


@dataclass(slots=True)
class IterationConfig:
    """Configuration for the inversion loop."""

    solver: SolverLiteral
    max_iterations: int
    conv_threshold: float
    solve_airquakes: bool
    max_depth: float
    init_phase_iteration_max: int
    mode: str
    fix_depth: bool
    precalculate_velocities: bool
    model_radius_of_planet_km: float


@dataclass(slots=True)
class InitConfig:
    """Configuration for initial hypocenter/origin time seeds."""

    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    phases: Sequence[str]
    n_stations: int
    n_phases: int
    min_picks_per_event: int
    max_depth: Optional[float]
    trial_depth: float
    trial_time: float


@dataclass(slots=True)
class FinalizeContext:
    """Inputs required to attach final Origin objects to events."""

    stations: list
    phases: Sequence[str]
    n_phases: int
    fixed_depth_mask: List[bool]


@dataclass(slots=True)
class FillDataConfig:
    """Configuration for converting picks into relative arrival matrices."""

    stations: List[Tuple[str, str]]
    phases: Sequence[str]
    n_stations: int
    n_phases: int
    min_picks_per_event: int
    time_error_floor: float


@dataclass(slots=True)
class FinalizeUncertainty:
    principal_std: np.ndarray
    principal_axes: np.ndarray
    erz_erh: np.ndarray


# Re-exports for convenience
__all__ = [
    "Path",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "Literal",
    "Sequence",
    "np",
    "Arrival",
    "Catalog",
    "Event",
    "Origin",
    "QuantityError",
    "ConfidenceEllipsoid",
    "OriginUncertainty",
    "Inventory",
    "TauPyModel",
    "DimLiteral",
    "ModeLiteral",
    "EngineLiteral",
    "DerivativeContext",
    "IterationConfig",
    "InitConfig",
    "FinalizeContext",
    "FillDataConfig",
    "FinalizeUncertainty",
]
