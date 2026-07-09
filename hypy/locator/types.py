"""Type declarations for locator components."""

from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
)

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

if TYPE_CHECKING:
    from hypy.model import VelocityModel
    from hypy.propagation import PropagationEngine

DimLiteral = Literal[4]
ModeLiteral = Literal["table-1d", "ray-1d", "table-3d"]
EngineLiteral = Literal["taup", "pyrocko", "fmm"]
SolverLiteral = Literal["lsqr", "svd"]
DerivativeMethodLiteral = Literal["numerical", "analytical"]


class VelocityModelLike(Protocol):
    """Protocol describing model capability metadata used by locator wiring."""

    supported_features: frozenset[str]


class PropagationBackendLike(Protocol):
    """Protocol describing propagation capability metadata used by locator wiring."""

    default_feature_flags: ClassVar[frozenset[str]]
    default_required_model_features: ClassVar[frozenset[str]]


VelocityModelBackendType = Type["VelocityModel"]
PropagationBackendType = Type["PropagationEngine"]
VelocityModelFactory = Callable[[Path], "VelocityModel"]


@dataclass
class InversionResults:
    """Results returned by the iterative inversion step.

    Attributes
    ----------
    jacobian_matrix : numpy.ndarray
        Full jacobian matrix used in the inversion.
    jacobian_matrix_weighted : numpy.ndarray
        Full weighted jacobian matrix used in the inversion.
    valid_rows : numpy.ndarray
        Row indices that correspond to valid observations.
    residuals_flat : numpy.ndarray
        Flattened residual vector after weighting.
    damping : float
        Final damping factor chosen by the solver.
    """

    jacobian_matrix: np.ndarray  # full jacobian matrix
    jacobian_matrix_weighted: np.ndarray  # full weighted jacobian matrix
    valid_rows: np.ndarray  # indices of rows with valid observations
    residuals_flat: np.ndarray  # flattened residual vector
    residuals_flat_weighted: np.ndarray  # flattened weighted residual vector
    damping: float  # final damping factor


@dataclass
class ErrorResults:
    """Results returned by the error-ellipsoid calculation.

    Attributes
    ----------
    spatial_covariances : numpy.ndarray
        Spatial covariance matrices without the origin-time dimension.
    principal_std : numpy.ndarray
        Standard deviations along the principal spatial axes.
    principal_axes : numpy.ndarray
        Eigenvectors defining the principal spatial axes.
    erz_erh : numpy.ndarray
        Vertical and horizontal error estimates for each event.
    """

    spatial_covariances: np.ndarray  # (n_events, dim-1, dim-1)
    principal_std: np.ndarray  # (n_events, dim-1)
    principal_axes: np.ndarray  # (n_events, dim-1, dim-1)
    erz_erh: np.ndarray  # (n_events, 2)


@dataclass
class PropagationConfig:
    """Typed configuration payload for propagation backend initialization.

    Attributes
    ----------
    mode : ModeLiteral
        Propagation mode selected for the backend.
    engine : EngineLiteral
        Backend engine used to compute travel times.
    phases : tuple of str
        Seismic phases requested from the backend.
    model_path : pathlib.Path
        Path to the travel-time model or velocity model.
    max_depth : float or None
        Maximum modeled depth in kilometers.
    max_distance : float or None
        Maximum modeled epicentral distance in kilometers.
    depth_step : float or None
        Depth spacing used by table-based backends.
    distance_step : float or None
        Distance spacing used by table-based backends.
    """

    mode: ModeLiteral
    engine: EngineLiteral
    phases: Tuple[str, ...]
    model_path: Path
    max_depth: Optional[float]
    max_distance: Optional[float]
    depth_step: Optional[float]
    distance_step: Optional[float]


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
    "DimLiteral",
    "ModeLiteral",
    "EngineLiteral",
    "SolverLiteral",
    "DerivativeMethodLiteral",
    "InversionResults",
    "ErrorResults",
    "PropagationConfig",
    "VelocityModelLike",
    "PropagationBackendLike",
    "VelocityModelBackendType",
    "PropagationBackendType",
    "VelocityModelFactory",
    "INFO",
]
