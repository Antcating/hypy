from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
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
from obspy.taup import TauPyModel  # type: ignore[import-untyped]
from typing_extensions import Literal

DimLiteral = Literal[4]
ModeLiteral = Literal["table", "ray"]
EngineLiteral = Literal["taup", "pyrocko"]
SolverLiteral = Literal["lsqr", "svd"]


@dataclass
class InversionResults:
    """Output of the iterative inversion step."""

    A_final: np.ndarray  # full weighted design matrix
    valid_rows_final: np.ndarray  # indices of rows with valid observations
    residuals_flat_final: np.ndarray  # flattened residual vector
    lambda_final: float  # final damping factor


@dataclass
class ErrorResults:
    """Output of error ellipsoid computation."""

    covariances: np.ndarray  # (n_events, dim, dim)
    spatial_covariances: np.ndarray  # (n_events, dim-1, dim-1)
    principal_std: np.ndarray  # (n_events, dim-1)
    principal_axes: np.ndarray  # (n_events, dim-1, dim-1)
    erz_erh: np.ndarray  # (n_events, 2)


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
    "InversionResults",
    "ErrorResults",
]
