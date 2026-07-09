"""Type declarations for locator components."""

from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
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

DimLiteral = Literal[4]
SolverLiteral = Literal["lsqr", "svd"]
DerivativeMethodLiteral = Literal["numerical", "analytical"]

VALID_SOLVERS: frozenset[str] = frozenset(get_args(SolverLiteral))
VALID_DERIVATIVE_METHODS: frozenset[str] = frozenset(get_args(DerivativeMethodLiteral))


@dataclass
class InversionResults:
    """Results returned by the iterative inversion step.

    Attributes
    ----------
    jacobian_matrix : numpy.ndarray
        Full jacobian matrix used in the inversion.
    valid_rows : numpy.ndarray
        Row indices that correspond to valid observations.
    residuals_flat : numpy.ndarray
        Flattened residual vector after weighting.
    """

    jacobian_matrix: np.ndarray  # full jacobian matrix
    valid_rows: np.ndarray  # indices of rows with valid observations
    residuals_flat: np.ndarray  # flattened residual vector


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
    "SolverLiteral",
    "DerivativeMethodLiteral",
    "VALID_DERIVATIVE_METHODS",
    "VALID_SOLVERS",
    "InversionResults",
    "ErrorResults",
    "INFO",
]
