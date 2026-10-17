"""Type declarations for locator components."""

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

DerivativeMethodLiteral = Literal["numerical", "analytical"]
UnresolvedDepthLiteral = Literal["current", "seed"]

VALID_DERIVATIVE_METHODS: frozenset[str] = frozenset(get_args(DerivativeMethodLiteral))
VALID_UNRESOLVED_DEPTHS: frozenset[str] = frozenset(get_args(UnresolvedDepthLiteral))

# Unknowns solved per event: latitude, longitude, depth, origin time.
DIM = 4


@dataclass
class InversionResults:
    """Results returned by the iterative inversion step."""

    jacobian_matrix: np.ndarray
    """Full jacobian matrix used in the inversion."""

    valid_rows: np.ndarray
    """Row indices that correspond to valid observations."""

    residuals_flat: np.ndarray
    """Flattened residual vector after weighting."""


@dataclass
class ErrorResults:
    """Results returned by the error-ellipsoid calculation."""

    spatial_covariances: np.ndarray
    """Spatial covariance matrices without the origin-time dimension.

    Shape ``(n_events, dim-1, dim-1)``.
    """

    principal_std: np.ndarray
    """Standard deviations along the principal spatial axes.

    Shape ``(n_events, dim-1)``.
    """

    principal_axes: np.ndarray
    """Eigenvectors defining the principal spatial axes.

    Shape ``(n_events, dim-1, dim-1)``.
    """

    erz_erh: np.ndarray
    """Vertical and horizontal error estimates for each event.

    Shape ``(n_events, 2)``.
    """
