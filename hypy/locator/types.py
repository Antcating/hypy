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
