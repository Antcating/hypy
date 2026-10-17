"""Error covariance and ellipsoid analysis for event locations."""

from typing import Optional

import numpy as np

from ..utils.geo import gps2dist_azimuth_batch_nb
from .config import LocatorConfig, RunConfig
from .data import DataHandler
from .types import DIM, ErrorResults, InversionResults


def build_observation_covariance_from_stations(
    pick_errors: np.ndarray,
    station_indices: np.ndarray,
    phase_indices: np.ndarray,
    station_distances_km: np.ndarray,
    corr_length_km: float,
    min_pick_error: float = 1e-6,
    jitter_fraction: float = 1e-6,
) -> Optional[np.ndarray]:
    """Build correlated observation covariance from station-pair distances."""
    pick_errors = np.maximum(
        np.asarray(pick_errors, dtype=float).reshape(-1), float(min_pick_error)
    )
    station_indices = np.asarray(station_indices, dtype=int).reshape(-1)
    phase_indices = np.asarray(phase_indices, dtype=int).reshape(-1)
    station_distances_km = np.asarray(station_distances_km, dtype=float)
    n_obs = pick_errors.size
    if (
        n_obs == 0
        or station_indices.size != n_obs
        or phase_indices.size != n_obs
        or corr_length_km <= 0.0
        or station_distances_km.ndim != 2
        or station_distances_km.shape[0] != station_distances_km.shape[1]
        or np.any(station_indices < 0)
        or np.any(station_indices >= station_distances_km.shape[0])
    ):
        return None

    d = station_distances_km[np.ix_(station_indices, station_indices)]
    R = np.exp(-d / corr_length_km)
    R = np.where(phase_indices[:, None] == phase_indices[None, :], R, 0.0)
    np.fill_diagonal(R, 1.0)
    sigma = (pick_errors[:, None] * pick_errors[None, :]) * R
    sigma += (
        float(jitter_fraction)
        * float(np.nanmedian(pick_errors**2))
        * np.eye(n_obs, dtype=float)
    )
    return 0.5 * (sigma + sigma.T)


def data_variance(
    residuals: np.ndarray,
    model_uncertainty_s: float,
    error_coefficient: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """HYPOINVERSE ``SIGMA^2 = RDERR^2 + ERCOF * WRMS^2``.

    ``weights`` are least-squares row scales: ``sqrt(sum((w r)^2) / sum(w^2))``.
    Omit them for an unweighted RMS. A zero weight drops that residual.
    ``model_uncertainty_s`` is RDERR, a property of the velocity model.
    """
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    ok = np.isfinite(residuals)
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape != residuals.shape:
            return np.nan, np.nan
        ok &= np.isfinite(weights) & (weights != 0.0)
        weights = weights[ok]
    residuals = residuals[ok]
    if residuals.size == 0 or not np.isfinite(model_uncertainty_s):
        return np.nan, np.nan
    if weights is None:
        wrms = float(np.sqrt(np.mean(residuals**2)))
    else:
        denom = float(np.dot(weights, weights))
        wrms = (
            float(np.sqrt(np.dot(weights**2, residuals**2) / denom))
            if denom > 0.0
            else np.nan
        )
    if not np.isfinite(wrms):
        return np.nan, np.nan
    return float(model_uncertainty_s**2 + float(error_coefficient) * wrms**2), wrms


def whiten_rows(
    jacobian_matrix: np.ndarray,
    variances: np.ndarray,
    reference_variance: float,
    eigenvalue_tolerance: float = 1e-12,
) -> Optional[np.ndarray]:
    """Scale each row by ``1/sqrt`` of its variance relative to the reference.

    This is the uncorrelated case, where whitening is a per-row rescale and
    no eigendecomposition is needed. Returns ``None`` when no row carries a
    usable variance.
    """
    relative = np.asarray(variances, dtype=float) / float(reference_variance)
    positive = relative > float(eigenvalue_tolerance)
    if not np.any(positive):
        return None
    scales = np.zeros(relative.shape, dtype=float)
    scales[positive] = 1.0 / np.sqrt(relative[positive])
    whitened = jacobian_matrix * scales[:, np.newaxis]
    return whitened if np.all(np.isfinite(whitened)) else None


def whiten_design(
    jacobian_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    reference_variance: float,
    eigenvalue_tolerance: float = 1e-12,
) -> Optional[np.ndarray]:
    """Whiten the design matrix with relative observation covariance."""
    if reference_variance <= 0.0:
        return None
    observation_covariance = np.asarray(observation_covariance, dtype=float)
    tol = float(eigenvalue_tolerance)
    if (
        observation_covariance.ndim == 2
        and observation_covariance.shape[0] == observation_covariance.shape[1]
    ):
        diag = np.diag(observation_covariance)
        if np.allclose(observation_covariance, np.diag(diag)):
            whitened = whiten_rows(jacobian_matrix, diag, reference_variance, tol)
            if whitened is not None:
                return whitened

    relative = 0.5 * (observation_covariance + observation_covariance.T)
    relative = relative / float(reference_variance)
    try:
        evals, evecs = np.linalg.eigh(relative)
    except np.linalg.LinAlgError:
        return None
    positive = evals > tol
    if not np.any(positive):
        return None
    inv_sqrt = np.zeros_like(evals, dtype=float)
    inv_sqrt[positive] = 1.0 / np.sqrt(evals[positive])
    whitened = ((evecs * inv_sqrt) @ evecs.T) @ jacobian_matrix
    if not np.all(np.isfinite(whitened)):
        return None
    return whitened


def covariance_from_design(
    design_matrix: np.ndarray,
    sigma2: float,
    eigenvalue_tolerance: float = 1e-12,
) -> Optional[np.ndarray]:
    """Formal covariance from a weighted design; zero-columns stay fixed."""
    design_matrix = np.asarray(design_matrix, dtype=float)
    if (
        design_matrix.ndim != 2
        or design_matrix.shape[0] == 0
        or sigma2 <= 0.0
        or not np.isfinite(sigma2)
        or not np.all(np.isfinite(design_matrix))
    ):
        return None

    col_norms = np.linalg.norm(design_matrix, axis=0)
    max_col_norm = float(np.max(col_norms)) if col_norms.size else 0.0
    active = col_norms > max(float(eigenvalue_tolerance) * max_col_norm, 1e-15)
    if not np.any(active):
        return None

    active_design = design_matrix[:, active]
    try:
        _, singular_values, vt = np.linalg.svd(active_design, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    n_active = active_design.shape[1]
    if singular_values.size < n_active or singular_values.size == 0:
        return None
    threshold = float(eigenvalue_tolerance) * float(np.max(singular_values))
    if np.count_nonzero(singular_values > threshold) < n_active:
        return None

    active_covariance = float(sigma2) * ((vt.T * (1.0 / singular_values**2)) @ vt)
    covariance = np.zeros((design_matrix.shape[1], design_matrix.shape[1]), dtype=float)
    covariance[np.ix_(active, active)] = active_covariance
    return 0.5 * (covariance + covariance.T)


def ellipsoid_axes(Cxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Principal axes plus ERZ/ERH. Coordinates are ``(north, east, down)`` km."""
    Cxyz = np.asarray(Cxyz, dtype=float)
    Cxyz = 0.5 * (Cxyz + Cxyz.T)
    evals, evecs = np.linalg.eigh(Cxyz)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    std_axes = np.sqrt(np.clip(evals, 0.0, None))
    if Cxyz.shape[0] == 3:
        erz = float(np.sqrt(max(float(Cxyz[2, 2]), 0.0)))
        h_evals = np.linalg.eigvalsh(0.5 * (Cxyz[:2, :2] + Cxyz[:2, :2].T))
        return std_axes, evecs, erz, float(np.sqrt(max(float(np.max(h_evals)), 0.0)))
    if Cxyz.shape[0] == 2:
        return std_axes, evecs, np.nan, float(std_axes[0])
    return (
        std_axes,
        evecs,
        np.nan,
        float(std_axes[0]) if std_axes.size else np.nan,
    )


def compute_event_errors(
    jacobian_matrix: np.ndarray,
    residuals: np.ndarray,
    pick_errors: np.ndarray,
    dim: int,
    model_uncertainty_s: float = 0.15,
    observation_covariance: Optional[np.ndarray] = None,
    error_coefficient: float = 1.0,
    additional_row_scales: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Covariance and ellipsoid metrics for one event, or ``None`` if unusable."""
    if (
        jacobian_matrix.ndim != 2
        or jacobian_matrix.shape[1] != dim
        or jacobian_matrix.shape[0] == 0
        or residuals.shape[0] != jacobian_matrix.shape[0]
    ):
        return None

    n_obs = jacobian_matrix.shape[0]
    pick_errors = np.asarray(pick_errors, dtype=float).reshape(-1)
    if pick_errors.size != n_obs:
        return None

    row_mask = (
        np.isfinite(residuals)
        & np.all(np.isfinite(jacobian_matrix), axis=1)
        & np.isfinite(pick_errors)
    )

    if additional_row_scales is not None:
        additional_row_scales = np.asarray(additional_row_scales, dtype=float).reshape(
            -1
        )

        if additional_row_scales.size != n_obs:
            return None

        # Zero weight means the observation is excluded.
        row_mask &= np.isfinite(additional_row_scales) & (additional_row_scales > 0.0)

    if observation_covariance is not None:
        observation_covariance = np.asarray(observation_covariance, dtype=float)
        if observation_covariance.shape != (n_obs, n_obs):
            return None

    if not np.any(row_mask):
        return None

    jacobian_matrix = jacobian_matrix[row_mask]
    residuals = residuals[row_mask]
    pick_errors = pick_errors[row_mask]

    if additional_row_scales is not None:
        additional_row_scales = additional_row_scales[row_mask]

    if observation_covariance is not None:
        observation_covariance = observation_covariance[np.ix_(row_mask, row_mask)]

    # Same row scales as the design: distance/residual weights, then 1/pick_error.
    row_scales = 1.0 / np.maximum(pick_errors, 1e-6)
    if additional_row_scales is not None:
        row_scales *= additional_row_scales
    sigma2, wrms = data_variance(
        residuals, model_uncertainty_s, error_coefficient, row_scales
    )
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return None

    reference_variance = max(float(model_uncertainty_s) ** 2, 1e-12)

    # Apply distance/residual weights before covariance whitening.
    if additional_row_scales is None:
        weighted_jacobian = jacobian_matrix
    else:
        weighted_jacobian = jacobian_matrix * additional_row_scales[:, np.newaxis]

    if observation_covariance is None:
        design_matrix = whiten_rows(
            weighted_jacobian,
            np.maximum(pick_errors, 1e-6) ** 2,
            reference_variance,
        )
    else:
        design_matrix = whiten_design(
            weighted_jacobian,
            observation_covariance,
            reference_variance,
        )

    if design_matrix is None:
        return None

    covariance_full = covariance_from_design(design_matrix, sigma2)
    if covariance_full is None:
        return None

    spatial_dim = min(3, dim - 1 if dim > 1 else dim)
    if spatial_dim <= 0:
        return None

    C_xyz = covariance_full[:spatial_dim, :spatial_dim]
    if not np.all(np.isfinite(C_xyz)):
        return None

    try:
        principal_std, principal_axes, erz, erh = ellipsoid_axes(C_xyz)
    except np.linalg.LinAlgError:
        return None

    return {
        "C_xyz": C_xyz,
        "principal_std_km_1sigma": principal_std,
        "principal_axes_unit_vectors": principal_axes,
        "ERZ_km_1sigma": erz,
        "ERH_km_1sigma": erh,
        "WRMS_s": wrms,
        "SIGMA2_s2": sigma2,
        "n_obs_used": int(jacobian_matrix.shape[0]),
    }


def _station_distance_matrix(data: DataHandler) -> np.ndarray:
    """Return the inter-station distance matrix in kilometers."""
    locations = np.asarray(data.stations_locations, dtype=float)
    n_sta = int(locations.shape[0])
    distances = np.zeros((n_sta, n_sta), dtype=float)
    lats, lons = locations[:, 0], locations[:, 1]
    for i in range(n_sta - 1):
        dist_m, _, _ = gps2dist_azimuth_batch_nb(
            float(lats[i]), float(lons[i]), lats[i + 1 :], lons[i + 1 :]
        )
        km = dist_m * 1e-3
        distances[i, i + 1 :] = km
        distances[i + 1 :, i] = km
    return distances


def compute_error_ellipsoids(
    config: LocatorConfig,
    data: DataHandler,
    inversion_results: InversionResults,
    run_config: RunConfig,
) -> ErrorResults:
    """Compute spatial covariances and ERZ/ERH for every event."""
    jacobian_matrix = np.asarray(inversion_results.jacobian_matrix, dtype=float)
    residuals_flat = np.asarray(inversion_results.residuals_flat, dtype=float).reshape(
        -1
    )
    if jacobian_matrix.ndim != 2:
        raise ValueError("inversion_results.A_final must be a 2D array")
    if residuals_flat.shape[0] != jacobian_matrix.shape[0]:
        raise ValueError(
            f"residuals_flat_final has length {residuals_flat.shape[0]} but "
            f"A_final has {jacobian_matrix.shape[0]} rows"
        )
    dim = DIM
    expected_cols = data.n_events * dim
    if jacobian_matrix.shape[1] != expected_cols:
        raise ValueError(
            f"A_final has {jacobian_matrix.shape[1]} columns but expected "
            f"{expected_cols} (n_events={data.n_events}, dim={dim})"
        )

    spatial_dim = min(3, dim - 1 if dim > 1 else dim)
    spatial_covariances = np.zeros(
        (data.n_events, spatial_dim, spatial_dim), dtype=float
    )
    principal_std = np.zeros((data.n_events, spatial_dim), dtype=float)
    principal_axes = np.zeros((data.n_events, spatial_dim, spatial_dim), dtype=float)
    erz_erh = np.full((data.n_events, 2), np.nan, dtype=float)

    rows = np.asarray(inversion_results.valid_rows, dtype=int)
    event_idx, station_idx, phase_idx = np.unravel_index(
        rows, (data.n_events, data.n_stations, config.n_phases)
    )
    pick_errors = data.arrival_time_errors.reshape(
        data.n_events, data.n_stations, config.n_phases
    )[event_idx, station_idx, phase_idx]
    station_distances_km = (
        _station_distance_matrix(data) if run_config.full_data_covariance else None
    )

    for ev in range(data.n_events):
        if data.invalid_events[ev]:
            continue
        sel = event_idx == ev
        if not np.any(sel):
            continue
        event_rows = rows[sel]
        observation_covariance = None
        if run_config.full_data_covariance:
            observation_covariance = build_observation_covariance_from_stations(
                pick_errors[sel],
                station_idx[sel],
                phase_idx[sel],
                station_distances_km,
                run_config.correlation_length_km,
            )
            if observation_covariance is None:
                continue
        errinfo = compute_event_errors(
            jacobian_matrix=jacobian_matrix[event_rows, ev * dim : (ev + 1) * dim],
            residuals=residuals_flat[event_rows],
            pick_errors=pick_errors[sel],
            dim=dim,
            model_uncertainty_s=run_config.model_uncertainty_s,
            observation_covariance=observation_covariance,
            error_coefficient=run_config.error_coefficient,
            additional_row_scales=np.asarray(data.distance_W[event_rows], dtype=float)
            * np.asarray(data.residual_W[event_rows], dtype=float),
        )
        if errinfo is None:
            continue
        spatial_covariances[ev] = errinfo["C_xyz"]
        principal_std[ev] = errinfo["principal_std_km_1sigma"]
        principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
        erz_erh[ev] = [errinfo["ERZ_km_1sigma"], errinfo["ERH_km_1sigma"]]

    return ErrorResults(
        spatial_covariances=spatial_covariances,
        principal_std=principal_std,
        principal_axes=principal_axes,
        erz_erh=erz_erh,
    )
