"""Error covariance and ellipsoid analysis for hypy event locations."""

from typing import Optional

import numpy as np

from .config import LocatorConfig, RunConfig
from .data import DataHandler
from .types import ErrorResults, InversionResults


def filter_input_rows(
    jacobian_matrix: np.ndarray,
    residuals: np.ndarray,
    pick_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter observation rows with non-finite entries.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape (n_obs, dim)
        Unweighted jacobian matrix for one event.
    residuals : ndarray, shape (n_obs,)
        Residual vector for the same rows.
    pick_errors : ndarray, shape (n_obs,) or (k,)
        Pick timing uncertainty values.

    Returns
    -------
    tuple
        Filtered (jacobian_matrix, residuals, pick_errors)
        containing only finite rows.
    """
    row_mask = np.isfinite(residuals) & np.all(np.isfinite(jacobian_matrix), axis=1)
    if pick_errors.size == jacobian_matrix.shape[0]:
        row_mask &= np.isfinite(pick_errors)

    jacobian_matrix = jacobian_matrix[row_mask]
    residuals = residuals[row_mask]
    if pick_errors.size == row_mask.size:
        pick_errors = pick_errors[row_mask]
    else:
        pick_errors = pick_errors[np.isfinite(pick_errors)]

    return jacobian_matrix, residuals, pick_errors


def build_observation_covariance_from_stations(
    pick_errors: np.ndarray,
    station_indices: np.ndarray,
    phase_indices: np.ndarray,
    station_distances_km: np.ndarray,
    corr_length_km: float,
    min_pick_error: float = 1e-6,
    jitter_fraction: float = 1e-6,
) -> Optional[np.ndarray]:
    """Build correlated observation covariance using station-pair distances."""
    pick_errors = np.maximum(
        np.asarray(pick_errors, dtype=float).reshape(-1),
        float(min_pick_error),
    )
    station_indices = np.asarray(station_indices, dtype=int).reshape(-1)
    phase_indices = np.asarray(phase_indices, dtype=int).reshape(-1)
    station_distances_km = np.asarray(station_distances_km, dtype=float)

    n_obs = pick_errors.size
    if n_obs == 0 or station_indices.size != n_obs or phase_indices.size != n_obs:
        return None
    if corr_length_km <= 0.0:
        return None
    if station_distances_km.ndim != 2 or (
        station_distances_km.shape[0] != station_distances_km.shape[1]
    ):
        return None
    if np.any(station_indices < 0) or np.any(
        station_indices >= station_distances_km.shape[0]
    ):
        return None

    d = station_distances_km[np.ix_(station_indices, station_indices)]
    R = np.exp(-d / corr_length_km)
    same_phase = phase_indices[:, None] == phase_indices[None, :]
    R = np.where(same_phase, R, 0.0)
    np.fill_diagonal(R, 1.0)

    sigma = (pick_errors[:, None] * pick_errors[None, :]) * R
    jitter = float(jitter_fraction) * float(np.nanmedian(pick_errors**2))
    sigma += jitter * np.eye(n_obs, dtype=float)
    return 0.5 * (sigma + sigma.T)


def data_variance(
    residuals: np.ndarray,
    pick_errors: np.ndarray,
    error_coefficient: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate HYPOINVERSE-style arrival-time variance.

    HYPOINVERSE computes the variance scale used for formal location errors as
    ``SIGMA^2 = RDERR^2 + (ERCOF * RMS)^2``.  Here ``RDERR`` is represented by
    the median pick timing uncertainty for the event.
    """
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    pick_errors = np.asarray(pick_errors, dtype=float).reshape(-1)
    pick_errors = pick_errors[np.isfinite(pick_errors)]

    if residuals.size == 0 or pick_errors.size == 0:
        return np.nan, np.nan, np.nan

    rms = float(np.sqrt(np.mean(residuals**2)))
    rderr = float(np.nanmedian(np.maximum(pick_errors, 1e-6)))
    ercof = float(error_coefficient)
    sigma2 = rderr**2 + (ercof * rms) ** 2
    return float(sigma2), rms, rderr


def whiten_design(
    jacobian_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    reference_variance: float,
    eigenvalue_tolerance: float = 1e-12,
) -> Optional[np.ndarray]:
    """Whiten the design matrix with relative observation covariance."""
    if reference_variance <= 0.0:
        return None

    relative_covariance = np.asarray(observation_covariance, dtype=float)
    relative_covariance = 0.5 * (relative_covariance + relative_covariance.T)
    relative_covariance = relative_covariance / float(reference_variance)

    try:
        evals, evecs = np.linalg.eigh(relative_covariance)
    except np.linalg.LinAlgError:
        return None

    positive = evals > float(eigenvalue_tolerance)
    if not np.any(positive):
        return None

    inv_sqrt = np.zeros_like(evals, dtype=float)
    inv_sqrt[positive] = 1.0 / np.sqrt(evals[positive])
    whitening = (evecs * inv_sqrt) @ evecs.T
    whitened_design = whitening @ jacobian_matrix

    if not np.all(np.isfinite(whitened_design)):
        return None

    return whitened_design


def covariance_from_design(
    design_matrix: np.ndarray,
    sigma2: float,
    eigenvalue_tolerance: float = 1e-12,
) -> Optional[np.ndarray]:
    """Compute HYPOINVERSE-style covariance from SVD of the design matrix."""
    if sigma2 <= 0.0 or not np.isfinite(sigma2):
        return None

    try:
        _, singular_values, vt = np.linalg.svd(design_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if singular_values.size == 0:
        return None

    threshold = float(eigenvalue_tolerance) * float(np.max(singular_values))
    resolved = singular_values > threshold
    if not np.any(resolved):
        return None

    inverse_squares = np.zeros_like(singular_values, dtype=float)
    inverse_squares[resolved] = 1.0 / (singular_values[resolved] ** 2)
    covariance = float(sigma2) * ((vt.T * inverse_squares) @ vt)
    return 0.5 * (covariance + covariance.T)


def ellipsoid_axes(
    Cxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Derive sorted principal axes and ERZ/ERH from spatial covariance."""
    Cxyz = 0.5 * (Cxyz + Cxyz.T)

    evals, evecs = np.linalg.eigh(Cxyz)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    std_axes = np.sqrt(np.clip(evals, 0.0, None))

    if Cxyz.shape[0] == 3:
        ex, ey, ez = evecs[0, :], evecs[1, :], evecs[2, :]
        proj_vert = np.abs(ez) * std_axes
        proj_horiz = np.sqrt(ex**2 + ey**2) * std_axes
        return std_axes, evecs, float(np.max(proj_vert)), float(np.max(proj_horiz))

    if Cxyz.shape[0] == 2:
        return std_axes, evecs, np.nan, float(np.max(std_axes))

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
    observation_covariance: Optional[np.ndarray] = None,
    error_coefficient: float = 1.0,
) -> Optional[dict]:
    """Compute error covariance and ellipsoid metrics for one event.

    Parameters
    ----------
    jacobian_matrix : ndarray, shape (n_obs, dim)
        Event-specific unweighted jacobian matrix rows used in inversion.
    residuals : ndarray, shape (n_obs,)
        True residuals for the same rows.
    pick_errors : ndarray, shape (n_obs,) or (m,)
        Pick timing uncertainties for those rows.
    dim : int
        Number of model parameters (e.g., 4 for x,y,z,t0).

    Returns
    -------
    dict or None
        Computed error measures, or None when not enough valid data.
    """
    if (
        jacobian_matrix.ndim != 2
        or jacobian_matrix.shape[1] != dim
        or jacobian_matrix.shape[0] == 0
    ):
        return None

    if residuals.shape[0] != jacobian_matrix.shape[0]:
        return None

    jacobian_matrix, residuals, pick_errors = filter_input_rows(
        jacobian_matrix, residuals, pick_errors
    )

    if jacobian_matrix.shape[0] == 0:
        return None

    if pick_errors.size != residuals.size:
        return None

    sigma2, rms, rderr = data_variance(
        residuals=residuals,
        pick_errors=pick_errors,
        error_coefficient=error_coefficient,
    )
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return None

    if observation_covariance is None:
        observation_covariance = np.diag(np.maximum(pick_errors, 1e-6) ** 2)
    else:
        observation_covariance = np.asarray(observation_covariance, dtype=float)
        if observation_covariance.shape != (
            jacobian_matrix.shape[0],
            jacobian_matrix.shape[0],
        ):
            return None

    design_matrix = whiten_design(
        jacobian_matrix=jacobian_matrix,
        observation_covariance=observation_covariance,
        reference_variance=max(rderr**2, 1e-12),
    )
    if design_matrix is None:
        return None

    covariance_full = covariance_from_design(
        design_matrix=design_matrix,
        sigma2=sigma2,
    )
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
        "RMS_s": rms,
        "RDERR_s": rderr,
        "SIGMA2_s2": sigma2,
        "n_obs_used": int(jacobian_matrix.shape[0]),
    }


def validate_inversion_results(
    config: LocatorConfig,
    data: DataHandler,
    inversion_results: InversionResults,
) -> None:
    """Verify the global inversion outputs are correctly shaped.

    Parameters
    ----------
    inversion_results : InversionResults
        Inversion outputs specifically A_final, valid_rows_final,
        and residuals_flat_final.

    Raises
    ------
    ValueError
        If A_final or residuals_flat_final are malformed or incompatible
        with the expected dimensions from the configuration.
    """
    jacobian_matrix = np.asarray(inversion_results.jacobian_matrix, dtype=float)
    residuals_flat = np.asarray(inversion_results.residuals_flat, dtype=float)
    residuals_flat = residuals_flat.reshape(-1)

    if jacobian_matrix.ndim != 2:
        raise ValueError("inversion_results.A_final must be a 2D array")
    if residuals_flat.ndim != 1:
        raise ValueError("inversion_results.residuals_flat_final must be a 1D array")
    if residuals_flat.shape[0] != jacobian_matrix.shape[0]:
        raise ValueError(
            f"residuals_flat_final has length {residuals_flat.shape[0]} but "
            f"A_final has {jacobian_matrix.shape[0]} rows"
        )

    dim = int(config.dim)
    expected_cols = data.n_events * dim
    if jacobian_matrix.shape[1] != expected_cols:
        raise ValueError(
            "A_final has {ncols} columns but expected {exp} "
            "(n_events={n_events}, dim={dim})".format(
                ncols=jacobian_matrix.shape[1],
                exp=expected_cols,
                n_events=data.n_events,
                dim=dim,
            )
        )


def aggregate_event_rows(
    config: LocatorConfig,
    data: DataHandler,
    inversion_results: InversionResults,
) -> tuple[list[list[int]], list[list[float]], list[list[int]], list[list[int]]]:
    """Group valid global observation rows into per-event inputs."""
    n_obs_per_event = data.n_stations * config.n_phases

    valid_rows_original = np.asarray(inversion_results.valid_rows, dtype=int).reshape(
        -1
    )

    n_jacobian_rows = inversion_results.jacobian_matrix.shape[0]

    rows_per_event: list[list[int]] = [[] for _ in range(data.n_events)]
    pick_errors_per_event: list[list[float]] = [[] for _ in range(data.n_events)]
    station_indices_per_event: list[list[int]] = [[] for _ in range(data.n_events)]
    phase_indices_per_event: list[list[int]] = [[] for _ in range(data.n_events)]

    n_err_cols = data.arrival_time_errors.shape[1]

    for original_row in valid_rows_original:
        if original_row < 0 or original_row >= n_jacobian_rows:
            continue

        ev = int(original_row) // n_obs_per_event
        if ev < 0 or ev >= data.n_events:
            continue

        local_row = int(original_row) - ev * n_obs_per_event
        if local_row < 0 or local_row >= n_err_cols:
            continue

        rows_per_event[ev].append(int(original_row))
        pick_errors_per_event[ev].append(float(data.arrival_time_errors[ev, local_row]))
        station_indices_per_event[ev].append(int(local_row // config.n_phases))
        phase_indices_per_event[ev].append(int(local_row % config.n_phases))

    return (
        rows_per_event,
        pick_errors_per_event,
        station_indices_per_event,
        phase_indices_per_event,
    )


def compute_error_ellipsoids(
    config: LocatorConfig,
    data: DataHandler,
    inversion_results: InversionResults,
    run_config: RunConfig,
) -> ErrorResults:
    """Compute error ellipsoids for all events in the inversion result.

    Parameters
    ----------
    inversion_results : InversionResults
        Outputs of the location inversion step.
    run_config : RunConfig
        Per-run configuration providing the observation error model
        (``full_data_covariance``, ``correlation_length_km``, ``error_coefficient``).

    Returns
    -------
    ErrorResults
        Spatial covariances, principal standard deviations, principal axes,
        and ERZ/ERH values.
    """
    validate_inversion_results(config, data, inversion_results)

    use_full_covariance = run_config.full_data_covariance
    jacobian_matrix = inversion_results.jacobian_matrix
    residuals_flat = inversion_results.residuals_flat.reshape(-1)

    dim = int(config.dim)
    spatial_dim = min(3, dim - 1 if dim > 1 else dim)

    spatial_covariances = np.zeros(
        (data.n_events, spatial_dim, spatial_dim), dtype=float
    )
    principal_std = np.zeros((data.n_events, spatial_dim), dtype=float)
    principal_axes = np.zeros((data.n_events, spatial_dim, spatial_dim), dtype=float)
    erz_erh = np.full((data.n_events, 2), np.nan, dtype=float)

    (
        rows_per_event,
        pick_errors_per_event,
        station_indices_per_event,
        phase_indices_per_event,
    ) = aggregate_event_rows(config, data, inversion_results)

    for ev in range(data.n_events):
        if data.invalid_events[ev]:
            continue

        rows = np.asarray(rows_per_event[ev], dtype=int)
        if rows.size == 0:
            continue

        event_jacobian_matrix = jacobian_matrix[rows, ev * dim : (ev + 1) * dim]
        event_residuals = residuals_flat[rows]
        event_pick_errors = np.asarray(pick_errors_per_event[ev], dtype=float)
        observation_covariance = None

        if use_full_covariance:
            station_indices = np.asarray(station_indices_per_event[ev], dtype=int)
            phase_indices = np.asarray(phase_indices_per_event[ev], dtype=int)
            observation_covariance = build_observation_covariance_from_stations(
                pick_errors=event_pick_errors,
                station_indices=station_indices,
                phase_indices=phase_indices,
                station_distances_km=np.asarray(data.station_distances_km, dtype=float),
                corr_length_km=run_config.correlation_length_km,
            )
            if observation_covariance is None:
                continue

        errinfo = compute_event_errors(
            jacobian_matrix=event_jacobian_matrix,
            residuals=event_residuals,
            pick_errors=event_pick_errors,
            dim=dim,
            observation_covariance=observation_covariance,
            error_coefficient=run_config.error_coefficient,
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
