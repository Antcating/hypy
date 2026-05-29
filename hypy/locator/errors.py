"""Error covariance and ellipsoid analysis for hypy event locations."""

from typing import Optional

import numpy as np

from ..utils.weighting import time_error_weight
from .config import LocatorConfig
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


class ErrorAnalyzer:
    """Calculate event location covariance and ellipsoid metrics.

    Parameters
    ----------
    config : LocatorConfig
        Locator configuration containing dimension and phase settings.
    data : DataHandler
        Data handler holding per-event station, pick and timing-error arrays.
    """

    def __init__(self, config: LocatorConfig, data: DataHandler) -> None:
        self.config = config
        self.data = data

    @staticmethod
    def _build_observation_covariance_from_stations(
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

    @staticmethod
    def _compute_sigma2(
        residuals: np.ndarray,
        n_params: int,
    ) -> float:
        """Estimate residual variance with degrees-of-freedom correction.

        If residuals_are_weighted=True, sigma^2 is dimensionless and should not
        add pick-error variance again.
        """
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]

        if residuals.size == 0:
            return np.nan

        dof = residuals.size - int(n_params)
        if dof <= 0:
            return np.nan

        rss = float(np.sum(residuals**2))
        return rss / dof

    @staticmethod
    def _regularized_inverse(hessian: np.ndarray) -> Optional[np.ndarray]:
        """Compute inverse of a Hessian/normal matrix.

        Parameters
        ----------
        hessian : ndarray, shape (dim, dim)
            Gauss-Newton Hessian/normal matrix.

        Returns
        -------
        ndarray or None
            Inverse of H + damping**2 I, or pseudo-inverse if singular.
        """
        hessian = np.asarray(hessian, dtype=float)
        hessian = 0.5 * (hessian + hessian.T)

        hessian_reg = 0.5 * (hessian + hessian.T)

        try:
            return np.linalg.inv(hessian_reg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(hessian_reg, rcond=1e-12)

    @staticmethod
    def _ellipsoid_axes(
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

    @staticmethod
    def _compute_event_errors(
        jacobian_matrix: np.ndarray,
        residuals: np.ndarray,
        pick_errors: np.ndarray,
        damping: float,
        dim: int,
        observation_covariance: Optional[np.ndarray] = None,
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
        damping : float
            Damping used in the inversion, added to normal matrix diagonal.
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

        if observation_covariance is None:
            if pick_errors.size != residuals.size:
                return None

            weights = time_error_weight(pick_errors)
            hessian = jacobian_matrix.T @ (weights[:, np.newaxis] * jacobian_matrix)
            hessian_inv = ErrorAnalyzer._regularized_inverse(hessian)
            if hessian_inv is None:
                return None

            dof = residuals.size - int(dim)
            if dof < 0:
                return None
            if dof == 0:
                # TODO: find sources in literature for how to handle this case,
                #  e.g. when number of picks equals number of parameters
                reduced_chi2 = 1.0
            else:
                sigma_inv_r = weights * residuals
                reduced_chi2 = float(residuals.T @ sigma_inv_r) / float(dof)
                if not np.isfinite(reduced_chi2):
                    return None
        else:
            observation_covariance = np.asarray(observation_covariance, dtype=float)
            if observation_covariance.shape != (
                jacobian_matrix.shape[0],
                jacobian_matrix.shape[0],
            ):
                return None

            try:
                sigma_inv_J = np.linalg.solve(observation_covariance, jacobian_matrix)
                sigma_inv_r = np.linalg.solve(observation_covariance, residuals)
            except np.linalg.LinAlgError:
                sigma_inv = np.linalg.pinv(observation_covariance, rcond=1e-8)
                sigma_inv_J = sigma_inv @ jacobian_matrix
                sigma_inv_r = sigma_inv @ residuals

            hessian = jacobian_matrix.T @ sigma_inv_J
            hessian_inv = ErrorAnalyzer._regularized_inverse(hessian)
            if hessian_inv is None:
                return None

            dof = residuals.size - int(dim)
            if dof < 0:
                return None
            if dof == 0:
                reduced_chi2 = 1.0
            else:
                reduced_chi2 = float(residuals.T @ sigma_inv_r) / float(dof)
                if not np.isfinite(reduced_chi2):
                    return None

        scale = reduced_chi2

        covariance_full = scale * hessian_inv
        covariance_full = 0.5 * (covariance_full + covariance_full.T)

        spatial_dim = min(3, dim - 1 if dim > 1 else dim)
        if spatial_dim <= 0:
            return None

        C_xyz = covariance_full[:spatial_dim, :spatial_dim]
        if not np.all(np.isfinite(C_xyz)):
            return None

        try:
            principal_std, principal_axes, erz, erh = ErrorAnalyzer._ellipsoid_axes(
                C_xyz
            )
        except np.linalg.LinAlgError:
            return None

        return {
            "C_full": covariance_full,
            "C_xyz": C_xyz,
            "principal_std_km_1sigma": principal_std,
            "principal_axes_unit_vectors": principal_axes,
            "ERZ_km_1sigma": erz,
            "ERH_km_1sigma": erh,
            "RMS_s": (
                float(np.sqrt(np.mean(residuals**2))) if residuals.size else np.nan
            ),
            "RDERR_s": (float(np.nanmedian(pick_errors)) if pick_errors.size else 0.0),
            "n_obs_used": int(jacobian_matrix.shape[0]),
        }

    def _validate_inversion_results(self, inversion_results: InversionResults) -> None:
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
            raise ValueError(
                "inversion_results.residuals_flat_final must be a 1D array"
            )
        if residuals_flat.shape[0] != jacobian_matrix.shape[0]:
            raise ValueError(
                f"residuals_flat_final has length {residuals_flat.shape[0]} but "
                f"A_final has {jacobian_matrix.shape[0]} rows"
            )

        dim = int(self.config.dim)
        expected_cols = self.data.n_events * dim
        if jacobian_matrix.shape[1] != expected_cols:
            raise ValueError(
                "A_final has {ncols} columns but expected {exp} "
                "(n_events={n_events}, dim={dim})".format(
                    ncols=jacobian_matrix.shape[1],
                    exp=expected_cols,
                    n_events=self.data.n_events,
                    dim=dim,
                )
            )

    def _aggregate_event_rows(
        self, inversion_results: InversionResults
    ) -> tuple[list[list[int]], list[list[float]], list[list[int]], list[list[int]]]:
        n_obs_per_event = int(self.data.n_stations * self.config.n_phases)

        valid_rows_original = np.asarray(
            inversion_results.valid_rows, dtype=int
        ).reshape(-1)

        n_jacobian_rows = np.asarray(
            inversion_results.jacobian_matrix, dtype=float
        ).shape[0]

        rows_per_event: list[list[int]] = [[] for _ in range(self.data.n_events)]
        pick_errors_per_event: list[list[float]] = [
            [] for _ in range(self.data.n_events)
        ]
        station_indices_per_event: list[list[int]] = [
            [] for _ in range(self.data.n_events)
        ]
        phase_indices_per_event: list[list[int]] = [
            [] for _ in range(self.data.n_events)
        ]

        n_err_cols = self.data.arrival_time_errors.shape[1]

        for original_row in valid_rows_original:
            if original_row < 0 or original_row >= n_jacobian_rows:
                continue

            ev = int(original_row) // n_obs_per_event
            if ev < 0 or ev >= self.data.n_events:
                continue

            local_row = int(original_row) - ev * n_obs_per_event
            if local_row < 0 or local_row >= n_err_cols:
                continue

            rows_per_event[ev].append(int(original_row))
            pick_errors_per_event[ev].append(
                float(self.data.arrival_time_errors[ev, local_row])
            )
            station_indices_per_event[ev].append(int(local_row // self.config.n_phases))
            phase_indices_per_event[ev].append(int(local_row % self.config.n_phases))

        return (
            rows_per_event,
            pick_errors_per_event,
            station_indices_per_event,
            phase_indices_per_event,
        )

    def compute_error_ellipsoids(
        self, inversion_results: InversionResults
    ) -> ErrorResults:
        """Compute error ellipsoids for all events in the inversion result.

        Parameters
        ----------
        inversion_results : InversionResults
            Outputs of the location inversion step.

        Returns
        -------
        ErrorResults
            Event-level covariance, spatial covariances, principal standard deviations,
            principal axes, and ERZ/ERH values.
        """
        self._validate_inversion_results(inversion_results)

        use_full_covariance = bool(getattr(self.config, "full_data_covariance", False))
        jacobian_matrix = np.asarray(inversion_results.jacobian_matrix, dtype=float)
        residuals_flat = np.asarray(inversion_results.residuals_flat, dtype=float)
        residuals_flat = residuals_flat.reshape(-1)

        dim = int(self.config.dim)
        spatial_dim = min(3, dim - 1 if dim > 1 else dim)

        covariances = np.zeros((self.data.n_events, dim, dim), dtype=float)
        spatial_covariances = np.zeros(
            (self.data.n_events, spatial_dim, spatial_dim), dtype=float
        )
        principal_std = np.zeros((self.data.n_events, spatial_dim), dtype=float)
        principal_axes = np.zeros(
            (self.data.n_events, spatial_dim, spatial_dim), dtype=float
        )
        erz_erh = np.full((self.data.n_events, 2), np.nan, dtype=float)

        (
            rows_per_event,
            pick_errors_per_event,
            station_indices_per_event,
            phase_indices_per_event,
        ) = self._aggregate_event_rows(inversion_results)

        for ev in range(self.data.n_events):
            if self.data.invalid_events[ev]:
                continue

            rows = np.asarray(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue

            col0 = ev * dim
            col1 = (ev + 1) * dim
            event_jacobian_matrix = jacobian_matrix[rows, col0:col1]
            event_residuals = residuals_flat[rows]
            event_pick_errors = np.asarray(pick_errors_per_event[ev], dtype=float)
            observation_covariance = None

            if use_full_covariance:
                station_indices = np.asarray(station_indices_per_event[ev], dtype=int)
                phase_indices = np.asarray(phase_indices_per_event[ev], dtype=int)
                observation_covariance = (
                    ErrorAnalyzer._build_observation_covariance_from_stations(
                        pick_errors=event_pick_errors,
                        station_indices=station_indices,
                        phase_indices=phase_indices,
                        station_distances_km=np.asarray(
                            self.data.station_distances_km, dtype=float
                        ),
                        corr_length_km=float(
                            getattr(self.config, "correlation_length_km", 1.0)
                        ),
                    )
                )
                if observation_covariance is None:
                    continue

            errinfo = self._compute_event_errors(
                jacobian_matrix=event_jacobian_matrix,
                residuals=event_residuals,
                pick_errors=event_pick_errors,
                damping=float(inversion_results.damping),
                dim=dim,
                observation_covariance=observation_covariance,
            )
            if errinfo is None:
                continue

            covariances[ev] = errinfo["C_full"]
            spatial_covariances[ev] = errinfo["C_xyz"]
            principal_std[ev] = errinfo["principal_std_km_1sigma"]
            principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
            erz_erh[ev] = [errinfo["ERZ_km_1sigma"], errinfo["ERH_km_1sigma"]]

        return ErrorResults(
            covariances=covariances,
            spatial_covariances=spatial_covariances,
            principal_std=principal_std,
            principal_axes=principal_axes,
            erz_erh=erz_erh,
        )
