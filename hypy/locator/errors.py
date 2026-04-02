"""Error covariance and ellipsoid analysis for hypy event locations."""

from typing import Optional

import numpy as np

from .config import LocatorConfig
from .data import DataHandler
from .types import ErrorResults, InversionResults


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
    def _filter_input_rows(
        design_matrix: np.ndarray,
        residuals: np.ndarray,
        pick_errors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter observation rows with non-finite entries.

        Parameters
        ----------
        design_matrix : ndarray, shape (n_obs, dim)
            Weighted design matrix for one event.
        residuals : ndarray, shape (n_obs,)
            Residual vector for the same rows.
        pick_errors : ndarray, shape (n_obs,) or (k,)
            Pick timing uncertainty values.

        Returns
        -------
        tuple
            Filtered (design_matrix, residuals, pick_errors)
            containing only finite rows.
        """
        design_matrix = np.asarray(design_matrix, dtype=float)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        pick_errors = np.asarray(pick_errors, dtype=float).reshape(-1)

        row_mask = np.isfinite(residuals) & np.all(np.isfinite(design_matrix), axis=1)
        if pick_errors.size == design_matrix.shape[0]:
            row_mask &= np.isfinite(pick_errors)

        design_matrix = design_matrix[row_mask]
        residuals = residuals[row_mask]
        if pick_errors.size == row_mask.size:
            pick_errors = pick_errors[row_mask]
        else:
            pick_errors = pick_errors[np.isfinite(pick_errors)]

        return design_matrix, residuals, pick_errors

    @staticmethod
    def _compute_sigma2(residuals: np.ndarray, pick_errors: np.ndarray) -> float:
        """Compute combined variance from residuals and pick uncertainties.

        Parameters
        ----------
        residuals : ndarray, shape (n_obs,)
            Residuals for valid observations.
        pick_errors : ndarray, shape (n_obs,)
            Pick timing error estimates.

        Returns
        -------
        float
            Combined variance sigma^2 = RMS^2 + RDERR^2.
        """
        rms = float(np.sqrt(np.mean(residuals**2))) if residuals.size else np.nan
        rderr = float(np.nanmedian(pick_errors)) if pick_errors.size else 0.0
        return float(rms**2 + rderr**2)

    @staticmethod
    def _regularized_inverse(N: np.ndarray, damping: float) -> Optional[np.ndarray]:
        """Compute damped inverse of the normal matrix.

        Parameters
        ----------
        N : ndarray, shape (dim, dim)
            Unregularized normal matrix A.T @ A.
        damping : float
            Damping factor applied to diagonal.

        Returns
        -------
        ndarray or None
            Inverse of N_reg or None when inversion fails.
        """
        N_reg = N + (float(damping) ** 2) * np.eye(N.shape[0], dtype=float)
        N_reg = 0.5 * (N_reg + N_reg.T)

        try:
            return np.linalg.inv(N_reg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(N_reg, rcond=1e-12)

    @staticmethod
    def _ellipsoid_axes(Cxyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Derive principal axes and ERZ/ERH from the spatial covariance block.

        Parameters
        ----------
        Cxyz : ndarray, shape (k, k)
            Spatial covariance matrix (k = 1,2,3).

        Returns
        -------
        tuple
            principal_std : ndarray
                Root-variances along principal axes.
            erz : float
                Equivalent vertical error (1-sigma) or nan if not 3D.
            erh : float
                Equivalent horizontal error (1-sigma).

        Raises
        ------
        LinAlgError
            For invalid covariance matrix eigendecomposition.
        """
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
            return std_axes, float(np.max(proj_vert)), float(np.max(proj_horiz))

        if Cxyz.shape[0] == 2:
            return std_axes, np.nan, float(np.max(std_axes))

        return std_axes, np.nan, float(std_axes[0]) if std_axes.size else np.nan

    @staticmethod
    def _compute_event_errors(
        design_matrix: np.ndarray,
        residuals: np.ndarray,
        pick_errors: np.ndarray,
        damping: float,
        dim: int,
    ) -> Optional[dict]:
        """Compute error covariance and ellipsoid metrics for one event.

        Parameters
        ----------
        design_matrix : ndarray, shape (n_obs, dim)
            Event-specific design matrix rows used in inversion.
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
            design_matrix.ndim != 2
            or design_matrix.shape[1] != dim
            or design_matrix.shape[0] == 0
        ):
            return None

        if residuals.shape[0] != design_matrix.shape[0]:
            return None

        design_matrix, residuals, pick_errors = ErrorAnalyzer._filter_input_rows(
            design_matrix, residuals, pick_errors
        )

        if design_matrix.shape[0] == 0:
            return None

        sigma2 = ErrorAnalyzer._compute_sigma2(residuals, pick_errors)
        if not np.isfinite(sigma2):
            return None

        normal_matrix = design_matrix.T @ design_matrix
        regularized_normal = ErrorAnalyzer._regularized_inverse(normal_matrix, damping)
        if regularized_normal is None:
            return None

        covariance_full = sigma2 * regularized_normal
        covariance_full = 0.5 * (covariance_full + covariance_full.T)

        spatial_dim = min(3, dim - 1 if dim > 1 else dim)
        if spatial_dim <= 0:
            return None

        C_xyz = covariance_full[:spatial_dim, :spatial_dim]
        if not np.all(np.isfinite(C_xyz)):
            return None

        try:
            principal_std, erz, erh = ErrorAnalyzer._ellipsoid_axes(C_xyz)
            _, principal_axes = np.linalg.eigh(C_xyz)
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
            "n_obs_used": int(design_matrix.shape[0]),
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
        global_design = np.asarray(inversion_results.design_matrix, dtype=float)
        residuals_flat = np.asarray(
            inversion_results.residuals_flat, dtype=float
        ).reshape(-1)

        if global_design.ndim != 2:
            raise ValueError("inversion_results.A_final must be a 2D array")
        if residuals_flat.ndim != 1:
            raise ValueError(
                "inversion_results.residuals_flat_final must be a 1D array"
            )
        if residuals_flat.shape[0] != global_design.shape[0]:
            raise ValueError(
                f"residuals_flat_final has length {residuals_flat.shape[0]} but "
                f"A_final has {global_design.shape[0]} rows"
            )

        dim = int(self.config.dim)
        expected_cols = self.data.n_events * dim
        if global_design.shape[1] != expected_cols:
            raise ValueError(
                "A_final has {ncols} columns but expected {exp} "
                "(n_events={n_events}, dim={dim})".format(
                    ncols=global_design.shape[1],
                    exp=expected_cols,
                    n_events=self.data.n_events,
                    dim=dim,
                )
            )

    def _aggregate_event_rows(
        self, inversion_results: InversionResults
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Aggregate valid row indices and pick-errors for each event.

        Parameters
        ----------
        inversion_results : InversionResults
            Inversion output containing A_final and valid_rows_final.

        Returns
        -------
        tuple
            (rows_per_event, pick_errs_per_event)

            rows_per_event : list of list of int
                Event-specific global row indices to use for error estimation.
            pick_errs_per_event : list of list of float
                Event-specific pick-timing uncertainties corresponding to valid rows.
        """
        n_obs_per_event = int(self.data.n_stations * self.config.n_phases)
        valid_rows_all = np.unique(
            np.asarray(inversion_results.valid_rows, dtype=int).reshape(-1)
        )
        n_global_rows = int(
            np.asarray(inversion_results.design_matrix, dtype=float).shape[0]
        )

        valid_rows_all = valid_rows_all[
            (valid_rows_all >= 0) & (valid_rows_all < n_global_rows)
        ]

        rows_per_event: list[list[int]] = [[] for _ in range(self.data.n_events)]
        pick_errors_per_event: list[list[float]] = [
            [] for _ in range(self.data.n_events)
        ]

        n_err_cols = self.data.arrival_time_errors.shape[1]

        for row in valid_rows_all:
            ev = row // n_obs_per_event
            if ev < 0 or ev >= self.data.n_events:
                continue

            local_row = row - ev * n_obs_per_event
            if local_row < 0 or local_row >= n_err_cols:
                continue

            rows_per_event[ev].append(int(row))
            pick_errors_per_event[ev].append(
                float(self.data.arrival_time_errors[ev, local_row])
            )

        return rows_per_event, pick_errors_per_event

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

        global_design = np.asarray(inversion_results.design_matrix, dtype=float)
        residuals_flat = np.asarray(
            inversion_results.residuals_flat, dtype=float
        ).reshape(-1)

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

        rows_per_event, pick_errors_per_event = self._aggregate_event_rows(
            inversion_results
        )

        for ev in range(self.data.n_events):
            if self.data.invalid_events[ev]:
                continue

            rows = np.asarray(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue

            col0 = ev * dim
            col1 = (ev + 1) * dim
            event_design_matrix = global_design[rows, col0:col1]
            event_residuals = residuals_flat[rows]
            event_pick_errors = np.asarray(pick_errors_per_event[ev], dtype=float)

            errinfo = self._compute_event_errors(
                design_matrix=event_design_matrix,
                residuals=event_residuals,
                pick_errors=event_pick_errors,
                damping=float(inversion_results.damping),
                dim=dim,
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
