from typing import Optional

import numpy as np

from .config import LocatorConfig
from .data import DataHandler
from .types import ErrorResults, InversionResults


class ErrorAnalyzer:
    """
    Computes error ellipsoids and uncertainty estimates for located events.

    Assumptions
    -----------
    - inversion_results.A_final is the FULL weighted design matrix over all global rows
    - inversion_results.valid_rows_final contains global row indices that were valid
    - inversion_results.residuals_flat_final is the FULL true residual vector over all global rows
    - parameters are ordered per event in blocks of size `dim`
    - first 3 parameters are spatial (x/y/z or lat/lon/depth-like)
    """

    def __init__(self, config: LocatorConfig, data: DataHandler) -> None:
        self.config = config
        self.data = data

    @staticmethod
    def _compute_event_errors(
        A_e: np.ndarray,
        r_e: np.ndarray,
        pick_errs_e: np.ndarray,
        damping: float,
        dim: int,
    ) -> Optional[dict]:
        """
        Compute covariance and principal axes for a single event.

        Parameters
        ----------
        A_e : ndarray, shape (n_obs_valid, dim)
            Weighted design matrix block for one event using only valid rows.
        r_e : ndarray, shape (n_obs_valid,)
            True residuals for the same valid rows.
        pick_errs_e : ndarray, shape (n_obs_valid,)
            Pick timing errors for the same valid rows.
        damping : float
            Final damping used in inversion.
        dim : int
            Number of model parameters per event.

        Returns
        -------
        dict or None
            None if the event does not have enough usable information.
        """
        A_e = np.asarray(A_e, dtype=float)
        r_e = np.asarray(r_e, dtype=float).reshape(-1)
        pick_errs_e = np.asarray(pick_errs_e, dtype=float).reshape(-1)

        if A_e.ndim != 2 or A_e.shape[1] != dim:
            return None
        if A_e.shape[0] == 0:
            return None
        if r_e.shape[0] != A_e.shape[0]:
            return None

        # Keep only fully finite rows, so arbitrary invalid stations/phases are supported.
        row_mask = np.isfinite(r_e) & np.all(np.isfinite(A_e), axis=1)
        if pick_errs_e.size == A_e.shape[0]:
            row_mask &= np.isfinite(pick_errs_e)

        A_e = A_e[row_mask]
        r_e = r_e[row_mask]
        if pick_errs_e.size == row_mask.size:
            pick_errs_e = pick_errs_e[row_mask]
        else:
            pick_errs_e = pick_errs_e[np.isfinite(pick_errs_e)]

        if A_e.shape[0] == 0:
            return None

        # Require at least one row; pseudoinverse handles underdetermined cases.
        N = A_e.T @ A_e
        N_reg = N + (float(damping) ** 2) * np.eye(dim, dtype=float)
        N_reg = 0.5 * (N_reg + N_reg.T)

        rms = float(np.sqrt(np.mean(r_e**2))) if r_e.size else np.nan
        rderr = float(np.nanmedian(pick_errs_e)) if pick_errs_e.size else 0.0
        sigma2 = float(rms**2 + rderr**2)

        if not np.isfinite(sigma2):
            return None

        try:
            N_reg_inv = np.linalg.inv(N_reg)
        except np.linalg.LinAlgError:
            N_reg_inv = np.linalg.pinv(N_reg, rcond=1e-12)

        C = sigma2 * N_reg_inv
        C = 0.5 * (C + C.T)

        spatial_dim = min(3, dim - 1 if dim > 1 else dim)
        if spatial_dim <= 0:
            return None

        Cxyz = C[:spatial_dim, :spatial_dim]
        Cxyz = 0.5 * (Cxyz + Cxyz.T)

        if not np.all(np.isfinite(Cxyz)):
            return None

        try:
            evals, evecs = np.linalg.eigh(Cxyz)
        except np.linalg.LinAlgError:
            return None

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        std_axes = np.sqrt(np.clip(evals, 0.0, None))

        # ERZ / ERH only make sense for 3D spatial block
        if spatial_dim == 3:
            ex, ey, ez = evecs[0, :], evecs[1, :], evecs[2, :]
            proj_vert = np.abs(ez) * std_axes
            proj_horiz = np.sqrt(ex**2 + ey**2) * std_axes
            erz = float(np.max(proj_vert))
            erh = float(np.max(proj_horiz))
        elif spatial_dim == 2:
            # No true vertical axis available
            erz = np.nan
            erh = float(np.max(std_axes))
        else:
            erz = np.nan
            erh = float(std_axes[0]) if std_axes.size else np.nan

        return {
            "C_full": C,
            "C_xyz": Cxyz,
            "principal_std_km_1sigma": std_axes,
            "principal_axes_unit_vectors": evecs,
            "ERZ_km_1sigma": erz,
            "ERH_km_1sigma": erh,
            "RMS_s": rms,
            "RDERR_s": rderr,
            "n_obs_used": int(A_e.shape[0]),
        }

    def compute_error_ellipsoids(
        self,
        inversion_results: InversionResults,
    ) -> ErrorResults:
        """
        Compute per-event error ellipsoids from the final design matrix and residuals.

        Robust to multiple invalid stations/phases per event because it uses
        the final valid global row indices directly.
        """
        config = self.config
        data = self.data

        dim = int(config.dim)
        spatial_dim = min(3, dim - 1 if dim > 1 else dim)
        n_obs_per_event = int(data.n_stations * config.n_phases)

        covariances = np.zeros((data.n_events, dim, dim), dtype=float)
        spatial_covariances = np.zeros(
            (data.n_events, spatial_dim, spatial_dim), dtype=float
        )
        principal_std = np.zeros((data.n_events, spatial_dim), dtype=float)
        principal_axes = np.zeros(
            (data.n_events, spatial_dim, spatial_dim), dtype=float
        )
        erz_erh = np.full((data.n_events, 2), np.nan, dtype=float)

        A_all = np.asarray(inversion_results.A_final, dtype=float)
        valid_rows_all = np.asarray(
            inversion_results.valid_rows_final, dtype=int
        ).reshape(-1)
        residuals_flat = np.asarray(
            inversion_results.residuals_flat_final, dtype=float
        ).reshape(-1)

        if A_all.ndim != 2:
            raise ValueError("inversion_results.A_final must be a 2D array")
        if residuals_flat.ndim != 1:
            raise ValueError(
                "inversion_results.residuals_flat_final must be a 1D array"
            )

        n_global_rows, n_global_cols = A_all.shape
        expected_cols = data.n_events * dim
        if n_global_cols != expected_cols:
            raise ValueError(
                f"A_final has {n_global_cols} columns but expected {expected_cols} "
                f"(n_events={data.n_events}, dim={dim})"
            )
        if residuals_flat.shape[0] != n_global_rows:
            raise ValueError(
                f"residuals_flat_final has length {residuals_flat.shape[0]} but "
                f"A_final has {n_global_rows} rows"
            )

        # Keep only unique, in-range valid rows
        valid_rows_all = np.unique(valid_rows_all)
        valid_rows_all = valid_rows_all[
            (valid_rows_all >= 0) & (valid_rows_all < n_global_rows)
        ]

        # Group valid global rows by event
        rows_per_event: list[list[int]] = [[] for _ in range(data.n_events)]
        pick_errs_per_event: list[list[float]] = [[] for _ in range(data.n_events)]

        n_err_cols = data.arrival_time_errors.shape[1]

        for row in valid_rows_all:
            ev = row // n_obs_per_event
            if ev < 0 or ev >= data.n_events:
                continue

            local_row = row - ev * n_obs_per_event
            if local_row < 0 or local_row >= n_err_cols:
                continue

            rows_per_event[ev].append(int(row))

            pick_err = data.arrival_time_errors[ev, local_row]
            pick_errs_per_event[ev].append(float(pick_err))

        for ev in range(data.n_events):
            if data.invalid_events[ev]:
                continue

            rows = np.asarray(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue

            col0 = ev * dim
            col1 = (ev + 1) * dim

            A_e = A_all[rows, col0:col1]
            r_e = residuals_flat[rows]
            pick_errs_e = np.asarray(pick_errs_per_event[ev], dtype=float)

            errinfo = self._compute_event_errors(
                A_e=A_e,
                r_e=r_e,
                pick_errs_e=pick_errs_e,
                damping=float(inversion_results.lambda_final),
                dim=dim,
            )
            if errinfo is None:
                continue

            covariances[ev] = errinfo["C_full"]
            spatial_covariances[ev] = errinfo["C_xyz"]
            principal_std[ev] = errinfo["principal_std_km_1sigma"]
            principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
            erz_erh[ev, 0] = errinfo["ERZ_km_1sigma"]
            erz_erh[ev, 1] = errinfo["ERH_km_1sigma"]

        return ErrorResults(
            covariances=covariances,
            spatial_covariances=spatial_covariances,
            principal_std=principal_std,
            principal_axes=principal_axes,
            erz_erh=erz_erh,
        )
