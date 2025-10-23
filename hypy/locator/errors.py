import numpy as np
from .types import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .state import LocatorState


class ErrorAnalysisMixin:
    def __init__(self, *args, **kwargs) -> None:
        # per-instance defaults
        self._covariances = np.empty((0, 0, 0), dtype=float)
        self._spatial_covariances = np.empty((0, 0, 0), dtype=float)
        self._principal_std = np.empty((0, 0), dtype=float)
        self._principal_axes = np.empty((0, 0, 0), dtype=float)
        self._erz_erh = np.empty((0, 0), dtype=float)

        self._residuals_flat_final = np.empty((0,), dtype=float)

        super().__init__(*args, **kwargs)  # cooperative MRO

    def _compute_event_errors(
        self: "LocatorState", A_e, r_e, pick_errs_e, damping: float
    ):
        if A_e is None or A_e.size == 0:
            return None

        N = A_e.T @ A_e
        N_reg = N + (damping**2) * np.eye(self.dim)

        rms = float(np.sqrt(np.mean(r_e**2))) if r_e.size else 0.0
        rderr = float(np.nanmedian(pick_errs_e)) if pick_errs_e.size else 0.0
        sigma2 = rms**2 + rderr**2

        try:
            N_reg_inv = np.linalg.inv(N_reg)
        except np.linalg.LinAlgError:
            N_reg_inv = np.linalg.pinv(N_reg)
        C = sigma2 * N_reg_inv  # (4x4)

        Cxyz = C[:3, :3]
        evals, evecs = np.linalg.eigh(Cxyz)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        std_axes = np.sqrt(np.clip(evals, 0.0, None))  # km

        ex, ey, ez = evecs[0, :], evecs[1, :], evecs[2, :]
        proj_vert = np.abs(ez) * std_axes
        proj_horiz = np.sqrt(ex**2 + ey**2) * std_axes
        ERZ = float(np.max(proj_vert))
        ERH = float(np.max(proj_horiz))

        return {
            "C_full": C,
            "C_xyz": Cxyz,
            "principal_std_km_1sigma": std_axes,
            "principal_axes_unit_vectors": evecs,
            "ERZ_km_1sigma": ERZ,
            "ERH_km_1sigma": ERH,
            "RMS_s": rms,
            "RDERR_s": rderr,
        }

    def _compute_error_ellipsoids(self: "LocatorState"):
        self._covariances = np.zeros((self.n_events, self.dim, self.dim))
        self._spatial_covariances = np.zeros(
            (self.n_events, self.dim - 1, self.dim - 1)
        )
        self._principal_std = np.zeros((self.n_events, self.dim - 1))
        self._principal_axes = np.zeros((self.n_events, self.dim - 1, self.dim - 1))
        self._erz_erh = np.zeros((self.n_events, 2))

        A_all = self._A_final
        valid_rows_all = np.asarray(self._valid_rows_final, dtype=int)
        residuals_flat = self._residuals_flat_final
        rows_per_event: List = [[] for _ in range(self.n_events)]
        errs_per_event: List = [[] for _ in range(self.n_events)]

        for row in valid_rows_all:
            ev_idx = row // (self.n_stations * self.n_phases)
            rows_per_event[ev_idx].append(row)
            local_row = row - ev_idx * (self.n_stations * self.n_phases)
            errs_per_event[ev_idx].append(self.arrival_time_errors[ev_idx, local_row])

        for ev in range(self.n_events):
            if self.invalid_events[ev]:
                continue
            rows = np.array(rows_per_event[ev], dtype=int)
            if rows.size == 0:
                continue

            # id_event = np.sum(self.invalid_events[:ev])
            id_start = (self.n_stations * self.n_phases) * ev
            id_end = (self.n_stations * self.n_phases) * (ev + 1)
            rows = id_start + np.arange(
                len(
                    valid_rows_all[
                        (valid_rows_all >= id_start) & (valid_rows_all < id_end)
                    ]
                )
            )  # map to global row indices
            A_e = A_all[rows, ev * self.dim : (ev + 1) * self.dim]
            r_e = residuals_flat[rows]
            pick_errs_e = np.array(errs_per_event[ev], dtype=float)
            errinfo = self._compute_event_errors(A_e, r_e, pick_errs_e, damping=0.1)
            if errinfo is None:
                continue
            self._covariances[ev] = errinfo["C_full"]
            self._spatial_covariances[ev] = errinfo["C_xyz"]
            self._principal_std[ev] = errinfo["principal_std_km_1sigma"]
            self._principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
            self._erz_erh[ev] = [errinfo["ERZ_km_1sigma"], errinfo["ERH_km_1sigma"]]
