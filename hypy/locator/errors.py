import numpy as np
from typing import Any, Dict, List, Optional


def compute_event_errors(
    *,
    dim: int,
    A_e: np.ndarray,
    r_e: np.ndarray,
    pick_errs_e: np.ndarray,
    damping: float,
) -> Optional[Dict[str, Any]]:
    if A_e is None or A_e.size == 0:
        return None

    N = A_e.T @ A_e
    N_reg = N + (damping**2) * np.eye(dim)

    rms = float(np.sqrt(np.mean(r_e**2))) if r_e.size else 0.0
    rderr = float(np.nanmedian(pick_errs_e)) if pick_errs_e.size else 0.0
    sigma2 = rms**2 + rderr**2

    try:
        N_reg_inv = np.linalg.inv(N_reg)
    except np.linalg.LinAlgError:
        N_reg_inv = np.linalg.pinv(N_reg)
    C = sigma2 * N_reg_inv

    Cxyz = C[:3, :3]
    evals, evecs = np.linalg.eigh(Cxyz)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    std_axes = np.sqrt(np.clip(evals, 0.0, None))

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


def compute_error_ellipsoids(
    *,
    dim: int,
    n_events: int,
    n_stations: int,
    n_phases: int,
    A_final: np.ndarray,
    valid_rows_final: np.ndarray,
    residuals_flat_final: np.ndarray,
    arrival_time_errors: np.ndarray,
    invalid_events: np.ndarray,
    damping: float = 0.1,
) -> Dict[str, np.ndarray]:
    covariances = np.zeros((n_events, dim, dim))
    spatial_covariances = np.zeros((n_events, dim - 1, dim - 1))
    principal_std = np.zeros((n_events, dim - 1))
    principal_axes = np.zeros((n_events, dim - 1, dim - 1))
    erz_erh = np.zeros((n_events, 2))

    A_all = A_final
    valid_rows_all = np.asarray(valid_rows_final, dtype=int)
    residuals_flat = residuals_flat_final
    rows_per_event: List = [[] for _ in range(n_events)]
    errs_per_event: List = [[] for _ in range(n_events)]

    for row in valid_rows_all:
        ev_idx = row // (n_stations * n_phases)
        rows_per_event[ev_idx].append(row)
        local_row = row - ev_idx * (n_stations * n_phases)
        errs_per_event[ev_idx].append(arrival_time_errors[ev_idx, local_row])

    for ev in range(n_events):
        if invalid_events[ev]:
            continue
        rows = np.array(rows_per_event[ev], dtype=int)
        if rows.size == 0:
            continue

        id_start = (n_stations * n_phases) * ev
        id_end = (n_stations * n_phases) * (ev + 1)
        rows = id_start + np.arange(
            len(
                valid_rows_all[(valid_rows_all >= id_start) & (valid_rows_all < id_end)]
            )
        )
        A_e = A_all[rows, ev * dim : (ev + 1) * dim]
        r_e = residuals_flat[rows]
        pick_errs_e = np.array(errs_per_event[ev], dtype=float)
        errinfo = compute_event_errors(
            dim=dim,
            A_e=A_e,
            r_e=r_e,
            pick_errs_e=pick_errs_e,
            damping=damping,
        )
        if errinfo is None:
            continue
        covariances[ev] = errinfo["C_full"]
        spatial_covariances[ev] = errinfo["C_xyz"]
        principal_std[ev] = errinfo["principal_std_km_1sigma"]
        principal_axes[ev] = errinfo["principal_axes_unit_vectors"].T
        erz_erh[ev] = [errinfo["ERZ_km_1sigma"], errinfo["ERH_km_1sigma"]]

    return {
        "covariances": covariances,
        "spatial_covariances": spatial_covariances,
        "principal_std": principal_std,
        "principal_axes": principal_axes,
        "erz_erh": erz_erh,
    }
