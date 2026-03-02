from scipy.sparse.linalg._isolve.lsmr import lsmr
from typing import Dict

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.damping import update_damping_factor
from . import derivatives as _derivatives
from .types import DerivativeContext, IterationConfig, Matrices, Weights


def run_location_iterations(
    *,
    cfg: IterationConfig,
    ctx: DerivativeContext,
    matrices: Matrices,
    weights: Weights,
) -> Dict[str, np.ndarray]:
    def _clip_updates(
        it, dx, max_horizontal_km=3.0, max_vertical_km=1.0, max_time_s=1.0
    ):
        dx[:, 0:2] = np.clip(dx[:, 0:2], -max_horizontal_km, max_horizontal_km)
        dx[:, 2] = np.clip(dx[:, 2], -max_vertical_km, max_vertical_km)
        dx[:, 3] = np.clip(dx[:, 3], -max_time_s, max_time_s)

        is_horiz_exceeds_depth = np.any(np.abs(dx[:, :2]) > max_vertical_km, axis=1)
        is_horiz_exceeds_limit = np.any(np.abs(dx[:, :2]) > max_horizontal_km, axis=1)

        if np.any(is_horiz_exceeds_limit):
            logger.warning(
                f"Iteration {it}: Preventing large horizontal shifts from causing large depth changes "
                f"for {np.sum(is_horiz_exceeds_depth)} events."
            )
            dx[:, 2] = np.where(is_horiz_exceeds_depth, 0, dx[:, 2])
            dx[:, 3] = np.where(is_horiz_exceeds_depth, 0, dx[:, 3])
        return dx

    # def _taper_depth_updates(current_depth, max_depth, offset=0.1):
    #     if not max_depth:
    #         max_depth = 1000
    #     return offset + (
    #         0.5 - 0.5 * np.cos(2 * np.pi * current_depth / max_depth)
    #     ) * (1 - offset)

    logger.info(f"Locating {ctx.n_events} events with {ctx.n_stations} stations.")
    logger.info(f"Using {cfg.solver} solver.")
    logger.info(f"Using {cfg.mode} mode for travel-time calculations.")
    logger.info(f"Distance weighting: {ctx.distance_weighting}")
    logger.info(f"Fix depth: {cfg.fix_depth}")
    logger.info(f"Precalculate velocities: {cfg.precalculate_velocities}")

    damping = 1.0
    max_damping = 10.0
    A = None
    residual_norms = []
    residuals_flat = np.array([])
    valid_rows = np.array([], dtype=int)

    prev_res = None

    it = -1

    for it in range(cfg.max_iterations):
        pop = _derivatives.populate_derivative_matrix(
            ctx=ctx,
            matrices=matrices,
            weights=weights,
        )
        valid_rows = pop["valid_rows"]
        matrices.theoretical_travel_times = pop["theoretical_travel_times"]
        matrices.derivative_matrix = pop["derivative_matrix"]
        weights.distance_W = pop["distance_W"]
        if len(valid_rows) == 0:
            logger.error("No valid observations available for inversion. Aborting.")
            break

        # Observed travel times = arrival times (RELATIVE) - origin times (RELATIVE)
        observed_travel_times = matrices.arrivals - matrices.origin_times[:, np.newaxis]

        # Residuals = observed - theoretical travel times per event-station-phase
        residuals = observed_travel_times - matrices.theoretical_travel_times
        residuals_flat = residuals.flatten()

        b = (
            weights.distance_W[valid_rows]
            * weights.time_error_W[valid_rows]
            * residuals_flat[valid_rows]
        )
        A = (
            matrices.derivative_matrix[valid_rows, :]
            * weights.time_error_W[valid_rows][:, np.newaxis]
            * weights.distance_W[valid_rows][:, np.newaxis]
        )

        # Solve for the increment
        if cfg.solver == "lsqr":
            dx_solution = lsmr(A, b, damp=damping)[0]
        else:  # SVD with Tikhonov-style filtering
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            s_inv = np.zeros_like(s)
            s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
            dx_solution = Vt.T @ (s_inv * (U.T @ b))

        # Reshape to event blocks and clip
        dx = dx_solution.reshape((ctx.n_events, ctx.dim))
        dx = _clip_updates(it, dx)

        # Residual AFTER step for the linearized subproblem
        residual_vector = A @ dx_solution - b  # r_new
        res_norm = np.linalg.norm(residual_vector)
        residual_norms.append(res_norm)
        rms = np.sqrt(np.mean(residual_vector**2))

        # Depth sanity / constraints
        for event_idx in range(ctx.n_events):
            if matrices.invalid_events[event_idx]:
                continue
            # dx[event_idx, 2] *= _taper_depth_updates(current_depth_km, max_depth)
            new_depth = matrices.locations[event_idx, 2] - dx[event_idx, 2]
            if not cfg.solve_airquakes and new_depth < 0.0:
                logger.warning(
                    f"Event {event_idx} depth would become negative ({new_depth} km). Mark invalid."
                )
                matrices.invalid_events[event_idx] = True
                dx[event_idx, :] = 0.0
            elif it > cfg.init_phase_iteration_max and new_depth > cfg.max_depth:
                logger.warning(
                    f"Event {event_idx} depth exceeds max depth ({new_depth} km > {max_depth} km). Mark invalid."
                )
                matrices.invalid_events[event_idx] = True
                dx[event_idx, :] = 0.0

        # Convert km -> deg for lat/lon updates; apply updates
        dx[:, 0] = kilometer2degrees(dx[:, 0], cfg.model_radius_of_planet_km)
        dx[:, 1] = kilometer2degrees(dx[:, 1], cfg.model_radius_of_planet_km)
        valid_mask = ~matrices.invalid_events
        matrices.locations[valid_mask, :3] -= dx[valid_mask, :3]
        matrices.origin_times[valid_mask] += dx[valid_mask, 3]

        logger.debug(
            f"Iteration {it + 1}: Residual norm={res_norm:.6f}, RMS={rms:.6f} s, Damping={damping:.6f}"
        )

        # --- Adaptive damping update
        damping = np.clip(
            update_damping_factor(
                current_factor=float(damping),
                rms_prev=float(prev_res) if prev_res is not None else float(rms),
                rms_new=float(rms),
            ),
            0.0,
            max_damping,
        )

        # Convergence check
        if prev_res is not None:
            rel_impr = np.abs((prev_res - rms) / max(prev_res, 1e-12))
            if rms < cfg.conv_threshold or rel_impr < 1e-8:
                logger.info(
                    f"conv_threshold={cfg.conv_threshold}, rms={rms:.6f} s, rel_impr={rel_impr:.6e}"
                )
                logger.info(
                    f"Convergence reached at iteration {it + 1}: RMS={rms:.6f} s, "
                    f"Relative improvement={rel_impr:.6e}"
                )
                break
        prev_res = rms

    logger.info(f"Location iterations completed in {it + 1} iterations.")
    logger.info(
        f"Located {np.sum(~matrices.invalid_events)} events, {np.sum(matrices.invalid_events)} events invalid."
    )

    return {
        "locations": matrices.locations,
        "origin_times": matrices.origin_times,
        "invalid_events": matrices.invalid_events,
        "theoretical_travel_times": matrices.theoretical_travel_times,
        "derivative_matrix": matrices.derivative_matrix,
        "distance_W": weights.distance_W,
        "A_final": (
            matrices.derivative_matrix
            * weights.time_error_W[:, np.newaxis]
            * weights.distance_W[:, np.newaxis]
        ),
        "valid_rows_final": valid_rows,
        "residuals_flat_final": residuals_flat,
        "lambda_final": float(damping),
    }
