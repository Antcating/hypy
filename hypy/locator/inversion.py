from scipy.sparse.linalg._isolve.lsmr import lsmr
from typing import Dict

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.damping import update_damping_factor
from . import derivatives as _derivatives
from .types import SolverLiteral


def run_location_iterations(
    *,
    solver: SolverLiteral,
    max_iterations: int,
    conv_threshold: float,
    solve_airquakes: bool,
    max_depth: float,
    init_phase_iteration_max: int,
    n_events: int,
    n_stations: int,
    n_phases: int,
    dim: int,
    mode: str,
    tracer,
    stations,
    stations_locations: np.ndarray,
    phases,
    fixed_depth_mask,
    distance_weighting: bool,
    k,
    x0,
    fix_depth: bool,
    precalculate_velocities: bool,
    model_radius_of_planet_km: float,
    arrivals: np.ndarray,
    locations: np.ndarray,
    origin_times: np.ndarray,
    time_refs: np.ndarray,
    theoretical_travel_times: np.ndarray,
    derivative_matrix: np.ndarray,
    time_error_W: np.ndarray,
    distance_W: np.ndarray,
    invalid_events: np.ndarray,
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

    logger.info(f"Locating {n_events} events with {n_stations} stations.")
    logger.info(f"Using {solver} solver.")
    logger.info(f"Using {mode} mode for travel-time calculations.")
    logger.info(f"Distance weighting: {distance_weighting}")
    logger.info(f"Fix depth: {fix_depth}")
    logger.info(f"Precalculate velocities: {precalculate_velocities}")

    damping = 1.0
    max_damping = 10.0
    A = None
    residual_norms = []
    residuals_flat = np.array([])
    valid_rows = np.array([], dtype=int)

    prev_res = None

    it = -1

    for it in range(max_iterations):
        pop = _derivatives.populate_derivative_matrix(
            tracer=tracer,
            locations=locations,
            stations_locations=stations_locations,
            stations=stations,
            phases=phases,
            n_events=n_events,
            n_stations=n_stations,
            n_phases=n_phases,
            dim=dim,
            arrivals=arrivals,
            theoretical_travel_times=theoretical_travel_times,
            derivative_matrix=derivative_matrix,
            distance_W=distance_W,
            time_error_W=time_error_W,
            invalid_events=invalid_events,
            fixed_depth_mask=fixed_depth_mask,
            distance_weighting=distance_weighting,
            k=k,
            x0=x0,
        )
        valid_rows = pop["valid_rows"]
        theoretical_travel_times = pop["theoretical_travel_times"]
        derivative_matrix = pop["derivative_matrix"]
        distance_W = pop["distance_W"]
        if len(valid_rows) == 0:
            logger.error("No valid observations available for inversion. Aborting.")
            break

        # Observed travel times = arrival times (RELATIVE) - origin times (RELATIVE)
        observed_travel_times = arrivals - origin_times[:, np.newaxis]

        # Residuals = observed - theoretical travel times per event-station-phase
        residuals = observed_travel_times - theoretical_travel_times
        residuals_flat = residuals.flatten()

        b = (
            distance_W[valid_rows]
            * time_error_W[valid_rows]
            * residuals_flat[valid_rows]
        )
        A = (
            derivative_matrix[valid_rows, :]
            * time_error_W[valid_rows][:, np.newaxis]
            * distance_W[valid_rows][:, np.newaxis]
        )

        # Solve for the increment
        if solver == "lsqr":
            dx_solution = lsmr(A, b, damp=damping)[0]
        else:  # SVD with Tikhonov-style filtering
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            s_inv = np.zeros_like(s)
            s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
            dx_solution = Vt.T @ (s_inv * (U.T @ b))

        # Reshape to event blocks and clip
        dx = dx_solution.reshape((n_events, dim))
        dx = _clip_updates(it, dx)

        # Residual AFTER step for the linearized subproblem
        residual_vector = A @ dx_solution - b  # r_new
        res_norm = np.linalg.norm(residual_vector)
        residual_norms.append(res_norm)
        rms = np.sqrt(np.mean(residual_vector**2))

        # Depth sanity / constraints
        for event_idx in range(n_events):
            if invalid_events[event_idx]:
                continue
            # dx[event_idx, 2] *= _taper_depth_updates(current_depth_km, max_depth)
            new_depth = locations[event_idx, 2] - dx[event_idx, 2]
            if not solve_airquakes and new_depth < 0.0:
                logger.warning(
                    f"Event {event_idx} depth would become negative ({new_depth} km). Mark invalid."
                )
                invalid_events[event_idx] = True
                dx[event_idx, :] = 0.0
            elif it > init_phase_iteration_max and new_depth > max_depth:
                logger.warning(
                    f"Event {event_idx} depth exceeds max depth ({new_depth} km > {max_depth} km). Mark invalid."
                )
                invalid_events[event_idx] = True
                dx[event_idx, :] = 0.0

        # Convert km -> deg for lat/lon updates; apply updates
        dx[:, 0] = kilometer2degrees(dx[:, 0], model_radius_of_planet_km)
        dx[:, 1] = kilometer2degrees(dx[:, 1], model_radius_of_planet_km)
        valid_mask = ~invalid_events
        locations[valid_mask, :3] -= dx[valid_mask, :3]
        origin_times[valid_mask] += dx[valid_mask, 3]

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
            if rms < conv_threshold or rel_impr < 1e-8:
                logger.info(
                    f"conv_threshold={conv_threshold}, rms={rms:.6f} s, rel_impr={rel_impr:.6e}"
                )
                logger.info(
                    f"Convergence reached at iteration {it + 1}: RMS={rms:.6f} s, "
                    f"Relative improvement={rel_impr:.6e}"
                )
                break
        prev_res = rms

    logger.info(f"Location iterations completed in {it + 1} iterations.")
    logger.info(
        f"Located {np.sum(~invalid_events)} events, {np.sum(invalid_events)} events invalid."
    )

    return {
        "locations": locations,
        "origin_times": origin_times,
        "invalid_events": invalid_events,
        "theoretical_travel_times": theoretical_travel_times,
        "derivative_matrix": derivative_matrix,
        "distance_W": distance_W,
        "A_final": (
            derivative_matrix * time_error_W[:, np.newaxis] * distance_W[:, np.newaxis]
        ),
        "valid_rows_final": valid_rows,
        "residuals_flat_final": residuals_flat,
        "lambda_final": float(damping),
    }
