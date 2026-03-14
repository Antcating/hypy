from scipy.sparse.linalg import lsmr

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.damping import update_damping_factor
from .config import LocatorConfig
from .data import DataHandler
from .derivatives import DerivativeEngine
from .model import VelocityModelHandler
from .types import InversionResults, SolverLiteral


class InversionEngine:
    """
    Iterative least-squares inversion for event relocation.

    All dependencies are passed explicitly via the constructor.
    Mutates DataHandler arrays (locations, origin_times, invalid_events) in-place
    and returns InversionResults for downstream error analysis.
    """

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        model_handler: VelocityModelHandler,
        derivative_engine: DerivativeEngine,
    ) -> None:
        self.config = config
        self.data = data
        self.model_handler = model_handler
        self.derivative_engine = derivative_engine

    def _clip_updates(
        self,
        it: int,
        dx: np.ndarray,
        max_horizontal_km: float = 3.0,
        max_vertical_km: float = 1.0,
        max_time_s: float = 1.0,
        couple_zt_to_horizontal: bool = True,
    ) -> np.ndarray:
        """
        Smooth safety clipping.

        Columns in dx:
            0: horizontal update component 1 [km]
            1: horizontal update component 2 [km]
            2: depth update [km]
            3: origin-time update [s]

        Horizontal updates are clipped by vector norm so direction is preserved.
        Optionally, z/t are shrunk by the same horizontal scale factor before
        their own scalar safety clipping.
        """
        dx_raw = dx.copy()
        dx_out = dx.copy()

        # Horizontal norm-based scaling
        hnorm = np.linalg.norm(dx_raw[:, :2], axis=1)
        hscale = np.minimum(1.0, max_horizontal_km / np.maximum(hnorm, 1e-12))

        dx_out[:, 0] = dx_raw[:, 0] * hscale
        dx_out[:, 1] = dx_raw[:, 1] * hscale

        if couple_zt_to_horizontal:
            dx_out[:, 2] = dx_raw[:, 2] * hscale
            dx_out[:, 3] = dx_raw[:, 3] * hscale
        else:
            dx_out[:, 2] = dx_raw[:, 2]
            dx_out[:, 3] = dx_raw[:, 3]

        # Final scalar safety clipping
        raw_depth = dx_out[:, 2].copy()
        raw_time = dx_out[:, 3].copy()
        dx_out[:, 2] = np.clip(dx_out[:, 2], -max_vertical_km, max_vertical_km)
        dx_out[:, 3] = np.clip(dx_out[:, 3], -max_time_s, max_time_s)

        n_h = int(np.sum(hnorm > max_horizontal_km))
        n_z = int(np.sum(np.abs(raw_depth) > max_vertical_km))
        n_t = int(np.sum(np.abs(raw_time) > max_time_s))

        if n_h or n_z or n_t:
            logger.warning(
                f"Iteration {it + 1}: clipped/scaled updates "
                f"(horizontal={n_h}, depth={n_z}, time={n_t})"
            )

        return dx_out

    def run_iterations(
        self,
        solver: SolverLiteral = "lsqr",
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        max_depth: float = 50.0,
        safe_iteration_limit: int = 20,
        max_lm_trials: int = 8,
        lm_acceptance_tol: float = 1e-6,
    ) -> InversionResults:
        """
        Main iterative inversion loop. Runs until convergence or max_iterations.

        LM-style logic:
        - linearize at current model
        - solve a damped least-squares step
        - trial the step on the true nonlinear objective
        - accept only if true RMS improves enough
        - otherwise increase damping and retry from the same base model
        """
        config = self.config
        data = self.data

        def _compute_true_residuals_and_rms(valid_rows: np.ndarray):
            """
            Compute true residuals for the CURRENT model state:
                residual = observed_travel_times - theoretical_travel_times
            and RMS over valid rows only.
            """
            observed_travel_times = data.arrivals - data.origin_times[:, np.newaxis]
            residuals = observed_travel_times - data.theoretical_travel_times
            residuals_flat_local = residuals.flatten()

            if len(valid_rows) == 0:
                rms_local = np.inf
            else:
                rms_local = float(
                    np.sqrt(np.mean(residuals_flat_local[valid_rows] ** 2))
                )

            return residuals_flat_local, rms_local

        logger.info(f"Locating {data.n_events} events with {data.n_stations} stations.")
        logger.info(f"Using phases: {config.phases}")
        logger.info(f"Using {solver} solver.")
        logger.info(f"Using {config.mode} mode for travel-time calculations.")
        logger.info(f"Distance weighting: {config.distance_weighting_enabled}")
        logger.info(f"Fix depth: {config.fix_depth}")
        logger.info(f"Precalculate velocities: {config.precalculate_velocities}")

        damping = 1.0
        max_damping = 10.0
        residuals_flat = np.array([], dtype=float)
        valid_rows = np.array([], dtype=int)
        prev_true_rms = None

        for it in range(max_iterations):
            # Build derivatives and theoretical travel times for CURRENT model
            valid_rows = self.derivative_engine.populate_derivative_matrix()
            if len(valid_rows) == 0:
                logger.error("No valid observations available for inversion. Aborting.")
                break

            # True residuals / true RMS BEFORE update
            residuals_flat, true_rms_before = _compute_true_residuals_and_rms(
                valid_rows
            )

            b = (
                data.distance_W[valid_rows]
                * data.time_error_W[valid_rows]
                * residuals_flat[valid_rows]
            )
            A = (
                data.derivative_matrix[valid_rows, :]
                * data.time_error_W[valid_rows][:, np.newaxis]
                * data.distance_W[valid_rows][:, np.newaxis]
            )

            # Save state before LM trials
            base_locations = data.locations.copy()
            base_origin_times = data.origin_times.copy()
            base_invalid_events = data.invalid_events.copy()

            accepted = False
            trial_linear_res_norm = np.nan
            trial_linear_rms = np.nan
            true_rms_after = true_rms_before

            for lm_trial in range(max_lm_trials):
                # Always restart each LM trial from the same base state
                data.locations[:] = base_locations
                data.origin_times[:] = base_origin_times
                data.invalid_events[:] = base_invalid_events

                # Solve for the increment with CURRENT damping
                if solver == "lsqr":
                    dx_solution = lsmr(A, b, damp=damping)[0]
                else:  # SVD with Tikhonov-style filtering
                    U, s, Vt = np.linalg.svd(A, full_matrices=False)
                    s_filt = np.zeros_like(s)
                    mask = s > 1e-12
                    s_filt[mask] = s[mask] / (s[mask] ** 2 + damping**2)
                    dx_solution = Vt.T @ (s_filt * (U.T @ b))

                # Reshape to event blocks and clip/smooth
                dx = dx_solution.reshape((data.n_events, config.dim))
                dx = self._clip_updates(it, dx)

                # Diagnostics based on the ACTUAL step that will be applied
                dx_used_flat = dx.reshape(-1)
                residual_vector_linear = A @ dx_used_flat - b
                trial_linear_res_norm = float(np.linalg.norm(residual_vector_linear))
                trial_linear_rms = float(np.sqrt(np.mean(residual_vector_linear**2)))

                # Depth sanity / constraints on trial step
                for event_idx in range(data.n_events):
                    if data.invalid_events[event_idx]:
                        continue

                    new_depth = data.locations[event_idx, 2] - dx[event_idx, 2]
                    if not solve_airquakes and new_depth < 0.0:
                        logger.warning(
                            f"Event {event_idx} depth would become negative ({new_depth} km). "
                            f"Mark invalid for this trial."
                        )
                        data.invalid_events[event_idx] = True
                        dx[event_idx, :] = 0.0

                    elif it > safe_iteration_limit and new_depth > max_depth:
                        logger.warning(
                            f"Event {event_idx} depth exceeds max depth "
                            f"({new_depth} km > {max_depth} km). Mark invalid for this trial."
                        )
                        data.invalid_events[event_idx] = True
                        dx[event_idx, :] = 0.0

                # Convert km -> deg for lat/lon updates; apply trial step
                planet_radius = self.model_handler.model.model.radius_of_planet
                dx_apply = dx.copy()
                dx_apply[:, 0] = kilometer2degrees(dx_apply[:, 0], planet_radius)
                dx_apply[:, 1] = kilometer2degrees(dx_apply[:, 1], planet_radius)

                valid_mask = ~data.invalid_events
                data.locations[valid_mask, :3] -= dx_apply[valid_mask, :3]
                data.origin_times[valid_mask] += dx_apply[valid_mask, 3]

                # Rebuild derivatives and recompute TRUE RMS AFTER trial update
                valid_rows_trial = self.derivative_engine.populate_derivative_matrix()
                if len(valid_rows_trial) == 0:
                    logger.warning(
                        f"Iteration {it + 1}, LM trial {lm_trial + 1}: "
                        "no valid observations after trial update; increasing damping."
                    )
                    damping = update_damping_factor(
                        current_factor=float(damping),
                        rms_prev=float(true_rms_before),
                        rms_new=float(np.inf),
                        accepted=False,
                        max_factor=max_damping,
                    )
                    continue

                residuals_flat_trial, true_rms_trial = _compute_true_residuals_and_rms(
                    valid_rows_trial
                )

                rel_impr_trial = (true_rms_before - true_rms_trial) / max(
                    true_rms_before, 1e-12
                )

                if rel_impr_trial > lm_acceptance_tol:
                    # Accept trial step
                    accepted = True
                    valid_rows = valid_rows_trial
                    residuals_flat = residuals_flat_trial
                    true_rms_after = true_rms_trial

                    damping = update_damping_factor(
                        current_factor=float(damping),
                        rms_prev=float(true_rms_before),
                        rms_new=float(true_rms_after),
                        accepted=True,
                        max_factor=max_damping,
                    )
                    break

                # Reject trial: do not keep trial state, increase damping and retry
                damping = update_damping_factor(
                    current_factor=float(damping),
                    rms_prev=float(true_rms_before),
                    rms_new=float(true_rms_trial),
                    accepted=False,
                    max_factor=max_damping,
                )

            if not accepted:
                # Restore original state if all trials failed
                data.locations[:] = base_locations
                data.origin_times[:] = base_origin_times
                data.invalid_events[:] = base_invalid_events

                valid_rows = self.derivative_engine.populate_derivative_matrix()
                if len(valid_rows) == 0:
                    logger.error(
                        "No valid observations available after failed LM trials. Aborting."
                    )
                    break

                residuals_flat, true_rms_after = _compute_true_residuals_and_rms(
                    valid_rows
                )

                logger.warning(
                    f"Iteration {it + 1}: no acceptable LM step found after "
                    f"{max_lm_trials} trials. Keeping previous model state."
                )

            logger.debug(
                f"Iteration {it + 1}: "
                f"linear residual norm={trial_linear_res_norm:.6f}, "
                f"linear RMS={trial_linear_rms:.6f} s, "
                f"true RMS before={true_rms_before:.6f} s, "
                f"true RMS after={true_rms_after:.6f} s, "
                f"Damping={damping:.6f}, "
                f"accepted={accepted}"
            )
            logger.debug(
                f"Iteration {it + 1}: Origin for first event: "
                f"Lat={data.locations[0, 0]:.4f}, Lon={data.locations[0, 1]:.4f}, "
                f"Depth={data.locations[0, 2]:.2f} km, "
                f"t0_rel={data.origin_times[0]:.2f}s, "
                f"t0_abs={(data.origin_times[0] + data.time_refs[0]):.2f}s"
            )

            # Convergence check
            if prev_true_rms is not None:
                rel_impr = np.abs(
                    (prev_true_rms - true_rms_after) / max(prev_true_rms, 1e-12)
                )
                if true_rms_after < conv_threshold or rel_impr < 1e-8:
                    logger.info(
                        f"conv_threshold={conv_threshold}, "
                        f"true_rms={true_rms_after:.6f} s, rel_impr={rel_impr:.6e}"
                    )
                    logger.info(
                        f"Convergence reached at iteration {it + 1}: "
                        f"true RMS={true_rms_after:.6f} s, "
                        f"Relative improvement={rel_impr:.6e}"
                    )
                    break

            prev_true_rms = true_rms_after

        # Ensure final derivatives / travel times / valid rows correspond to final state
        valid_rows = self.derivative_engine.populate_derivative_matrix()
        residuals_flat, final_true_rms = _compute_true_residuals_and_rms(valid_rows)

        A_final = (
            data.derivative_matrix
            * data.time_error_W[:, np.newaxis]
            * data.distance_W[:, np.newaxis]
        )

        logger.info(f"Location iterations completed in {it + 1} iterations.")
        logger.info(
            f"Located {np.sum(~data.invalid_events)} events, "
            f"{np.sum(data.invalid_events)} events invalid."
        )
        logger.info(f"Final true RMS = {final_true_rms:.6f} s")

        return InversionResults(
            A_final=A_final,
            valid_rows_final=valid_rows,
            residuals_flat_final=residuals_flat,
            lambda_final=damping,
        )
