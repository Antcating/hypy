"""Inversion routines for seismic event relocation."""

from typing import Optional

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from scipy.sparse.linalg import lsmr

from hypy.logger import logger

from ..utils.damping import update_damping_factor
from ..utils.weighting import residual_weight_factor
from .config import LocatorConfig, RunConfig
from .data import DataHandler
from .derivatives import DerivativesHandler
from .types import InversionResults, SolverLiteral


class InversionEngine:
    """Iterative least-squares inversion for seismic event relocation.

    This class encapsulates the main inversion loop for locating seismic events
    using arrival time residuals and a linearized least-squares model.

    The engine updates the runtime `DataHandler` arrays:
    - `locations` (lat, lon, depth)
    - `origin_times` (relative origin times)
    - `invalid_events` (boolean mask for unresolvable events)

    Assumptions
    -----------
    - The velocity model and derivatives are valid for the current event state.
    - Observed travel times and theoretical travel times have the same ordering
      and shape.

    Notes
    -----
    - The method applies Levenberg-Marquardt-style damping via `damping`.
    - Depth and time updates are clipped for numerical stability.

    Examples
    --------
    >>> engine = InversionEngine(config, data, model_handler, derivative_engine)
    >>> results = engine.run_iterations(max_iterations=100, conv_threshold=1e-3)
    >>> print(results.rms)
    """

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        derivative_engine: DerivativesHandler,
        run_config: RunConfig,
    ) -> None:
        self.config = config
        self.data = data
        self.derivative_engine = derivative_engine
        self.run_config = run_config

    def _clip_updates(
        self,
        iteration_index: int,
        dx: np.ndarray,
        max_horizontal_km: float = 3.0,
        max_vertical_km: float = 1.0,
        max_time_s: float = 1.0,
        couple_zt_to_horizontal: bool = True,
    ) -> np.ndarray:
        """Clip and smooth proposed event update steps.

        Parameters
        ----------
        iteration_index : int
            Iteration counter (zero-based).
        dx : np.ndarray, shape (n_events, 4)
            Candidate updates in the per-event block format [dx(lat), dy(lon), dz, dt].
        max_horizontal_km : float, default=3.0
            Maximum allowed horizontal displacement per iteration in kilometers.
        max_vertical_km : float, default=1.0
            Maximum allowed depth change per iteration in kilometers.
        max_time_s : float, default=1.0
            Maximum allowed origin-time shift per iteration in seconds.
        couple_zt_to_horizontal : bool, default=True
            If True, depth and time updates are scaled by horizontal clipping
            factor to ensure consistent step magnitude.

        Returns
        -------
        np.ndarray
            Clipped updates with the same shape as ``dx``.

        Notes
        -----
        - Horizontal updates are clipped by vector magnitude, preserving direction.
        - Depth and time updates are clipped independently after optional coupling.
        - A warning is emitted through logger when any component is clipped.

        Examples
        --------
        >>> dx = np.array([[5.0, 0.0, 0.5, 0.2]])
        >>> clipped = engine._clip_updates(0, dx, max_horizontal_km=3.0)
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

        if self.config.fix_depth_km is not None:
            # Number of clipped depth is equal to number of events,
            # since all depth updates are forced to zero.
            n_z = int(len(dx[:, 2]))
            dx_out[:, 2] = 0.0  # Force depth updates to zero if depth is fixed

        if n_h or n_z or n_t:
            logger.debug(
                f"Iteration {iteration_index + 1}: clipped/scaled updates "
                f"(horizontal={n_h}, depth={n_z}, time={n_t})"
            )

        return dx_out

    def _apply_residual_weights(
        self,
        valid_rows: np.ndarray,
        residuals_flat: np.ndarray,
        iteration_index: int,
    ) -> np.ndarray:
        """Update ``data.residual_W`` and return row scales for valid rows.

        Residual weighting is recomputed each outer iteration from the current
        residuals. It does not permanently remove picks: a previously zeroed
        arrival can regain weight if its residual improves.
        """
        data = self.data
        run_config = self.run_config
        data.residual_W[:] = 1.0

        base_W = data.distance_W[valid_rows] * data.time_error_W[valid_rows]
        if (
            run_config.residual_weighting_enabled
            and run_config.residual_start_iteration is not None
            and (iteration_index + 1) >= run_config.residual_start_iteration
        ):
            residual_W = residual_weight_factor(
                residuals_flat[valid_rows],
                base_W,
                residual_rms_floor_s=run_config.residual_rms_floor_s,
                residual_full_weight_factor=run_config.residual_full_weight_factor,
                residual_zero_weight_factor=run_config.residual_zero_weight_factor,
            )
            data.residual_W[valid_rows] = residual_W
        else:
            residual_W = np.ones(len(valid_rows), dtype=float)

        return base_W * residual_W

    def run_iterations(
        self,
        solver: SolverLiteral = "lsqr",
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        max_depth: Optional[float] = None,
        depth_elimination_iteration: int = 20,
        max_lm_trials: int = 8,
        lm_acceptance_tol: float = 1e-6,
    ) -> InversionResults:
        """Run the inversion loop to update event locations and times.

        Parameters
        ----------
        solver : {'lsqr', 'svd'}, default='lsqr'
            Linear solver type for each Levenberg-Marquardt trial step.
        max_iterations : int, default=1000
            Maximum number of outer iteration cycles.
        conv_threshold : float, default=1e-3
            Convergence threshold for RMS residual reduction.
        solve_airquakes : bool, default=False
            If True, allow depth to go negative (airquakes); otherwise mark as invalid.
        max_depth : float, default=50.0
            Maximum permitted depth (km) after the safe iteration limit.
        depth_elimination_iteration : int, default=20
            Number of iterations before strict depth bounds are enforced.
        max_lm_trials : int, default=8
            Number of damping trials attempted per iteration.
        lm_acceptance_tol : float, default=1e-6
            Relative RMS improvement threshold to accept a trial step.

        Returns
        -------
        InversionResults
            Object containing final jacobian matrix, valid rows, residuals, and damping.

        Notes
        -----
        - The algorithm performs a damped least-squares update with trial-and-error
          damping adjustment.
        - If all trials fail, the state is restored to the previous iteration state.
        - Behavior is deterministic for a fixed random-free data input set.

        Examples
        --------
        >>> results = engine.run_iterations(max_iterations=50, conv_threshold=1e-4)
        >>> print(results.residuals_flat.shape)
        """
        config = self.config
        run_config = self.run_config
        data = self.data

        def _compute_residuals_and_rms(valid_rows: np.ndarray):
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
        logger.info(
            f"Using {self.derivative_engine.tracer.name} forward engine "
            f"for travel-time calculations."
        )
        logger.info(f"Distance weighting: {run_config.distance_weighting_enabled}")
        if run_config.residual_weighting_enabled:
            logger.info(
                "Residual weighting: True "
                f"(start_iteration={run_config.residual_start_iteration}, "
                f"rms_floor_s={run_config.residual_rms_floor_s}, "
                f"full_weight_factor={run_config.residual_full_weight_factor}, "
                f"zero_weight_factor={run_config.residual_zero_weight_factor})"
            )
        else:
            logger.info("Residual weighting: False")
        logger.info(f"Fix depth: {config.fix_depth_km}")
        logger.info(f"Precalculate velocities: {config.precalculate_velocities}")

        best_damping = 0.0
        damping = 1.0
        max_damping = 10.0
        residuals_flat = np.array([], dtype=float)
        valid_rows = np.array([], dtype=int)
        derivatives_built = False
        prev_rms = None
        iteration_index = 0

        for iteration_index in range(max_iterations):
            if not derivatives_built:
                valid_rows = self.derivative_engine.build_derivative_matrix(
                    data, config, run_config
                )
                derivatives_built = True
            if len(valid_rows) == 0:
                logger.error("No valid observations available for inversion. Aborting.")
                break

            # Residuals / RMS BEFORE update
            residuals_flat, rms_before = _compute_residuals_and_rms(valid_rows)
            row_W = self._apply_residual_weights(
                valid_rows, residuals_flat, iteration_index
            )

            b = row_W * residuals_flat[valid_rows]
            A = data.jacobian_matrix[valid_rows, :] * row_W[:, np.newaxis]

            # Save state before LM trials
            base_locations = data.locations.copy()
            base_origin_times = data.origin_times.copy()
            base_invalid_events = data.invalid_events.copy()

            accepted = False
            rms_after = rms_before
            best_rms_trial = np.inf
            best_oob_events = np.zeros(data.n_events, dtype=bool)

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
                dx = self._clip_updates(iteration_index, dx)

                # Depth sanity / constraints on trial step.
                # Do not mark invalid in trial stage until all trials fail.
                trial_oob = np.zeros(data.n_events, dtype=bool)
                for event_idx in range(data.n_events):
                    if data.invalid_events[event_idx]:
                        continue

                    new_depth = data.locations[event_idx, 2] + dx[event_idx, 2]
                    if not solve_airquakes and new_depth < 0.0:
                        logger.warning(
                            "Event %s depth would become negative (%.3f km),"
                            " rejecting trial %d update.",
                            event_idx,
                            new_depth,
                            lm_trial + 1,
                        )
                        trial_oob[event_idx] = True
                        dx[event_idx, :] = 0.0

                    elif (
                        iteration_index > depth_elimination_iteration
                        and new_depth > max_depth
                    ):
                        logger.warning(
                            "Event %s depth exceeds max depth (%.3f km > %.3f km),"
                            " rejecting trial %d update.",
                            event_idx,
                            new_depth,
                            max_depth,
                            lm_trial + 1,
                        )
                        trial_oob[event_idx] = True
                        dx[event_idx, :] = 0.0

                # Convert km -> deg for lat/lon updates; apply trial step
                # planet_radius = self.model_handler.model.model.radius_of_planet
                dx_apply = dx.copy()
                dx_apply[:, 0] = kilometer2degrees(dx_apply[:, 0])
                dx_apply[:, 1] = kilometer2degrees(dx_apply[:, 1])

                valid_event_mask = ~data.invalid_events
                data.locations[valid_event_mask, :3] += dx_apply[valid_event_mask, :3]
                data.origin_times[valid_event_mask] += dx_apply[valid_event_mask, 3]

                # Rebuild derivatives and recompute RMS AFTER trial update
                valid_rows_trial = self.derivative_engine.build_derivative_matrix(
                    data, config, run_config
                )
                if len(valid_rows_trial) == 0:
                    logger.debug(
                        f"Iteration {iteration_index + 1}, LM trial {lm_trial + 1}: "
                        "no valid observations after trial update; increasing damping."
                    )
                    damping = update_damping_factor(
                        current_factor=float(damping),
                        rms_prev=float(rms_before),
                        rms_new=float(np.inf),
                        accepted=False,
                        max_factor=max_damping,
                    )
                    continue

                residuals_flat_trial, rms_trial = _compute_residuals_and_rms(
                    valid_rows_trial
                )

                if rms_trial < best_rms_trial:
                    best_rms_trial = rms_trial
                    best_oob_events = trial_oob.copy()

                rel_impr_trial = (rms_before - rms_trial) / max(rms_before, 1e-12)

                if rel_impr_trial > lm_acceptance_tol:
                    # Accept trial step
                    accepted = True
                    valid_rows = valid_rows_trial
                    residuals_flat = residuals_flat_trial
                    rms_after = rms_trial

                    best_damping = damping

                    damping = update_damping_factor(
                        current_factor=float(damping),
                        rms_prev=float(rms_before),
                        rms_new=float(rms_after),
                        accepted=True,
                        max_factor=max_damping,
                    )
                    break

                logger.debug(
                    f"Iteration {iteration_index + 1}, LM trial {lm_trial + 1}: "
                    f"RMS={rms_trial:.6f} s, relative improvement={rel_impr_trial:.6e} "
                    "not sufficient to accept trial step; increasing damping.",
                )

                # Reject trial: do not keep trial state, increase damping and retry
                damping = update_damping_factor(
                    current_factor=float(damping),
                    rms_prev=float(rms_before),
                    rms_new=float(rms_trial),
                    accepted=False,
                    max_factor=max_damping,
                )

            if not accepted:
                # Restore original state if all trials failed
                data.locations[:] = base_locations
                data.origin_times[:] = base_origin_times
                data.invalid_events[:] = base_invalid_events

                if np.any(best_oob_events):
                    data.invalid_events[best_oob_events] = True
                    logger.debug(
                        "Iteration %s: %s events invalidated after LM failure.",
                        iteration_index + 1,
                        int(np.sum(best_oob_events)),
                    )

                valid_rows = self.derivative_engine.build_derivative_matrix(
                    data, config, run_config
                )
                derivatives_built = True
                if len(valid_rows) == 0:
                    logger.error(
                        "No valid observations available after failed LM trials. "
                        "Aborting."
                    )
                    break

                residuals_flat, rms_after = _compute_residuals_and_rms(valid_rows)

                logger.debug(
                    f"Iteration {iteration_index + 1}: no acceptable LM step found "
                    f"after {max_lm_trials} trials. Keeping previous model state."
                )

            logger.debug(
                f"Iteration {iteration_index + 1}: "
                f"RMS before={rms_before:.6f} s, "
                f"RMS after={rms_after:.6f} s, "
                f"Damping={damping:.6f}, "
                f"accepted={accepted}"
            )
            logger.debug(
                f"Iteration {iteration_index + 1}: Origin for first event: "
                f"Lat={data.locations[0, 0]:.4f}, Lon={data.locations[0, 1]:.4f}, "
                f"Depth={data.locations[0, 2]:.2f} km, "
                f"t0_rel={data.origin_times[0]:.2f}s, "
                f"t0_abs={(data.origin_times[0] + data.time_refs[0]):.2f}s"
            )

            # Convergence check
            if prev_rms is not None:
                rel_impr = np.abs((prev_rms - rms_after) / max(prev_rms, 1e-12))
                if rms_after < conv_threshold or rel_impr < 1e-8:
                    logger.info(
                        f"conv_threshold={conv_threshold}, "
                        f"rms={rms_after:.6f} s, rel_impr={rel_impr:.6e}"
                    )
                    logger.info(
                        f"Convergence reached at iteration {iteration_index + 1}: "
                        f"RMS={rms_after:.6f} s, "
                        f"Relative improvement={rel_impr:.6e}"
                    )
                    break

            prev_rms = rms_after

        # Ensure final derivatives / travel times / valid rows correspond to final state
        if not derivatives_built:
            valid_rows = self.derivative_engine.build_derivative_matrix(
                data, config, run_config
            )
        residuals_flat, rms = _compute_residuals_and_rms(valid_rows)
        # Refresh residual weights at the final hypocenter
        if len(valid_rows) > 0:
            self._apply_residual_weights(valid_rows, residuals_flat, iteration_index)

        logger.info(
            f"Location iterations completed in {iteration_index + 1} iterations."
        )
        logger.info(
            f"Located {np.sum(~data.invalid_events)} events, "
            f"{np.sum(data.invalid_events)} events invalid."
        )
        logger.info(f"Final RMS = {rms:.6f} s, Final damping = {best_damping:.6f}")

        return InversionResults(
            jacobian_matrix=data.jacobian_matrix,
            valid_rows=valid_rows,
            residuals_flat=residuals_flat,
        )
