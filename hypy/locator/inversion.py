"""Inversion routines for seismic event relocation."""

from typing import Optional

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from scipy.sparse.linalg import lsqr  # type: ignore[import-untyped]

from ..logger import logger
from ..utils.weighting import residual_weight_factor
from .config import LocatorConfig, RunConfig
from .data import DataHandler
from .derivatives import DerivativesHandler
from .types import DIM, InversionResults, UnresolvedDepthLiteral

# Ceiling for the Levenberg-Marquardt damping factor. Past this the step is
# so short that further trials cannot rescue the iteration.
MAX_DAMPING = 10.0


class InversionEngine:
    """Iterative damped least-squares inversion for event relocation.

    Updates ``data.locations``, ``data.origin_times``, and
    ``data.invalid_events``. A free-depth step that leaves the model
    (above the free surface or below ``max_depth``) freezes depth and
    continues in ``(x, y, t)``.
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
        self.unresolved_depth: Optional[UnresolvedDepthLiteral] = "seed"
        self.damping = 1.0
        self.max_depth = np.inf
        self.depth_elimination_iteration = 20
        self.originally_fixed = np.array([], dtype=bool)
        self.freeze_depth_km = np.array([], dtype=float)

    def _clip_updates(
        self,
        iteration_index: int,
        dx: np.ndarray,
        max_horizontal_km: float = 3.0,
        max_vertical_km: float = 1.0,
        max_time_s: float = 1.0,
    ) -> np.ndarray:
        """Scale each event update as one vector so no component exceeds its cap."""
        dx_new = dx.copy()
        fixed_depth = self.data.resolve_fixed_depth_mask()
        if np.any(fixed_depth):
            dx_new[fixed_depth, 2] = 0.0

        horizontal_shift_km = np.linalg.norm(dx_new[:, :2], axis=1)
        horizontal_scale = np.minimum(
            1.0, max_horizontal_km / np.maximum(horizontal_shift_km, 1e-12)
        )
        depth_scale = np.minimum(
            1.0, max_vertical_km / np.maximum(np.abs(dx_new[:, 2]), 1e-12)
        )
        time_scale = np.minimum(
            1.0, max_time_s / np.maximum(np.abs(dx_new[:, 3]), 1e-12)
        )
        depth_time_scale = np.minimum(depth_scale, time_scale)

        step_scale = np.minimum(horizontal_scale, depth_time_scale)
        dx_new *= step_scale[:, np.newaxis]

        n_clipped_horizontal = int(np.sum(horizontal_scale < 1.0))
        n_clipped_depth = int(np.sum((depth_scale < 1.0) | fixed_depth))
        n_clipped_time = int(np.sum(time_scale < 1.0))
        if n_clipped_horizontal or n_clipped_depth or n_clipped_time:
            logger.debug(
                "Iteration %s: clipped/scaled updates "
                "(horizontal=%s, depth=%s, time=%s)",
                iteration_index + 1,
                n_clipped_horizontal,
                n_clipped_depth,
                n_clipped_time,
            )
        return dx_new

    def _weighted_rms(
        self,
        residuals: np.ndarray,
        rows: np.ndarray,
        row_weights: np.ndarray,
    ) -> float:
        """WRMS in seconds on the rows that entered the step.

        ``sqrt(sum((w r)^2) / sum(w^2))`` with the same ``row_weights`` that
        built ``dx``. A zero weight drops that row. A missing residual on a
        weighted row makes the objective unusable.
        """
        rows = np.asarray(rows, dtype=int)
        weights = np.asarray(row_weights, dtype=float).reshape(-1)
        if rows.size == 0 or weights.shape != rows.shape:
            return np.inf
        picked = np.asarray(residuals, dtype=float).reshape(-1)[rows]
        active = np.isfinite(weights) & (weights != 0.0)
        if not np.any(active) or not np.all(np.isfinite(picked[active])):
            return np.inf
        weights = weights[active]
        denom = float(np.sum(weights * weights))
        if denom <= 0.0:
            return np.inf
        return float(np.sqrt(np.sum((weights * picked[active]) ** 2) / denom))

    def _solve_dx(
        self,
        design: np.ndarray,
        rhs: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        """Solve the damped normal equations for one Marquardt trial."""
        return np.asarray(
            lsqr(design, rhs, damp=float(damping), atol=1.0e-10, btol=1.0e-10)[0],
            dtype=float,
        )

    def _apply_row_weights(
        self,
        residuals_flat: np.ndarray,
        valid_rows: np.ndarray,
        iteration_index: int,
    ) -> np.ndarray:
        """Refresh ``data.residual_W`` and return the combined row weights.

        Residual down-weighting only starts at ``residual_start_iteration``
        so that early iterations are not steered by residuals still
        dominated by the seed location.
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
            data.residual_W[valid_rows] = residual_weight_factor(
                residuals_flat[valid_rows],
                base_W,
                residual_rms_floor_s=run_config.residual_rms_floor_s,
                residual_full_weight_factor=run_config.residual_full_weight_factor,
                residual_zero_weight_factor=run_config.residual_zero_weight_factor,
            )
        return base_W * data.residual_W[valid_rows]

    def _mark_underdetermined_events(self, valid_rows: np.ndarray) -> np.ndarray:
        """Invalidate events left with too few usable rows and drop their rows.

        Stations beyond the propagation engine's ``max_distance_km`` are
        skipped when building the design matrix, so an event can fall below
        ``min_picks_per_event`` here even though it had enough picks loaded.
        """
        data = self.data
        config = self.config
        n_obs_per_event = data.n_stations * config.n_phases
        if n_obs_per_event <= 0 or data.n_events == 0:
            return valid_rows

        keep = ~np.asarray(data.invalid_events, dtype=bool)
        for event_idx in np.flatnonzero(keep):
            start = event_idx * n_obs_per_event
            n_valid = int(
                np.sum((valid_rows >= start) & (valid_rows < start + n_obs_per_event))
            )
            if n_valid < config.min_picks_per_event:
                data.invalid_events[event_idx] = True
                keep[event_idx] = False
                logger.warning(
                    "Event %s: only %d usable observations remain "
                    "(min_picks_per_event=%d); marking invalid. "
                    "Stations beyond the propagation engine "
                    "max_distance_km are skipped.",
                    event_idx,
                    n_valid,
                    config.min_picks_per_event,
                )
        if np.all(keep):
            return valid_rows
        row_event = np.floor_divide(valid_rows, n_obs_per_event).astype(int)
        return valid_rows[keep[row_event]]

    def _clamp_to_search_bounds(
        self,
        event_mask: np.ndarray,
        iteration_index: int,
        lm_trial: int,
    ) -> None:
        """Hold epicenters inside the configured search box."""
        bounds = self.config.search_bounds
        if bounds is None or not np.any(event_mask):
            return
        data = self.data
        lat_min, lat_max, lon_min, lon_max = bounds
        lats = data.locations[event_mask, 0]
        lons = data.locations[event_mask, 1]
        lat_clipped = np.clip(lats, lat_min, lat_max)
        lon_clipped = np.clip(lons, lon_min, lon_max)
        n_clamped = int(np.sum((lats != lat_clipped) | (lons != lon_clipped)))
        data.locations[event_mask, 0] = lat_clipped
        data.locations[event_mask, 1] = lon_clipped
        if n_clamped:
            logger.debug(
                "Iteration %s, LM trial %s: clamped %s event epicenter(s) to "
                "search bounds [lat=(%.4f, %.4f), lon=(%.4f, %.4f)].",
                iteration_index + 1,
                lm_trial + 1,
                n_clamped,
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )

    def _apply_step(
        self,
        dx: np.ndarray,
        base_locations: np.ndarray,
        iteration_index: int,
        lm_trial: int,
    ) -> None:
        """Apply a kilometre-space step to the hypocenters, in place."""
        data = self.data
        dx_apply = dx.copy()
        dx_apply[:, 0] = kilometer2degrees(dx_apply[:, 0])
        cos_lat = np.maximum(np.abs(np.cos(np.deg2rad(base_locations[:, 0]))), 1.0e-6)
        dx_apply[:, 1] = kilometer2degrees(dx_apply[:, 1]) / cos_lat

        valid = ~data.invalid_events
        data.locations[valid, :3] += dx_apply[valid, :3]
        data.origin_times[valid] += dx_apply[valid, 3]
        data.locations[valid, 2] = np.maximum(data.locations[valid, 2], 0.0)
        self._restore_frozen_depths()
        self._clamp_to_search_bounds(valid, iteration_index, lm_trial)

    def _run_lm_trials(
        self,
        design: np.ndarray,
        rhs: np.ndarray,
        valid_rows: np.ndarray,
        residuals_flat: np.ndarray,
        wrms_before: float,
        row_weights: np.ndarray,
        base_state: tuple[np.ndarray, np.ndarray, np.ndarray],
        iteration_index: int,
        max_lm_trials: int,
        lm_acceptance_tol: float,
    ) -> tuple[bool, np.ndarray, np.ndarray, float]:
        """Search damping factors until a step lowers the WRMS enough.

        Every trial restarts from ``base_state``, so a rejected step leaves
        no trace in ``data``. Returns the winning trial's ``(accepted,
        valid_rows, residuals_flat, wrms)``, or the inputs unchanged when no
        trial was accepted.
        """
        data = self.data
        base_locations, base_origin_times, base_invalid_events = base_state

        for lm_trial in range(max_lm_trials):
            data.locations[:] = base_locations
            data.origin_times[:] = base_origin_times
            data.invalid_events[:] = base_invalid_events

            dx = self._clip_updates(
                iteration_index,
                self._solve_dx(design, rhs, self.damping).reshape((data.n_events, DIM)),
            )
            newly_constrained = self._freeze_out_of_bounds_depths(
                dx, iteration_index=iteration_index, lm_trial=lm_trial
            )
            if np.any(newly_constrained):
                # Drop the depth column so the re-solve cannot move it.
                for event_idx in np.flatnonzero(newly_constrained):
                    design[:, int(event_idx) * DIM + 2] = 0.0
                dx = self._clip_updates(
                    iteration_index,
                    self._solve_dx(design, rhs, self.damping).reshape(
                        (data.n_events, DIM)
                    ),
                )

            self._apply_step(dx, base_locations, iteration_index, lm_trial)

            valid_rows_trial = self.derivative_engine.refresh_theoretical_travel_times(
                data, self.config, self.run_config
            )
            if len(valid_rows_trial) == 0:
                logger.debug(
                    "Iteration %s, LM trial %s: no valid observations after "
                    "trial update; increasing damping.",
                    iteration_index + 1,
                    lm_trial + 1,
                )
                self.damping = min(self.damping * 5.0, MAX_DAMPING)
                continue

            residuals_trial = self.data.observation_residuals().flatten()
            wrms_trial = self._weighted_rms(residuals_trial, valid_rows, row_weights)
            rel_impr = (wrms_before - wrms_trial) / max(wrms_before, 1e-12)
            if rel_impr > lm_acceptance_tol:
                self.damping = max(self.damping * 0.5, 1e-8)
                return True, valid_rows_trial, residuals_trial, wrms_trial

            logger.debug(
                "Iteration %s, LM trial %s: WRMS=%.6f s, relative improvement="
                "%.6e not sufficient to accept trial step; increasing damping.",
                iteration_index + 1,
                lm_trial + 1,
                wrms_trial,
                rel_impr,
            )
            self.damping = min(self.damping * 5.0, MAX_DAMPING)

        return False, valid_rows, residuals_flat, wrms_before

    def _restore_frozen_depths(self) -> None:
        """Pin depths frozen during this run back to their freeze depth.

        Events that were already depth-fixed before the run keep whatever
        depth they came in with.
        """
        frozen = self.data.resolve_fixed_depth_mask() & ~self.originally_fixed
        if np.any(frozen):
            self.data.locations[frozen, 2] = self.freeze_depth_km[frozen]

    def _freeze_out_of_bounds_depths(
        self,
        dx: np.ndarray,
        iteration_index: int = 0,
        lm_trial: int = 0,
    ) -> np.ndarray:
        """Freeze depth when a free-depth step leaves the model.

        Out of bounds is a proposed depth above the free surface or below
        ``max_depth``. ``unresolved_depth``
        of ``"current"`` or ``"seed"`` freezes immediately.
        ``None`` waits until ``depth_elimination_iteration`` before a
        too-deep step freezes depth in place.
        """
        n_events = self.data.n_events
        newly = np.zeros(n_events, dtype=bool)
        if n_events == 0:
            return newly
        already_fixed = self.data.resolve_fixed_depth_mask()
        invalid = np.asarray(self.data.invalid_events, dtype=bool)
        proposed = self.data.locations[:, 2] + dx[:, 2]
        too_shallow = proposed < 0.0
        too_deep = proposed > self.max_depth
        if self.unresolved_depth is None:
            too_deep = too_deep & (iteration_index > self.depth_elimination_iteration)
        newly = (~already_fixed) & (~invalid) & (too_shallow | too_deep)
        if not np.any(newly):
            return newly
        if self.unresolved_depth != "seed":
            self.freeze_depth_km[newly] = self.data.locations[newly, 2]
        dx[newly, 2] = 0.0
        for event_idx in np.flatnonzero(newly):
            self.data.fixed_depth_mask[event_idx] = True
            logger.debug(
                "Event %s: proposed depth %.3f km is out of bounds; "
                "freezing at %s (%.3f km) on trial %d.",
                event_idx,
                float(proposed[event_idx]),
                self.unresolved_depth or "current",
                float(self.freeze_depth_km[event_idx]),
                lm_trial + 1,
            )
        logger.info(
            "Iteration %s, LM trial %s: %s event(s) left the depth "
            "bounds; freezing those depths at the %s hypocenter.",
            iteration_index + 1,
            lm_trial + 1,
            int(np.sum(newly)),
            self.unresolved_depth or "current",
        )
        return newly

    def run_iterations(
        self,
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        depth_elimination_iteration: int = 20,
        max_lm_trials: int = 8,
        lm_acceptance_tol: float = 1e-6,
    ) -> InversionResults:
        """Run the damped least-squares loop.

        Parameters
        ----------
        max_iterations : int, default=1000
            Maximum outer iterations.
        conv_threshold : float, default=1e-3
            WRMS convergence threshold, in seconds.
        depth_elimination_iteration : int, default=20
            Used only when ``RunConfig.unresolved_depth`` is ``None``: iterations
            before a too-deep step freezes depth in place.
        max_lm_trials : int, default=8
            Damping trials per outer iteration.
        lm_acceptance_tol : float, default=1e-6
            Relative WRMS improvement required to accept a trial.
        """
        config = self.config
        run_config = self.run_config
        data = self.data
        engine = self.derivative_engine.propagation_engine
        self.unresolved_depth = run_config.unresolved_depth
        self.max_depth = float(engine.max_depth_km)
        self.depth_elimination_iteration = depth_elimination_iteration
        self.originally_fixed = np.asarray(data.fixed_depth_mask, dtype=bool).copy()
        self.freeze_depth_km = np.asarray(data.locations[:, 2], dtype=float).copy()

        logger.info(
            "Locating %s events with %s stations using %s.",
            data.n_events,
            data.n_stations,
            engine.name,
        )

        self.damping = 1.0
        residuals_flat = np.array([], dtype=float)
        valid_rows = np.array([], dtype=int)
        prev_wrms = None
        iteration_index = 0

        for iteration_index in range(max_iterations):
            valid_rows = self.derivative_engine.build_derivative_matrix(
                data, config, run_config
            )
            if len(valid_rows) == 0:
                logger.error("No valid observations available for inversion. Aborting.")
                break

            valid_rows = self._mark_underdetermined_events(valid_rows)
            if len(valid_rows) == 0:
                logger.error(
                    "No valid observations remain after dropping underdetermined "
                    "events. Aborting."
                )
                break

            residuals_flat = self.data.observation_residuals().flatten()
            row_W = self._apply_row_weights(residuals_flat, valid_rows, iteration_index)
            wrms_before = self._weighted_rms(residuals_flat, valid_rows, row_W)
            design_rows = valid_rows
            b = row_W * residuals_flat[design_rows]
            A = data.jacobian_matrix[design_rows, :] * row_W[:, np.newaxis]

            base_state = (
                data.locations.copy(),
                data.origin_times.copy(),
                data.invalid_events.copy(),
            )

            accepted, valid_rows, residuals_flat, wrms_after = self._run_lm_trials(
                A,
                b,
                design_rows,
                residuals_flat,
                wrms_before,
                row_W,
                base_state,
                iteration_index,
                max_lm_trials,
                lm_acceptance_tol,
            )

            if not accepted:
                base_locations, base_origin_times, base_invalid_events = base_state
                data.locations[:] = base_locations
                data.origin_times[:] = base_origin_times
                data.invalid_events[:] = base_invalid_events
                self._restore_frozen_depths()

                valid_rows = self.derivative_engine.refresh_theoretical_travel_times(
                    data, config, run_config
                )
                if len(valid_rows) == 0:
                    logger.error(
                        "No valid observations available after failed LM trials. "
                        "Aborting."
                    )
                    break

                residuals_flat = self.data.observation_residuals().flatten()
                wrms_after = self._weighted_rms(residuals_flat, design_rows, row_W)
                logger.debug(
                    "Iteration %s: no acceptable LM step found after %s trials. "
                    "Keeping previous model state.",
                    iteration_index + 1,
                    max_lm_trials,
                )

            logger.debug(
                "Iteration %s: WRMS before=%.6f s, WRMS after=%.6f s, "
                "Damping=%.6f, accepted=%s",
                iteration_index + 1,
                wrms_before,
                wrms_after,
                self.damping,
                accepted,
            )
            logger.debug(
                "Iteration %s: Origin for first event: "
                "Lat=%.4f, Lon=%.4f, Depth=%.2f km, t0_rel=%.2fs, t0_abs=%.2fs",
                iteration_index + 1,
                data.locations[0, 0],
                data.locations[0, 1],
                data.locations[0, 2],
                data.origin_times[0],
                data.origin_times[0] + data.time_refs[0],
            )

            if prev_wrms is not None:
                rel_impr = np.abs((prev_wrms - wrms_after) / max(prev_wrms, 1e-12))
                if wrms_after < conv_threshold or rel_impr < 1e-8:
                    logger.info(
                        "conv_threshold=%s, wrms=%.6f s, rel_impr=%.6e",
                        conv_threshold,
                        wrms_after,
                        rel_impr,
                    )
                    logger.info(
                        "Convergence reached at iteration %s: "
                        "WRMS=%.6f s, Relative improvement=%.6e",
                        iteration_index + 1,
                        wrms_after,
                        rel_impr,
                    )
                    break

            prev_wrms = wrms_after

        valid_rows = self.derivative_engine.build_derivative_matrix(
            data, config, run_config
        )
        residuals_flat = self.data.observation_residuals().flatten()
        if len(valid_rows) > 0:
            row_W = self._apply_row_weights(residuals_flat, valid_rows, iteration_index)
            wrms = self._weighted_rms(residuals_flat, valid_rows, row_W)
        else:
            wrms = np.inf

        logger.info(
            "Location iterations completed in %s iterations.",
            iteration_index + 1,
        )
        logger.info(
            "Located %s events, %s events invalid.",
            int(np.sum(~data.invalid_events)),
            int(np.sum(data.invalid_events)),
        )
        logger.info("Final WRMS = %.6f s, Final damping = %.6f", wrms, self.damping)

        return InversionResults(
            jacobian_matrix=data.jacobian_matrix,
            valid_rows=valid_rows,
            residuals_flat=residuals_flat,
        )
