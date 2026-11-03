"""Inversion routines for seismic event relocation."""

from typing import Optional

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from ..logger import logger
from ..utils.weighting import residual_weight_factor
from .config import LocatorConfig, RunConfig
from .data import DataHandler
from .derivatives import DerivativesHandler
from .types import DIM, InversionResults, UnresolvedDepthLiteral

# Past this the LM step is too short for another trial to help.
MAX_DAMPING = 10.0


class InversionEngine:
    """Iterative damped least-squares relocation, HypoInverse form.

    The hypocenter is geographic, but each event is updated in a local
    frame ``δm = [δy, δx, δz, δt0]ᵀ`` (north km, east km, depth km, s).
    A free-depth step that leaves the model freezes ``δz`` and continues
    in ``(δy, δx, δt0)``.
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
        self.max_lm_trials = 8
        self.lm_acceptance_tol = 1e-6
        self.max_depth = np.inf
        self.depth_elimination_iteration = 20
        self.originally_fixed = np.array([], dtype=bool)
        self.freeze_depth_km = np.array([], dtype=float)

    def run_iterations(
        self,
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        depth_elimination_iteration: int = 20,
        max_lm_trials: int = 8,
        lm_acceptance_tol: float = 1e-6,
    ) -> InversionResults:
        """Run the iterative inversion.

        Each pass steps every event that is still converging. The loop
        stops when every event has finished or ``max_iterations`` is spent.
        A seed whose WRMS stays above ``max(model uncertainty, conv_threshold)``
        yields to the next fallback while passes remain.
        """
        run_config = self.run_config
        data = self.data
        engine = self.derivative_engine.propagation_engine
        self.unresolved_depth = run_config.unresolved_depth
        self.max_depth = float(engine.max_depth_km)
        self.depth_elimination_iteration = depth_elimination_iteration
        self.max_lm_trials = max_lm_trials
        self.lm_acceptance_tol = lm_acceptance_tol
        self.originally_fixed = np.asarray(data.fixed_depth_mask, dtype=bool).copy()
        self.freeze_depth_km = np.asarray(data.locations[:, 2], dtype=float).copy()
        logger.info(
            "Locating %s events with %s stations using %s.",
            data.n_events,
            data.n_stations,
            engine.name,
        )

        data.residual_W[:] = 1.0
        n_events = data.n_events
        done = np.asarray(data.invalid_events, dtype=bool).copy()
        seed_at = np.zeros(n_events, dtype=int)
        step_at = np.zeros(n_events, dtype=int)
        prev_wrms = np.full(n_events, np.nan)
        damping = np.ones(n_events)
        best: list[Optional[tuple]] = [None] * n_events
        starts = [self._starting_points(i) for i in range(n_events)]
        fix_z = [
            2
            if (
                not self.originally_fixed[i]
                and self.config.fix_depth_km is None
                and max_iterations > 4
            )
            else 0
            for i in range(n_events)
        ]
        retry = max(float(run_config.model_uncertainty_s), conv_threshold)
        kept = [np.array([], dtype=int) for _ in range(n_events)]
        for event_idx in range(n_events):
            if not done[event_idx]:
                self._arm_seed(event_idx, starts[event_idx][0])
                damping[event_idx] = self.damping

        def finish(event_idx: int, final: bool) -> None:
            # Lowest-WRMS seed wins. A worse fit arms the next fallback.
            last = max(int(step_at[event_idx]) - 1, 0)
            seed_i = int(seed_at[event_idx])
            trial_wrms, rows = self._seed_wrms(event_idx, last)
            captured = self._capture(event_idx, trial_wrms, seed_i)
            if best[event_idx] is None or trial_wrms < best[event_idx][0]:
                best[event_idx] = captured
            winner = best[event_idx]
            assert winner is not None
            if not final and trial_wrms > retry and seed_i + 1 < len(starts[event_idx]):
                seed_at[event_idx] = seed_i + 1
                step_at[event_idx] = 0
                prev_wrms[event_idx] = np.nan
                self._arm_seed(event_idx, starts[event_idx][seed_i + 1])
                damping[event_idx] = self.damping
                return
            done[event_idx] = True
            if winner[6] == seed_i and np.isfinite(trial_wrms):
                kept[event_idx] = rows
                return
            self._install(event_idx, winner)
            if winner[5]:
                return
            rows = self._derivatives(event_idx)
            if len(rows) > 0:
                self._score_rows(rows, last)
                kept[event_idx] = rows

        # One shared pass count. A finished event is skipped, not restarted.
        passes = 0
        wrms = np.inf
        for iteration in range(max_iterations):
            if n_events == 0 or bool(np.all(done)):
                break
            passes = iteration + 1
            for event_idx in range(n_events):
                if done[event_idx]:
                    continue
                self.damping = float(damping[event_idx])
                stop, wrms_after = self._step_event(
                    event_idx,
                    int(step_at[event_idx]),
                    fix_z[event_idx],
                    conv_threshold,
                    float(prev_wrms[event_idx]),
                )
                damping[event_idx] = self.damping
                step_at[event_idx] += 1
                if stop:
                    finish(event_idx, False)
                else:
                    prev_wrms[event_idx] = wrms_after
        for event_idx in range(n_events):
            if not done[event_idx]:
                finish(event_idx, True)
        for saved in best:
            if saved is not None:
                wrms = float(saved[0])
        present = [rows for rows in kept if len(rows) > 0]
        logger.info("Location iterations completed in %s iterations.", passes)
        logger.info(
            "Located %s events, %s events invalid.",
            int(np.sum(~data.invalid_events)),
            int(np.sum(data.invalid_events)),
        )
        logger.info("Final WRMS = %.6f s, Final damping = %.6f", wrms, self.damping)
        return InversionResults(
            jacobian_matrix=data.jacobian_matrix,
            valid_rows=np.concatenate(present) if present else np.array([], dtype=int),
            residuals_flat=data.observation_residuals().flatten(),
        )

    def _starting_points(self, event_idx):
        data = self.data
        seeds = [
            (
                float(data.locations[event_idx, 0]),
                float(data.locations[event_idx, 1]),
                float(data.locations[event_idx, 2]),
                float(data.origin_times[event_idx]),
            )
        ]
        if event_idx < len(data.fallback_seeds):
            seeds.extend(data.fallback_seeds[event_idx])
        return seeds

    def _arm_seed(self, event_idx, seed):
        # λ starts at 1 on every seed.
        data = self.data
        data.locations[event_idx] = seed[:3]
        data.origin_times[event_idx] = seed[3]
        data.fixed_depth_mask[event_idx] = bool(self.originally_fixed[event_idx])
        self.freeze_depth_km[event_idx] = seed[2]
        data.invalid_events[event_idx] = False
        self.damping = 1.0

    def _capture(self, event_idx, trial_wrms, seed_i):
        data = self.data
        return (
            trial_wrms,
            data.locations[event_idx].copy(),
            float(data.origin_times[event_idx]),
            bool(data.fixed_depth_mask[event_idx]),
            float(self.freeze_depth_km[event_idx]),
            bool(data.invalid_events[event_idx]),
            seed_i,
        )

    def _install(self, event_idx, saved):
        _, loc, t0, fix, freeze, inv, _ = saved
        self.data.locations[event_idx] = loc
        self.data.origin_times[event_idx] = t0
        self.data.fixed_depth_mask[event_idx] = fix
        self.freeze_depth_km[event_idx] = freeze
        self.data.invalid_events[event_idx] = inv

    def _seed_wrms(self, event_idx, step):
        if self.data.invalid_events[event_idx]:
            return np.inf, np.array([], dtype=int)
        rows = self._derivatives(event_idx)
        if len(rows) < self.config.min_picks_per_event:
            return np.inf, np.array([], dtype=int)
        return self._score_rows(rows, step)[2], rows

    def _step_event(self, event_idx, step, fix_z, conv_threshold, prev_wrms):
        """Take one damped step. True means this seed should stop.

        A free-depth seed holds ``z`` for its first two steps, then releases
        it. Fewer than ``min_picks_per_event`` usable rows marks the event
        invalid and leaves the others untouched.
        """
        data = self.data
        if fix_z > 0:
            if step < fix_z:
                data.fixed_depth_mask[event_idx] = True
            elif step == fix_z:
                data.fixed_depth_mask[event_idx] = False
        design = self._derivatives(event_idx)
        if len(design) < self.config.min_picks_per_event:
            data.invalid_events[event_idx] = True
            logger.warning(
                "Event %s: only %d usable observations remain "
                "(min_picks_per_event=%d); marking invalid.",
                event_idx,
                len(design),
                self.config.min_picks_per_event,
            )
            return True, np.inf

        residuals, weights, wrms_before = self._score_rows(design, step)
        base = self._snapshot()
        accepted, wrms_after = self._run_lm_trials(
            design, residuals, wrms_before, weights, base, step, event_idx
        )
        if not accepted:
            self._restore(base)
            self._restore_frozen_depths()
            if len(self._refresh(event_idx)) == 0:
                logger.error("No valid observations after failed LM trials. Aborting.")
                return True, wrms_before
            wrms_after = self._weighted_rms(
                data.observation_residuals().flatten(), design, weights
            )
            logger.debug(
                "Event %s, iteration %s: no acceptable LM step after %s trials.",
                event_idx,
                step + 1,
                self.max_lm_trials,
            )
        logger.debug(
            "Event %s, iteration %s: WRMS %.6f -> %.6f s, damping %.6f, accepted=%s",
            event_idx,
            step + 1,
            wrms_before,
            wrms_after,
            self.damping,
            accepted,
        )
        # Stop when WRMS < conv_threshold, or the relative change since
        # the previous step is < 1e-8. Wait until depth has been released.
        if np.isfinite(prev_wrms) and step > fix_z:
            rel = np.abs((prev_wrms - wrms_after) / max(prev_wrms, 1e-12))
            if wrms_after < conv_threshold or rel < 1e-8:
                logger.info(
                    "Event %s converged at iteration %s: WRMS=%.6f s, rel=%.6e",
                    event_idx,
                    step + 1,
                    wrms_after,
                    rel,
                )
                return True, wrms_after
        return False, wrms_after

    def _run_lm_trials(
        self, rows, residuals, wrms_before, weights, base, iteration, event_idx
    ):
        # Accept a trial only if (WRMS_before - WRMS) / WRMS_before > 1e-6,
        # then λ ← λ/2 (floor 1e-8). Otherwise λ ← 5λ, up to λ_max = 10.
        # Each trial rolls back to the pre-step state; at most 8 are tried.
        for trial in range(self.max_lm_trials):
            self._restore(base)
            step = self._solve(rows, residuals, weights)
            dx = self._clip_updates(iteration, step)
            if np.any(self._freeze_out_of_bounds_depths(dx, iteration, trial)):
                step = self._solve(rows, residuals, weights)
                dx = self._clip_updates(iteration, step)
            self._apply_step(dx, iteration, trial)
            if len(self._refresh(event_idx)) < self.config.min_picks_per_event:
                self.damping = min(self.damping * 5.0, MAX_DAMPING)
                continue
            trial_residuals = self.data.observation_residuals().flatten()
            eval_rows, eval_weights = rows, weights
            finite = np.isfinite(trial_residuals[rows])
            if not np.all(finite):
                if int(np.sum(finite)) < self.config.min_picks_per_event:
                    self.damping = min(self.damping * 5.0, MAX_DAMPING)
                    continue
                eval_rows, eval_weights = rows[finite], weights[finite]
            wrms = self._weighted_rms(trial_residuals, eval_rows, eval_weights)
            rel = (wrms_before - wrms) / max(wrms_before, 1e-12)
            if rel > self.lm_acceptance_tol:
                self.damping = max(self.damping * 0.5, 1e-8)
                return True, wrms
            self.damping = min(self.damping * 5.0, MAX_DAMPING)
        return False, wrms_before

    def _solve(self, rows, residuals, weights):
        """Solve ``[W J; λ I] δm = [W r; 0]``.

        ``J`` rows are ``[∂T/∂y, ∂T/∂x, ∂T/∂z, 1]`` and
        ``r = t_obs - t_m``. Fixed depth drops ``δz``. A zero weight
        drops the row. ``lstsq`` keeps the condition number of ``J``.
        """
        data = self.data
        dx = np.zeros((data.n_events, DIM))
        n_obs = data.n_stations * self.config.n_phases
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if n_obs <= 0 or rows.size == 0:
            return dx
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        fixed = data.resolve_fixed_depth_mask()
        lam = float(self.damping)
        event_of = rows // n_obs
        for ev in np.unique(event_of):
            w = weights[event_of == ev]
            on = w != 0.0
            if not np.any(on):
                continue
            free = np.ones(DIM, dtype=bool)
            free[2] = not fixed[ev]
            picked = rows[event_of == ev][on]
            coeff = data.jacobian_matrix[picked][:, free] * w[on, None]
            rhs = residuals[picked] * w[on]
            n = int(free.sum())
            dx[ev, free] = np.linalg.lstsq(
                np.vstack((coeff, lam * np.eye(n))),
                np.concatenate((rhs, np.zeros(n))),
                rcond=None,
            )[0]
        return dx

    def _score_rows(self, rows, iteration):
        residuals = self.data.observation_residuals().flatten()
        weights = self._apply_row_weights(residuals, rows, iteration)
        return residuals, weights, self._weighted_rms(residuals, rows, weights)

    def _snapshot(self):
        # Pre-step model. A rejected damping trial is rolled back onto it.
        data = self.data
        return (
            data.locations.copy(),
            data.origin_times.copy(),
            data.invalid_events.copy(),
            list(data.fixed_depth_mask),
            self.freeze_depth_km.copy(),
        )

    def _restore(self, state):
        data = self.data
        (
            data.locations[:],
            data.origin_times[:],
            data.invalid_events[:],
            data.fixed_depth_mask[:],
            self.freeze_depth_km[:],
        ) = state

    def _derivatives(self, event_idx):
        return self.derivative_engine.build_derivative_matrix(
            self.data, self.config, self.run_config, event_index=event_idx
        )

    def _refresh(self, event_idx):
        return self.derivative_engine.refresh_theoretical_travel_times(
            self.data, self.config, self.run_config, event_index=event_idx
        )

    def _clip_updates(
        self,
        iteration_index,
        dx,
        max_horizontal_km=3.0,
        max_vertical_km=1.0,
        max_time_s=1.0,
    ):
        """Scale ``δm`` as one vector: 3 km horizontal, 1 km depth, 1 s.

        Fixed depth forces ``δz = 0`` before the scale is chosen.
        """
        dx = dx.copy()
        fixed = self.data.resolve_fixed_depth_mask()
        dx[fixed, 2] = 0.0
        cap = np.minimum(
            max_horizontal_km / np.maximum(np.linalg.norm(dx[:, :2], axis=1), 1e-12),
            max_vertical_km / np.maximum(np.abs(dx[:, 2]), 1e-12),
        )
        cap = np.minimum(cap, max_time_s / np.maximum(np.abs(dx[:, 3]), 1e-12))
        dx *= np.minimum(cap, 1.0)[:, None]
        return dx

    def _weighted_rms(self, residuals, rows, row_weights):
        """WRMS ``sqrt(sum (w r)² / sum w²)`` on the rows that built ``δm``.

        A zero weight drops that row. A non-finite residual on a weighted
        row makes the objective unusable.
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

    def _apply_row_weights(self, residuals_flat, valid_rows, iteration_index):
        # Diagonal W: pick-time weight × distance taper × residual weight.
        # Residual weighting starts at residual_start_iteration, so the
        # seed's residuals do not steer the early steps. No pick is removed.
        data = self.data
        run_config = self.run_config
        data.residual_W[valid_rows] = 1.0
        base_w = data.distance_W[valid_rows] * data.time_error_W[valid_rows]
        if (
            run_config.residual_weighting_enabled
            and run_config.residual_start_iteration is not None
            and (iteration_index + 1) >= run_config.residual_start_iteration
        ):
            data.residual_W[valid_rows] = residual_weight_factor(
                residuals_flat[valid_rows],
                base_w,
                residual_rms_floor_s=run_config.residual_rms_floor_s,
                residual_full_weight_factor=run_config.residual_full_weight_factor,
                residual_zero_weight_factor=run_config.residual_zero_weight_factor,
            )
        return base_w * data.residual_W[valid_rows]

    def _clamp_to_search_bounds(self, event_mask, iteration_index, lm_trial):
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
                "Iteration %s, LM trial %s: clamped %s epicenter(s) to the search box.",
                iteration_index + 1,
                lm_trial + 1,
                n_clamped,
            )

    def _apply_step(self, dx, iteration_index, lm_trial):
        # δy, δx in km become Δlat, Δlon; Δlon is divided by cos(lat).
        # Depth is then clamped to z ≥ 0, frozen depths are pinned, and
        # epicenters are clipped back into the search box.
        data = self.data
        step = dx.copy()
        step[:, 0] = kilometer2degrees(step[:, 0])
        cos_lat = np.maximum(np.abs(np.cos(np.deg2rad(data.locations[:, 0]))), 1e-6)
        step[:, 1] = kilometer2degrees(step[:, 1]) / cos_lat
        valid = ~data.invalid_events
        data.locations[valid, :3] += step[valid, :3]
        data.origin_times[valid] += step[valid, 3]
        data.locations[valid, 2] = np.maximum(data.locations[valid, 2], 0.0)
        self._restore_frozen_depths()
        self._clamp_to_search_bounds(valid, iteration_index, lm_trial)

    def _restore_frozen_depths(self):
        frozen = self.data.resolve_fixed_depth_mask() & ~self.originally_fixed
        if np.any(frozen):
            self.data.locations[frozen, 2] = self.freeze_depth_km[frozen]

    def _freeze_out_of_bounds_depths(self, dx, iteration_index=0, lm_trial=0):
        """Freeze depth when a proposed step leaves the model.

        Out of bounds is ``z < 0`` or below the engine's maximum depth.
        ``seed`` freezes at the seed depth, ``current`` at the hypocenter
        in place. ``None`` waits until ``depth_elimination_iteration``
        before a too-deep step freezes. The depth unknown is then dropped
        and the step is re-solved.
        """
        n_events = self.data.n_events
        if n_events == 0:
            return np.zeros(0, dtype=bool)
        fixed = self.data.resolve_fixed_depth_mask()
        invalid = np.asarray(self.data.invalid_events, dtype=bool)
        proposed = self.data.locations[:, 2] + dx[:, 2]
        too_deep = proposed > self.max_depth
        if self.unresolved_depth is None:
            too_deep &= iteration_index > self.depth_elimination_iteration
        newly = ~fixed & ~invalid & ((proposed < 0.0) | too_deep)
        if not np.any(newly):
            return newly
        if self.unresolved_depth != "seed":
            self.freeze_depth_km[newly] = self.data.locations[newly, 2]
        dx[newly, 2] = 0.0
        for event_idx in np.flatnonzero(newly):
            self.data.fixed_depth_mask[event_idx] = True
        logger.info(
            "Iteration %s, LM trial %s: %s event(s) froze depth at the %s hypocenter.",
            iteration_index + 1,
            lm_trial + 1,
            int(np.sum(newly)),
            self.unresolved_depth or "current",
        )
        return newly
