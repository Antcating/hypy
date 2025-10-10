from typing import TYPE_CHECKING

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]
from scipy.sparse.linalg import lsqr  # type: ignore[import-untyped]

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.damping import update_damping_factor
from .types import SolverLiteral
from typing_extensions import Self

if TYPE_CHECKING:
    from .state import LocatorState


class InversionMixin:
    def locate(
        self: Self,
        max_iterations=1000,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=50.0,
        safe_iteration_limit=20,
        trial_depth=10.0,
        solver: SolverLiteral = "lsqr",
    ):
        if solver not in ["lsqr", "svd"]:
            raise ValueError("Unsupported solver: {solver}. Supported: 'lsqr', 'svd'.")
        self.solver = solver

        self.init_locations(trial_depth=trial_depth)
        self._run_location_iterations(
            max_iterations,
            conv_threshold,
            solve_airquakes,
            max_depth,
            safe_iteration_limit,
        )
        self._compute_error_ellipsoids()
        logger.info("Event location completed.")
        return self._finalize_origins(), self.report_invalid_events()

    def _run_location_iterations(
        self: "LocatorState",
        max_iterations,
        conv_threshold,
        solve_airquakes,
        max_depth,
        init_phase_iteration_max,
    ):
        def _clip_updates(
            it, dx, max_horizontal_km=3.0, max_vertical_km=1.0, max_time_s=1.0
        ):
            dx[:, 0:2] = np.clip(dx[:, 0:2], -max_horizontal_km, max_horizontal_km)
            dx[:, 2] = np.clip(dx[:, 2], -max_vertical_km, max_vertical_km)
            dx[:, 3] = np.clip(dx[:, 3], -max_time_s, max_time_s)

            is_horiz_exceeds_depth = np.any(np.abs(dx[:, :2]) > max_vertical_km, axis=1)
            is_horiz_exceeds_limit = np.any(
                np.abs(dx[:, :2]) > max_horizontal_km, axis=1
            )

            if (
                it < init_phase_iteration_max and np.any(is_horiz_exceeds_depth)
            ) or np.any(is_horiz_exceeds_limit):
                logger.warning(
                    f"Iteration {it}: Preventing large horizontal shifts from causing large depth changes "
                    f"for {np.sum(is_horiz_exceeds_depth)} events."
                )
                dx[:, 2] = np.where(is_horiz_exceeds_depth, 0, dx[:, 2])
                dx[:, 3] = np.where(is_horiz_exceeds_depth, 0, dx[:, 3])
            return dx

        logger.info(f"Locating {self.n_events} events with {self.n_stations} stations.")
        logger.info(f"Using phases: {self.phases}")
        logger.info(f"Using {self.solver} solver.")
        logger.info(f"Using {self.mode} mode for travel-time calculations.")
        logger.info(f"Distance weighting: {self._distance_weighting}")
        logger.info(f"Fix depth: {self._fix_depth}")
        logger.info(f"Precalculate velocities: {self._precalculate_velocities}")

        damping = 1.0
        max_damping = 10
        A = None
        residual_norms = []
        residuals_flat = np.array([])
        valid_rows = np.array([], dtype=int)

        for it in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()

            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = travel_times_obs - self.theoretical_travel_times
            residuals_flat = residuals.flatten()

            b = (
                self.distance_W[valid_rows]
                * self.time_error_W[valid_rows]
                * residuals_flat[valid_rows]
            )
            A = (
                self.derivative_matrix[valid_rows, :]
                * self.time_error_W[valid_rows][:, np.newaxis]
                * self.distance_W[valid_rows][:, np.newaxis]
            )

            if self.solver == "lsqr":
                dx_solution = lsqr(A, b, damp=0)[0]
            else:  # SVD with Tikhonov-style filtering
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
                dx_solution = Vt.T @ (s_inv * (U.T @ b))

            dx = dx_solution.reshape((self.n_events, self.dim))
            dx = _clip_updates(it, dx)

            residual_vector = A @ dx_solution - b
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)

            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - dx[event_idx, 2]
                if not solve_airquakes and new_depth < 0.0:
                    logger.warning(
                        f"Event {event_idx} depth would become negative ({new_depth} km). Mark invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0
                elif it > init_phase_iteration_max and new_depth > max_depth:
                    logger.warning(
                        f"Event {event_idx} depth exceeds max depth ({new_depth} km > {max_depth} km). Mark invalid."
                    )
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)
            valid_mask = ~self.invalid_events
            self.locations[valid_mask, :3] -= dx[valid_mask, :3]
            self.origin_times[valid_mask] += dx[valid_mask, 3]

            damping = np.clip(update_damping_factor(damping), 0, max_damping)
            if res_norm < conv_threshold:
                break

        self._A_final = A
        self._valid_rows_final = valid_rows
        self._residuals_flat_final = residuals_flat
        self._lambda_final = damping
        logger.info(f"Location iterations completed in {it + 1} iterations.")
        logger.info(
            f"Located {np.sum(~self.invalid_events)} events, {np.sum(self.invalid_events)} events invalid."
        )
