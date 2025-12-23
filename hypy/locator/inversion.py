from scipy.sparse.linalg._isolve.lsmr import lsmr
from typing import TYPE_CHECKING

import numpy as np
from obspy.geodetics import kilometer2degrees  # type: ignore[import-untyped]

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.damping import update_damping_factor
from .types import SolverLiteral

if TYPE_CHECKING:
    from .state import LocatorState
else:

    class LocatorState:
        pass


class InversionMixin:
    def __init__(self, *args, **kwargs) -> None:
        self.solver: SolverLiteral = "lsqr"  # per-instance default

        super().__init__(*args, **kwargs)  # cooperative MRO

    def locate(
        self: "LocatorState",
        max_iterations=1000,
        conv_threshold=1e-3,
        solve_airquakes=False,
        max_depth=50.0,
        safe_iteration_limit=20,
        trial_depth=10.0,
        solver: SolverLiteral = "lsqr",
    ):
        """
        Main entry point for event location. Runs initialization, inversion, error analysis, and finalization.
        Returns (finalized origins, invalid event indices).
        """
        if solver not in ["lsqr", "svd"]:
            raise ValueError("Unsupported solver: {solver}. Supported: 'lsqr', 'svd'.")
        self.solver = solver
        logger.info(f"Using {self.solver} solver")
        logger.debug(f"Iteration limit: {max_iterations}")
        logger.debug(f"Solve airquakes: {solve_airquakes}")

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

        logger.info(f"Locating {self.n_events} events with {self.n_stations} stations.")
        logger.info(f"Using phases: {self.phases}")
        logger.info(f"Using {self.solver} solver.")
        logger.info(f"Using {self.mode} mode for travel-time calculations.")
        logger.info(f"Distance weighting: {self._distance_weighting}")
        logger.info(f"Fix depth: {self._fix_depth}")
        logger.info(f"Precalculate velocities: {self._precalculate_velocities}")

        damping = 1.0
        max_damping = 10.0
        A = None
        residual_norms = []
        residuals_flat = np.array([])
        valid_rows = np.array([], dtype=int)

        prev_res = None

        for it in range(max_iterations):
            valid_rows = self._populate_derivative_matrix()
            if len(valid_rows) == 0:
                logger.error("No valid observations available for inversion. Aborting.")
                break

            # Observed travel times = arrival times (RELATIVE) - origin times (RELATIVE)
            observed_travel_times = self.arrivals - self.origin_times[:, np.newaxis]

            # Residuals = observed - theoretical travel times per event-station-phase
            residuals = observed_travel_times - self.theoretical_travel_times
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

            # Solve for the increment
            if self.solver == "lsqr":
                dx_solution = lsmr(A, b, damp=damping)[0]
            else:  # SVD with Tikhonov-style filtering
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
                dx_solution = Vt.T @ (s_inv * (U.T @ b))

            # Reshape to event blocks and clip
            dx = dx_solution.reshape((self.n_events, self.dim))
            dx = _clip_updates(it, dx)

            # Residual AFTER step for the linearized subproblem
            residual_vector = A @ dx_solution - b  # r_new
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)
            rms = np.sqrt(np.mean(residual_vector**2))

            # Depth sanity / constraints
            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                # dx[event_idx, 2] *= _taper_depth_updates(
                #     self.locations[event_idx, 2], max_depth
                # )
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

            # Convert km -> deg for lat/lon updates; apply updates
            dx[:, 0] = kilometer2degrees(dx[:, 0], self.model.model.radius_of_planet)
            dx[:, 1] = kilometer2degrees(dx[:, 1], self.model.model.radius_of_planet)
            valid_mask = ~self.invalid_events
            self.locations[valid_mask, :3] -= dx[valid_mask, :3]
            self.origin_times[valid_mask] += dx[valid_mask, 3]

            logger.debug(
                f"Iteration {it + 1}: Residual norm={res_norm:.6f}, RMS={rms:.6f} s, Damping={damping:.6f}"
            )
            logger.debug(
                f"Iteration {it + 1}: Origin for first event: "
                f"Lat={self.locations[0, 0]:.4f}, Lon={self.locations[0, 1]:.4f}, "
                f"Depth={self.locations[0, 2]:.2f} km, "
                f"t0_rel={self.origin_times[0]:.2f}s, "
                f"t0_abs={(self.origin_times[0] + self.time_refs[0]):.2f}s"
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

        self._A_final = (
            self.derivative_matrix
            * self.time_error_W[:, np.newaxis]
            * self.distance_W[:, np.newaxis]
        )
        self._valid_rows_final = valid_rows
        self._residuals_flat_final = residuals_flat
        self._lambda_final = damping
        logger.info(f"Location iterations completed in {it + 1} iterations.")
        logger.info(
            f"Located {np.sum(~self.invalid_events)} events, {np.sum(self.invalid_events)} events invalid."
        )
