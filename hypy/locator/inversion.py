from scipy.sparse.linalg._isolve.lsmr import lsmr

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

    def run_iterations(
        self,
        solver: SolverLiteral = "lsqr",
        max_iterations: int = 1000,
        conv_threshold: float = 1e-3,
        solve_airquakes: bool = False,
        max_depth: float = 50.0,
        safe_iteration_limit: int = 20,
    ) -> InversionResults:
        """
        Main iterative inversion loop. Runs until convergence or max_iterations.
        Returns InversionResults for error analysis.
        """
        config = self.config
        data = self.data

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

        logger.info(f"Locating {data.n_events} events with {data.n_stations} stations.")
        logger.info(f"Using phases: {config.phases}")
        logger.info(f"Using {solver} solver.")
        logger.info(f"Using {config.mode} mode for travel-time calculations.")
        logger.info(f"Distance weighting: {config.distance_weighting_enabled}")
        logger.info(f"Fix depth: {config.fix_depth}")
        logger.info(f"Precalculate velocities: {config.precalculate_velocities}")

        damping = 1.0
        max_damping = 10.0
        A = None
        residual_norms = []
        residuals_flat = np.array([])
        valid_rows = np.array([], dtype=int)

        prev_res = None

        for it in range(max_iterations):
            valid_rows = self.derivative_engine.populate_derivative_matrix()
            if len(valid_rows) == 0:
                logger.error("No valid observations available for inversion. Aborting.")
                break

            # Observed travel times = arrival times (RELATIVE) - origin times (RELATIVE)
            observed_travel_times = data.arrivals - data.origin_times[:, np.newaxis]

            # Residuals = observed - theoretical travel times per event-station-phase
            residuals = observed_travel_times - data.theoretical_travel_times
            residuals_flat = residuals.flatten()

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

            # Solve for the increment
            if solver == "lsqr":
                dx_solution = lsmr(A, b, damp=damping)[0]
            else:  # SVD with Tikhonov-style filtering
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                s_inv = np.zeros_like(s)
                s_inv[s > 1e-12] = s[s > 1e-12] / (s[s > 1e-12] ** 2 + damping**2)
                dx_solution = Vt.T @ (s_inv * (U.T @ b))

            # Reshape to event blocks and clip
            dx = dx_solution.reshape((data.n_events, config.dim))
            dx = _clip_updates(it, dx)

            # Residual AFTER step for the linearized subproblem
            residual_vector = A @ dx_solution - b  # r_new
            res_norm = np.linalg.norm(residual_vector)
            residual_norms.append(res_norm)
            rms = np.sqrt(np.mean(residual_vector**2))

            # Depth sanity / constraints
            for event_idx in range(data.n_events):
                if data.invalid_events[event_idx]:
                    continue
                new_depth = data.locations[event_idx, 2] - dx[event_idx, 2]
                if not solve_airquakes and new_depth < 0.0:
                    logger.warning(
                        f"Event {event_idx} depth would become negative ({new_depth} km). Mark invalid."
                    )
                    data.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0
                elif it > safe_iteration_limit and new_depth > max_depth:
                    logger.warning(
                        f"Event {event_idx} depth exceeds max depth ({new_depth} km > {max_depth} km). Mark invalid."
                    )
                    data.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0.0

            # Convert km -> deg for lat/lon updates; apply updates
            planet_radius = self.model_handler.model.model.radius_of_planet
            dx[:, 0] = kilometer2degrees(dx[:, 0], planet_radius)
            dx[:, 1] = kilometer2degrees(dx[:, 1], planet_radius)
            valid_mask = ~data.invalid_events
            data.locations[valid_mask, :3] -= dx[valid_mask, :3]
            data.origin_times[valid_mask] += dx[valid_mask, 3]

            logger.debug(
                f"Iteration {it + 1}: Residual norm={res_norm:.6f}, RMS={rms:.6f} s, Damping={damping:.6f}"
            )
            logger.debug(
                f"Iteration {it + 1}: Origin for first event: "
                f"Lat={data.locations[0, 0]:.4f}, Lon={data.locations[0, 1]:.4f}, "
                f"Depth={data.locations[0, 2]:.2f} km, "
                f"t0_rel={data.origin_times[0]:.2f}s, "
                f"t0_abs={(data.origin_times[0] + data.time_refs[0]):.2f}s"
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

        A_final = (
            data.derivative_matrix
            * data.time_error_W[:, np.newaxis]
            * data.distance_W[:, np.newaxis]
        )
        logger.info(f"Location iterations completed in {it + 1} iterations.")
        logger.info(
            f"Located {np.sum(~data.invalid_events)} events, {np.sum(data.invalid_events)} events invalid."
        )

        return InversionResults(
            A_final=A_final,
            valid_rows_final=valid_rows,
            residuals_flat_final=residuals_flat,
            lambda_final=damping,
        )
