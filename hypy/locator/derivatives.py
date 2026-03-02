import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import distance_weight
from .types import DerivativeContext, Matrices, Weights


def populate_derivative_matrix(
    *,
    ctx: DerivativeContext,
    matrices: Matrices,
    weights: Weights,
) -> dict:
    """Populate design matrix and theoretical travel times.

    Returns updated arrays and valid row indices.
    """
    valid_row_indices: list = []

    for event_index in range(ctx.n_events):
        if matrices.invalid_events[event_index]:
            continue

        ev_lat, ev_lon, ev_depth_km = matrices.locations[event_index]

        for station_index in range(ctx.n_stations):
            st_lat, st_lon, st_elev_m = ctx.stations_locations[station_index]

            for phase_index, phase_name in enumerate(ctx.phases):
                col0 = event_index * ctx.dim
                obs_col = ctx.n_phases * station_index + phase_index

                observed_time_s = matrices.arrivals[event_index, obs_col]
                if np.isnan(observed_time_s):
                    continue

                distance_km, phase_travel_time, (dt_dx, dt_dy, dt_dz) = (
                    ctx.tracer.compute_travel_parameters(
                        ev_lon,
                        ev_lat,
                        ev_depth_km,
                        st_lon,
                        st_lat,
                        st_elev_m,
                        phase_name,
                    )
                )
                if phase_travel_time is None:
                    logger.warning(
                        "No travel time for event %d, station %s, phase %s → skip.",
                        event_index,
                        ctx.stations[station_index],
                        phase_name,
                    )
                    continue

                matrices.theoretical_travel_times[event_index, obs_col] = (
                    phase_travel_time
                )

                row_index = event_index * (ctx.n_stations * ctx.n_phases) + obs_col

                if ctx.distance_weighting:
                    weights.distance_W[row_index] = distance_weight(
                        distance_km, k=ctx.k, x0=ctx.x0
                    )

                # follow convention: [y, x, z, t0] ordering
                matrices.derivative_matrix[row_index, col0 + 0] = dt_dy or 0.0
                matrices.derivative_matrix[row_index, col0 + 1] = dt_dx or 0.0
                if not ctx.fixed_depth_mask[event_index]:
                    matrices.derivative_matrix[row_index, col0 + 2] = dt_dz or 0.0
                matrices.derivative_matrix[row_index, col0 + 3] = 1.0

                valid_row_indices.append(row_index)

    return {
        "valid_rows": np.array(valid_row_indices, dtype=int),
        "theoretical_travel_times": matrices.theoretical_travel_times,
        "derivative_matrix": matrices.derivative_matrix,
        "distance_W": weights.distance_W,
        "time_error_W": weights.time_error_W,
    }
