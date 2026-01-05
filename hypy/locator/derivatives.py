import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import distance_weight
from ..propagation import PropagationModel


def populate_derivative_matrix(
    *,
    tracer: PropagationModel,
    locations: np.ndarray,
    stations_locations: np.ndarray,
    stations,
    phases,
    n_events: int,
    n_stations: int,
    n_phases: int,
    dim: int,
    arrivals: np.ndarray,
    theoretical_travel_times: np.ndarray,
    derivative_matrix: np.ndarray,
    distance_W: np.ndarray,
    time_error_W: np.ndarray,
    invalid_events: np.ndarray,
    fixed_depth_mask,
    distance_weighting: bool,
    k,
    x0,
) -> dict:
    """Populate design matrix and theoretical travel times.

    Returns updated arrays and valid row indices.
    """
    valid_row_indices: list = []

    for event_index in range(n_events):
        if invalid_events[event_index]:
            continue

        ev_lat, ev_lon, ev_depth_km = locations[event_index]

        for station_index in range(n_stations):
            st_lat, st_lon, st_elev_m = stations_locations[station_index]

            for phase_index, phase_name in enumerate(phases):
                col0 = event_index * dim
                obs_col = n_phases * station_index + phase_index

                observed_time_s = arrivals[event_index, obs_col]
                if np.isnan(observed_time_s):
                    continue

                distance_km, phase_travel_time, (dt_dx, dt_dy, dt_dz) = (
                    tracer.compute_travel_parameters(
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
                        stations[station_index],
                        phase_name,
                    )
                    continue

                theoretical_travel_times[event_index, obs_col] = phase_travel_time

                row_index = event_index * (n_stations * n_phases) + obs_col

                if distance_weighting:
                    distance_W[row_index] = distance_weight(distance_km, k=k, x0=x0)

                # follow convention: [y, x, z, t0] ordering
                derivative_matrix[row_index, col0 + 0] = dt_dy or 0.0
                derivative_matrix[row_index, col0 + 1] = dt_dx or 0.0
                if not fixed_depth_mask[event_index]:
                    derivative_matrix[row_index, col0 + 2] = dt_dz or 0.0
                derivative_matrix[row_index, col0 + 3] = 1.0

                valid_row_indices.append(row_index)

    return {
        "valid_rows": np.array(valid_row_indices, dtype=int),
        "theoretical_travel_times": theoretical_travel_times,
        "derivative_matrix": derivative_matrix,
        "distance_W": distance_W,
        "time_error_W": time_error_W,
    }
