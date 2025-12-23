import copy
from typing import List, Optional, Sequence, Tuple

from ..logger import logger  # type: ignore[import-not-found]
from .types import Event, Inventory, np


def init_locations(
    *,
    inventory: Inventory,
    events: List[Event],
    stations: List[Tuple[str, str]],
    stations_locations: np.ndarray,
    phases: Sequence[str],
    n_stations: int,
    n_phases: int,
    min_picks_per_event: int,
    max_depth: Optional[float],
    tracer,
    trial_depth: float = 10.0,
    trial_time: float = 2.0,
    arrivals: np.ndarray,
    arrival_time_errors: np.ndarray,
    locations: np.ndarray,
    origin_times: np.ndarray,
    time_refs: np.ndarray,
) -> dict:
    """Smart multi-seed initialization; falls back to a legacy heuristic."""

    events_list = copy.deepcopy(list(events))

    def _usable_arrivals_for_event(event_id: int):
        usable = []
        for station_id in range(n_stations):
            for phase_id in range(n_phases):
                event_arrival_id = phase_id + station_id * n_phases
                arrival_time = arrivals[event_id, event_arrival_id]
                if not np.isnan(arrival_time):
                    sigma = arrival_time_errors[event_id, event_arrival_id]
                    if not (sigma is None or np.isnan(sigma) or sigma <= 0.0):
                        usable.append(
                            (
                                station_id,
                                phase_id,
                                float(arrival_time),
                                float(sigma),
                            )
                        )
        return usable

    def _estimate_t0_and_score(ev_idx: int, lat: float, lon: float, depth_km: float):
        picks = _usable_arrivals_for_event(ev_idx)
        if len(picks) < max(3, min_picks_per_event // 2):
            return None

        residual_arrival_times: list[float] = []
        weights_list: list[float] = []

        for station_id, phase_id, observed_arrival_time_rel, measurement_error in picks:
            sta_lat, sta_lon, sta_elev = stations_locations[station_id]
            phase = phases[phase_id]

            _, theoretical_arrival_time, _ = tracer.calculate_arrival_parameters(
                lon, lat, depth_km, sta_lon, sta_lat, sta_elev, phase
            )
            if theoretical_arrival_time is None or not np.isfinite(
                theoretical_arrival_time
            ):
                continue

            residual_arrival_times.append(
                observed_arrival_time_rel - float(theoretical_arrival_time)
            )
            weights_list.append(1.0 / (measurement_error * measurement_error))

        if len(residual_arrival_times) < max(3, min_picks_per_event // 2):
            return None

        residuals = np.asarray(residual_arrival_times, dtype=float)
        weights = np.asarray(weights_list, dtype=float)

        t0_rel = round(float(np.sum(weights * residuals) / np.sum(weights)), 4)
        wrms = float(
            np.sqrt(np.sum(weights * (residuals - t0_rel) ** 2) / np.sum(weights))
        )
        return t0_rel, wrms, int(residuals.size)

    def _seed_candidates_for_event(event_index: int, trial_depth_km: float):
        seeds: list[tuple[float, float]] = []
        selected_station_indices: list[int] = []

        for pick in events_list[event_index].picks:
            net_sta_key = (pick.waveform_id.network_code, pick.waveform_id.station_code)
            if net_sta_key in stations:
                selected_station_indices.append(stations.index(net_sta_key))

        selected_station_indices = sorted(set(selected_station_indices))
        if not selected_station_indices:
            return [], []

        station_latlon = np.asarray(
            [stations_locations[i][:2] for i in selected_station_indices],
            dtype=float,
        )

        seeds.append(
            (float(np.mean(station_latlon[:, 0])), float(np.mean(station_latlon[:, 1])))
        )
        seeds.append(
            (
                float(np.median(station_latlon[:, 0])),
                float(np.median(station_latlon[:, 1])),
            )
        )

        earliest_station_index = min(
            selected_station_indices,
            key=lambda idx: min(
                pk.time
                for pk in events_list[event_index].picks
                if (pk.waveform_id.network_code, pk.waveform_id.station_code)
                == stations[idx]
            ),
        )
        earliest_full_lat, earliest_full_lon = stations_locations[
            earliest_station_index
        ][:2]
        seeds.append((float(earliest_full_lat), float(earliest_full_lon)))

        unique_seeds: list[tuple[float, float]] = []
        seen_keys: set[tuple[float, float]] = set()
        for lat_s, lon_s in seeds:
            key = (round(lat_s, 4), round(lon_s, 4))
            if key not in seen_keys:
                unique_seeds.append((lat_s, lon_s))
                seen_keys.add(key)

        initial_depth_km = max(0.0, float(trial_depth_km))
        max_depth_km = (
            float(max_depth) if max_depth is not None else initial_depth_km + 20.0
        )
        depth_grid_candidates = {
            round(x, 2)
            for x in [
                initial_depth_km * 0.5,
                initial_depth_km,
                initial_depth_km * 1.5,
                initial_depth_km + 5.0,
                initial_depth_km + 10.0,
            ]
        }
        candidate_depths = sorted(
            [d for d in depth_grid_candidates if 0.0 <= d <= max_depth_km]
        ) or [initial_depth_km]
        return unique_seeds, candidate_depths

    for event_index, event in enumerate(events_list):
        if len(event.picks) == 0:
            continue

        event.picks = [
            pick
            for pick in event.picks
            if (pick.waveform_id.network_code, pick.waveform_id.station_code)
            in stations
        ]

        seeds, candidate_depths = _seed_candidates_for_event(event_index, trial_depth)
        best_solution: Optional[Tuple[float, float, float, float, float]] = None

        for lat_seed, lon_seed in seeds:
            for depth_candidate_km in candidate_depths:
                scored = _estimate_t0_and_score(
                    event_index, lat_seed, lon_seed, depth_candidate_km
                )
                if scored is None:
                    continue
                t0_estimate_rel, wrms, _ = scored
                if np.isfinite(wrms) and (
                    best_solution is None or wrms < best_solution[0]
                ):
                    best_solution = (
                        wrms,
                        lat_seed,
                        lon_seed,
                        depth_candidate_km,
                        t0_estimate_rel,
                    )

        if best_solution is not None:
            _, lat_best, lon_best, depth_best_km, t0_best_rel = best_solution
            locations[event_index] = (lat_best, lon_best, depth_best_km)
            origin_times[event_index] = t0_best_rel
            t0_abs = t0_best_rel + time_refs[event_index]
            logger.debug(
                f"[init] ev#{event_index}: lat={lat_best:.4f} lon={lon_best:.4f} "
                f"z={depth_best_km:.2f} km, t0_rel={t0_best_rel:.3f}s, t0_abs={t0_abs:.3f}s"
            )
        else:
            first_pick = min(event.picks, key=lambda pk: pk.time)
            net_code = first_pick.waveform_id.network_code
            sta_code = first_pick.waveform_id.station_code
            station_meta = inventory.select(network=net_code, station=sta_code)[0][0]
            locations[event_index] = (
                station_meta.latitude,
                station_meta.longitude,
                trial_depth,
            )
            origin_times[event_index] = (
                float(first_pick.time) - time_refs[event_index]
            ) - trial_time
            t0_abs = origin_times[event_index] + time_refs[event_index]
            logger.debug(
                f"[init:fallback] ev#{event_index}: lat={station_meta.latitude:.4f} "
                f"lon={station_meta.longitude:.4f} z={trial_depth:.2f} km, "
                f"t0_rel≈first_pick-{trial_time:.2f}s, t0_abs={t0_abs:.3f}s"
            )
    return {
        "events": events_list,
        "locations": locations,
        "origin_times": origin_times,
    }
