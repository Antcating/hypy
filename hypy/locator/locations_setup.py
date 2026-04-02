from typing import Optional

from ..logger import logger  # type: ignore[import-not-found]
from .config import LocatorConfig
from .data import DataHandler
from .derivatives import DerivativeEngine
from .types import Event, np


def init_locations(
    data: DataHandler,
    config: LocatorConfig,
    derivative_engine: DerivativeEngine,
    trial_depth_km: float,
    max_depth_km: Optional[float],
) -> None:

    def _seed_candidates_for_event(
        event: Event, trial_depth_km: float, max_depth_km: Optional[float]
    ):
        """
        Build candidate epicenter seeds and a small depth grid for a given event.
        """
        seeds = []
        selected_station_indices: list[int] = []

        for pick in event.picks:
            net_sta_key = (
                pick.waveform_id.network_code,
                pick.waveform_id.station_code,
            )
            if net_sta_key in data.stations:
                station_index = data.stations.index(net_sta_key)
                selected_station_indices.append(station_index)

        selected_station_indices = sorted(set(selected_station_indices))
        if not selected_station_indices:
            return [], []

        station_latlon = np.asarray(
            [data.stations_locations[i][:2] for i in selected_station_indices],
            dtype=float,
        )

        # 1) centroid
        lat_centroid = float(np.mean(station_latlon[:, 0]))
        lon_centroid = float(np.mean(station_latlon[:, 1]))
        seeds.append((lat_centroid, lon_centroid))

        # 2) component-wise median
        lat_median = float(np.median(station_latlon[:, 0]))
        lon_median = float(np.median(station_latlon[:, 1]))
        seeds.append((lat_median, lon_median))

        # 3) earliest-pick station (by full scan)
        earliest_station_index = min(
            selected_station_indices,
            key=lambda idx: min(
                pk.time
                for pk in event.picks
                if (pk.waveform_id.network_code, pk.waveform_id.station_code)
                == data.stations[idx]
            ),
        )
        earliest_full_lat, earliest_full_lon = data.stations_locations[
            earliest_station_index
        ][:2]
        seeds.append((float(earliest_full_lat), float(earliest_full_lon)))

        # Deduplicate numerically
        unique_seeds: list[tuple[float, float]] = []
        seen_keys: set[tuple[float, float]] = set()
        for lat, lon in seeds:
            key = (round(lat, 4), round(lon, 4))
            if key not in seen_keys:
                unique_seeds.append((lat, lon))
                seen_keys.add(key)

        # Depth grid around trial depth
        initial_depth_km = max(0.0, float(trial_depth_km))
        if max_depth_km is None:
            # FIXME: It is a very dodgy assumption to set max_depth_km = initial_depth_km + 20.0,
            # should be based on the velocity model and typical depth distribution of events in the region instead.
            max_depth_km = initial_depth_km + 20.0

        if config.fix_depth_km is not None:
            # If depth is fixed, just use that as the single candidate
            candidate_depths = [float(config.fix_depth_km)]
            return unique_seeds, candidate_depths

        depth_grid_candidates = {
            round(x, 2)
            for x in np.random.normal(loc=initial_depth_km, scale=5.0, size=10)
        }

        candidate_depths = sorted(
            [d for d in depth_grid_candidates if 0.0 <= d <= max_depth_km]
        ) or [initial_depth_km]
        return unique_seeds, candidate_depths

    def _estimate_t0_and_score(ev_idx: int, lat: float, lon: float, depth_km: float):
        """
        Estimate origin time (t0, RELATIVE to time_refs) and WRMS for a candidate hypocenter.
        Returns (t0_rel, wrms, n_obs) or None.
        """
        # Collect all usable picks for the event — each item is (station_idx, phase_idx, t_obs_rel, sigma)
        picks = _usable_arrivals_for_event(ev_idx)

        # Require at least a minimal number of picks to compute a stable origin time
        if len(picks) < max(3, data.min_picks_per_event // 2):
            return None

        residual_arrival_times: list[float] | np.ndarray = []
        weights: list[float] | np.ndarray = []

        for (
            station_id,
            phase_id,
            observed_arrival_time_rel,
            measurement_error,
        ) in picks:
            # Get the station’s geographic position
            sta_lat, sta_lon, sta_elev = data.stations_locations[station_id]
            phase = data.phases[phase_id]

            # Compute theoretical travel time for this candidate source–station pair
            _, theoretical_arrival_time, _ = derivative_engine.derive_travel_time_param(
                (lat, lon, depth_km), (sta_lat, sta_lon, sta_elev), phase
            )

            # Skip if travel time couldn’t be computed or is invalid
            if theoretical_arrival_time is None or not np.isfinite(
                theoretical_arrival_time
            ):
                continue

            # Residual in RELATIVE frame: (t_obs_rel - t_theory)
            residual_arrival_times.append(
                observed_arrival_time_rel - float(theoretical_arrival_time)
            )

            # Weight: inverse of variance (1/σ²)
            weights.append(1.0 / (measurement_error * measurement_error))

        if len(residual_arrival_times) < max(3, data.min_picks_per_event // 2):
            return None

        residual_arrival_times = np.asarray(residual_arrival_times, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # Weighted average of (t_obs_rel - t_theory) gives t0_rel
        t0_rel = round(
            float(np.sum(weights * residual_arrival_times) / np.sum(weights)), 4
        )

        # Weighted RMS (WRMS)
        wrms = float(
            np.sqrt(
                np.sum(weights * (residual_arrival_times - t0_rel) ** 2)
                / np.sum(weights)
            )
        )

        return t0_rel, wrms, len(residual_arrival_times)

    def _usable_arrivals_for_event(event_id: int):
        usable = []
        for station_id in range(data.n_stations):
            for phase_id in range(data.n_phases):
                event_arrival_id = phase_id + station_id * data.n_phases
                arrival_time = data.arrivals[event_id, event_arrival_id]
                if not np.isnan(arrival_time):
                    sigma = data.arrival_time_errors[event_id, event_arrival_id]
                    if not (sigma is None or np.isnan(sigma) or sigma <= 0.0):
                        usable.append(
                            (
                                station_id,
                                phase_id,
                                float(arrival_time),  # RELATIVE arrival time
                                float(sigma),
                            )
                        )
        return usable

    # ---- Initialization loop using the seeds above ----
    for event_index, event in enumerate(data.events):
        if len(event.picks) == 0:
            continue

        # TODO: Move this station filtering logic into DataHandler
        # so that it’s not repeated for each event.
        # Keep only picks from stations we know about
        event.picks = [
            pick
            for pick in event.picks
            if (pick.waveform_id.network_code, pick.waveform_id.station_code)
            in data.stations
        ]

        seeds, candidate_depths = _seed_candidates_for_event(
            event, trial_depth_km, max_depth_km
        )
        logger.warning(
            f"Seeds: {seeds}, Depth candidates: {candidate_depths} for event {event_index}"
        )
        best_solution: tuple[float, float, float, float, float] | None = None
        # (wrms, lat, lon, depth_km, t0_rel)

        # Evaluate all seed/depth pairs, keep the one with the lowest WRMS
        for lat_seed, lon_seed in seeds:
            for depth_candidate_km in candidate_depths:
                logger.debug(
                    f"Evaluating seed (lat={lat_seed:.4f}, lon={lon_seed:.4f}, "
                    f"depth={depth_candidate_km:.2f} km) for event {event_index}"
                )
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

                    logger.debug(
                        f"New best solution for event {event_index}: "
                        f"WRMS={wrms:.3f}, lat={lat_seed:.4f}, lon={lon_seed:.4f}, "
                        f"depth={depth_candidate_km:.2f} km, t0_rel={t0_estimate_rel:.3f}s"
                    )

        if best_solution is not None:
            _, lat_best, lon_best, depth_best_km, t0_best_rel = best_solution
            data.locations[event_index] = (lat_best, lon_best, depth_best_km)
            data.origin_times[event_index] = t0_best_rel
            t0_abs = t0_best_rel + data.time_refs[event_index]
            logger.debug(
                f"[init] ev#{event_index}: lat={lat_best:.4f} lon={lon_best:.4f} "
                f"z={depth_best_km:.2f} km, t0_rel={t0_best_rel:.3f}s, t0_abs={t0_abs:.3f}s"
            )
        else:
            # Fallback: use the station of the earliest pick and estimate origin time
            first_pick = min(event.picks, key=lambda pk: pk.time)
            net_code = first_pick.waveform_id.network_code
            sta_code = first_pick.waveform_id.station_code
            station_meta = data.inv.select(network=net_code, station=sta_code)[0][0]
            data.locations[event_index] = (
                station_meta.latitude,
                station_meta.longitude,
                trial_depth_km,
            )
            # Rough P travel-time estimate from trial depth
            estimated_travel_time = trial_depth_km / 6.0
            # store t0 **relative** to per-event reference
            data.origin_times[event_index] = (
                float(first_pick.time) - data.time_refs[event_index]
            ) - estimated_travel_time
            t0_abs = data.origin_times[event_index] + data.time_refs[event_index]
            logger.debug(
                f"[init:fallback] ev#{event_index}: lat={station_meta.latitude:.4f} "
                f"lon={station_meta.longitude:.4f} z={trial_depth_km:.2f} km, "
                f"t0_rel≈first_pick-{estimated_travel_time:.2f}s, t0_abs={t0_abs:.3f}s"
            )
