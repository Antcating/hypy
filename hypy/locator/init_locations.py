from .types import np
from ..logger import logger  # type: ignore[import-not-found]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import LocatorState


class InitLocationsMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # cooperative MRO

    def init_locations(
        self: "LocatorState", trial_depth: float = 10.0, trial_time: float = 2.0
    ):
        """Smart multi-seed initialization; falls back to legacy heuristic."""

        def _usable_arrivals_for_event(event_id: int):
            usable = []
            for station_id in range(self.n_stations):
                for phase_id in range(self.n_phases):
                    event_arrival_id = phase_id + station_id * self.n_phases
                    arrival_time = self.arrivals[event_id, event_arrival_id]
                    if not np.isnan(arrival_time):
                        sigma = self.arrival_time_errors[event_id, event_arrival_id]
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

        def _estimate_t0_and_score(
            ev_idx: int, lat: float, lon: float, depth_km: float
        ):
            """
            Estimate origin time (t0, RELATIVE to time_refs) and WRMS for a candidate hypocenter.
            Returns (t0_rel, wrms, n_obs) or None.
            """

            # Collect all usable picks for the event — each item is (station_idx, phase_idx, t_obs_rel, sigma)
            picks = _usable_arrivals_for_event(ev_idx)

            # Require at least a minimal number of picks to compute a stable origin time
            if len(picks) < max(3, self.min_picks_per_event // 2):
                return None

            residual_arrival_times: list[float] = []
            weights: list[float] = []

            for (
                station_id,
                phase_id,
                observed_arrival_time_rel,
                measurement_error,
            ) in picks:
                # Get the station’s geographic position
                sta_lat, sta_lon, sta_elev = self.stations_locations[station_id]
                phase = self.phases[phase_id]

                # Compute theoretical travel time for this candidate source–station pair
                _, theoretical_arrival_time, _ = self._derive_travel_time_param(
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

            if len(residual_arrival_times) < max(3, self.min_picks_per_event // 2):
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

        def _seed_candidates_for_event(event_index: int, trial_depth_km: float):
            """
            Build candidate epicenter seeds and a small depth grid for a given event.
            """
            seeds = []
            selected_station_indices: list[int] = []

            for pick in self.events[event_index].picks:
                net_sta_key = (
                    pick.waveform_id.network_code,
                    pick.waveform_id.station_code,
                )
                if net_sta_key in self.stations:
                    station_index = self.stations.index(net_sta_key)
                    selected_station_indices.append(station_index)

            selected_station_indices = sorted(set(selected_station_indices))
            if not selected_station_indices:
                return [], []

            station_latlon = np.asarray(
                [self.stations_locations[i][:2] for i in selected_station_indices],
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
                    for pk in self.events[event_index].picks
                    if (pk.waveform_id.network_code, pk.waveform_id.station_code)
                    == self.stations[idx]
                ),
            )
            earliest_full_lat, earliest_full_lon = self.stations_locations[
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
            max_depth_km = (
                float(self.max_depth)
                if self.max_depth is not None
                else initial_depth_km + 20.0
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

        # ---- Initialization loop using the seeds above ----
        for event_index, event in enumerate(self.events):
            if len(event.picks) == 0:
                continue

            # Keep only picks from stations we know about
            event.picks = [
                pick
                for pick in event.picks
                if (pick.waveform_id.network_code, pick.waveform_id.station_code)
                in self.stations
            ]

            seeds, candidate_depths = _seed_candidates_for_event(
                event_index, trial_depth
            )
            best_solution: tuple[float, float, float, float, float] | None = None
            # (wrms, lat, lon, depth_km, t0_rel)

            # Evaluate all seed/depth pairs, keep the one with the lowest WRMS
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
                self.locations[event_index] = (lat_best, lon_best, depth_best_km)
                self.origin_times[event_index] = t0_best_rel
                t0_abs = t0_best_rel + self.time_refs[event_index]
                logger.debug(
                    f"[init] ev#{event_index}: lat={lat_best:.4f} lon={lon_best:.4f} "
                    f"z={depth_best_km:.2f} km, t0_rel={t0_best_rel:.3f}s, t0_abs={t0_abs:.3f}s"
                )
            else:
                # Fallback: use the station of the earliest pick and back off origin time by trial_time
                first_pick = min(event.picks, key=lambda pk: pk.time)
                net_code = first_pick.waveform_id.network_code
                sta_code = first_pick.waveform_id.station_code
                station_meta = self.inv.select(network=net_code, station=sta_code)[0][0]
                self.locations[event_index] = (
                    station_meta.latitude,
                    station_meta.longitude,
                    trial_depth,
                )
                # store t0 **relative** to per-event reference
                self.origin_times[event_index] = (
                    float(first_pick.time) - self.time_refs[event_index]
                ) - trial_time
                t0_abs = self.origin_times[event_index] + self.time_refs[event_index]
                logger.debug(
                    f"[init:fallback] ev#{event_index}: lat={station_meta.latitude:.4f} "
                    f"lon={station_meta.longitude:.4f} z={trial_depth:.2f} km, "
                    f"t0_rel≈first_pick-{trial_time:.2f}s, t0_abs={t0_abs:.3f}s"
                )

        return self.locations
