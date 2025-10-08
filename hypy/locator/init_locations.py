from .types import np
from ..logger import logger  # type: ignore[import-not-found]

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state import LocatorState

class InitLocationsMixin:
    def init_locations(self: "LocatorState", trial_depth: float = 10.0, trial_time: float = 2.0):
        """Smart multi-seed initialization; falls back to legacy heuristic."""
        def _usable_arrivals_for_event(ev_idx: int):
            usable = []
            for s in range(self.n_stations):
                for p in range(self.n_phases):
                    j = p + s * self.n_phases
                    t_obs = self.arrivals[ev_idx, j]
                    if not np.isnan(t_obs):
                        sigma = self.arrival_time_errors[ev_idx, j]
                        if not (sigma is None or np.isnan(sigma) or sigma <= 0.0):
                            usable.append((s, p, float(t_obs), float(sigma)))
            return usable

        def _estimate_t0_and_score(ev_idx: int, lat: float, lon: float, depth_km: float):
            items = _usable_arrivals_for_event(ev_idx)
            if len(items) < max(3, self.min_picks_per_event // 2):
                return None
            obs_minus_tt, w = [], []
            for s, p, t_obs, sigma in items:
                sta_lat, sta_lon, sta_elev = self.stations_locations[s]
                phase = self.phases[p]
                _, t_theory, _ = self._derive_travel_time_param(
                    (lat, lon, depth_km), (sta_lat, sta_lon, sta_elev), phase
                )
                if t_theory is None or not np.isfinite(t_theory):
                    continue
                obs_minus_tt.append(t_obs - float(t_theory))
                w.append(1.0 / (sigma * sigma))
            if len(obs_minus_tt) < max(3, self.min_picks_per_event // 2):
                return None
            obs_minus_tt = np.asarray(obs_minus_tt, dtype=float)
            w = np.asarray(w, dtype=float)
            t0 = float(np.sum(w * obs_minus_tt) / np.sum(w))
            wrms = float(np.sqrt(np.sum(w * (obs_minus_tt - t0) ** 2) / np.sum(w)))
            return t0, wrms, len(obs_minus_tt)

        def _seed_candidates_for_event(event_idx: int, trial_depth_km: float):
            picked, t_first, earliest_idx = [], float("inf"), None
            for pk in self.events[event_idx].picks:
                key = (pk.waveform_id.network_code, pk.waveform_id.station_code)
                if key in self.stations:
                    sidx = self.stations.index(key)
                    picked.append(sidx)
                    if pk.time < t_first:
                        t_first, earliest_idx = pk.time, sidx
            picked = sorted(set(picked))
            if not picked:
                return [], []

            pts = np.asarray([self.stations_locations[i][:2] for i in picked], float)
            lat_c, lon_c = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            lat_m, lon_m = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
            seeds = [(lat_c, lon_c), (lat_m, lon_m)]
            if earliest_idx is not None:
                e_lat, e_lon = self.stations_locations[earliest_idx][:2]
                seeds.append((float(e_lat), float(e_lon)))

            earliest_pick_idx = min(
                picked,
                key=lambda idx: min(
                    pk.time for pk in self.events[event_idx].picks
                    if (pk.waveform_id.network_code, pk.waveform_id.station_code) == self.stations[idx]
                ),
            )
            earliest_lat, earliest_lon = self.stations_locations[earliest_pick_idx][:2]
            seeds.append((float(earliest_lat), float(earliest_lon)))

            uniq, seen = [], set()
            for lat, lon in seeds:
                key = (round(lat, 4), round(lon, 4))
                if key not in seen:
                    uniq.append((lat, lon)); seen.add(key)

            d0 = max(0.0, float(trial_depth_km))
            dmax = float(self.max_depth) if self.max_depth is not None else d0 + 20.0
            grid = {round(x, 2) for x in [d0 * 0.5, d0, d0 * 1.5, d0 + 5.0, d0 + 10.0]}
            depths = sorted([d for d in grid if 0.0 <= d <= dmax]) or [d0]
            return uniq, depths

        for event_idx, event in enumerate(self.events):
            if len(event.picks) == 0:
                continue
            event.picks = [
                pick for pick in event.picks
                if (pick.waveform_id.network_code, pick.waveform_id.station_code) in self.stations
            ]

            seeds, depths = _seed_candidates_for_event(event_idx, trial_depth)
            best = None  # (wrms, lat, lon, depth_km, t0)

            for lat, lon in seeds:
                for depth_km in depths:
                    scored = _estimate_t0_and_score(event_idx, lat, lon, depth_km)
                    if scored is None:
                        continue
                    t0, wrms, _ = scored
                    if np.isfinite(wrms) and (best is None or wrms < best[0]):
                        best = (wrms, lat, lon, depth_km, t0)

            if best is not None:
                _, lat, lon, depth_km, t0 = best
                self.locations[event_idx] = (lat, lon, depth_km)
                self.origin_times[event_idx] = t0
                logger.debug(f"[init] ev#{event_idx}: lat={lat:.4f} lon={lon:.4f} z={depth_km:.2f} km, t0={t0:.3f}")
            else:
                first_pick = min(event.picks, key=lambda pk: pk.time)
                net_code = first_pick.waveform_id.network_code
                sta_code = first_pick.waveform_id.station_code
                station = self.inv.select(network=net_code, station=sta_code)[0][0]
                self.locations[event_idx] = (station.latitude, station.longitude, trial_depth)
                self.origin_times[event_idx] = first_pick.time - trial_time
                logger.debug(
                    f"[init:fallback] ev#{event_idx}: lat={station.latitude:.4f} "
                    f"lon={station.longitude:.4f} z={trial_depth:.2f} km, t0≈first_pick−{trial_time:.2f}s"
                )
        return self.locations
