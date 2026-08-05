"""Seed and initialize event locations before inversion."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from hypy.logger import logger  # type: ignore[import-not-found]
from hypy.propagation import PropagationEngine

from .config import LocatorConfig, RunConfig
from .data import DataHandler

# Hardcoded seeding knobs (TODO: Document hardcoded values).
_DEPTH_SAMPLE_SCALE_KM = 5.0
_DEPTH_SAMPLE_SIZE = 10
_DEPTH_BOUND_OFFSET_KM = 20.0


def _known_station_picks(event_picks: Sequence, stations: list) -> list:
    """Return picks whose network/station keys exist in ``stations`` (non-mutating)."""
    station_set = set(stations)
    return [
        pick
        for pick in event_picks
        if (pick.waveform_id.network_code, pick.waveform_id.station_code) in station_set
    ]


def _station_indices_for_picks(picks: Sequence, data: DataHandler) -> list[int]:
    """Map picks to known station indices (deduplicated, sorted)."""
    selected: list[int] = []
    for pick in picks:
        net_sta_key = (
            pick.waveform_id.network_code,
            pick.waveform_id.station_code,
        )
        if net_sta_key in data.stations:
            selected.append(data.stations.index(net_sta_key))
    return sorted(set(selected))


def _epicenter_seeds(
    picks: Sequence, data: DataHandler, station_indices: list[int]
) -> list[tuple[float, float]]:
    """Build centroid, median, and earliest-pick epicenter seeds."""
    if not station_indices:
        return []

    station_latlon = np.asarray(
        [data.stations_locations[i][:2] for i in station_indices],
        dtype=float,
    )

    seeds: list[tuple[float, float]] = [
        (float(np.mean(station_latlon[:, 0])), float(np.mean(station_latlon[:, 1]))),
        (
            float(np.median(station_latlon[:, 0])),
            float(np.median(station_latlon[:, 1])),
        ),
    ]

    earliest_station_index = min(
        station_indices,
        key=lambda idx: min(
            pk.time
            for pk in picks
            if (pk.waveform_id.network_code, pk.waveform_id.station_code)
            == data.stations[idx]
        ),
    )
    earliest_lat, earliest_lon = data.stations_locations[earliest_station_index][:2]
    seeds.append((float(earliest_lat), float(earliest_lon)))

    unique_seeds: list[tuple[float, float]] = []
    seen_keys: set[tuple[float, float]] = set()
    for lat, lon in seeds:
        key = (
            round(lat, 4),
            round(lon, 4),
        )
        if key not in seen_keys:
            unique_seeds.append((lat, lon))
            seen_keys.add(key)
    return unique_seeds


def _depth_candidates(
    trial_depth_km: float,
    depth_step_km: Optional[float],
    max_depth_km: Optional[float],
    fix_depth_km: Optional[float],
    rng: np.random.Generator,
) -> list[float]:
    """Build a small depth grid around the trial depth."""
    if fix_depth_km is not None:
        return [float(fix_depth_km)]

    initial_depth_km = max(0.0, float(trial_depth_km))
    if max_depth_km is None:
        # FIXME: This heuristic should be derived from the velocity model
        # and regional event-depth statistics instead of a fixed offset.
        max_depth_km = initial_depth_km + _DEPTH_BOUND_OFFSET_KM

    # TODO: Document hardcoded upper bound
    min_depth = max(1, depth_step_km) if depth_step_km is not None else 1
    # FIXME: This random sampling sometimes produces candidates with no valid tt.
    depth_grid_candidates = {
        round(x, 2)
        for x in rng.normal(
            loc=initial_depth_km,
            scale=_DEPTH_SAMPLE_SCALE_KM,
            size=_DEPTH_SAMPLE_SIZE,
        )
        if min_depth < x < max_depth_km
    }

    return sorted(
        [d for d in depth_grid_candidates if min_depth <= d <= max_depth_km]
    ) or [initial_depth_km]


def _usable_arrivals(
    data: DataHandler, event_id: int
) -> list[tuple[int, int, float, float]]:
    """Collect ``(station_id, phase_id, t_obs_rel, sigma)`` for an event."""
    usable: list[tuple[int, int, float, float]] = []
    for station_id in range(data.n_stations):
        for phase_id in range(data.n_phases):
            event_arrival_id = phase_id + station_id * data.n_phases
            arrival_time = data.arrivals[event_id, event_arrival_id]
            if np.isnan(arrival_time):
                continue
            sigma = data.arrival_time_errors[event_id, event_arrival_id]
            if sigma is None or np.isnan(sigma) or sigma <= 0.0:
                continue
            usable.append(
                (
                    station_id,
                    phase_id,
                    float(arrival_time),
                    float(sigma),
                )
            )
    return usable


def _estimate_t0_and_score(
    data: DataHandler,
    config: LocatorConfig,
    propagation_engine: PropagationEngine,
    event_id: int,
    lat: float,
    lon: float,
    depth_km: float,
) -> tuple[float, float, int] | None:
    """Estimate origin time and weighted RMS for a candidate hypocenter.

    Returns
    -------
    tuple of float or None
        ``(t0_rel, wrms, n_obs)`` when enough usable arrivals are present,
        otherwise ``None``. The origin time is relative to the per-event
        reference time.
    """
    picks = _usable_arrivals(data, event_id)
    min_needed = max(3, data.min_picks_per_event // 2)
    if len(picks) < min_needed:
        return None

    residual_arrival_times: list[float] = []
    weights: list[float] = []

    for station_id, phase_id, observed_arrival_time_rel, measurement_error in picks:
        sta_lat, sta_lon, sta_elev = data.stations_locations[station_id]
        phase = data.phases[phase_id]

        _, theoretical_arrival_time, _ = propagation_engine.compute_travel_time_params(
            source_latitude=lat,
            source_longitude=lon,
            source_depth_km=depth_km,
            receiver_latitude=sta_lat,
            receiver_longitude=sta_lon,
            receiver_elevation_m=sta_elev,
            phase=phase,
            delta=config.derivative_delta_km,
            derivative_method=config.derivative_method,
        )

        if theoretical_arrival_time is None or not np.isfinite(
            theoretical_arrival_time
        ):
            continue

        residual_arrival_times.append(
            observed_arrival_time_rel - float(theoretical_arrival_time)
        )
        weights.append(1.0 / (measurement_error * measurement_error))

    if len(residual_arrival_times) < min_needed:
        return None

    residual_arr = np.asarray(residual_arrival_times, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)

    t0_rel = round(float(np.sum(weight_arr * residual_arr) / np.sum(weight_arr)), 4)
    wrms = float(
        np.sqrt(np.sum(weight_arr * (residual_arr - t0_rel) ** 2) / np.sum(weight_arr))
    )
    return t0_rel, wrms, len(residual_arrival_times)


class LocationSeeder:
    """Initialize event hypocenters and origin-time seeds before inversion.

    Parameters
    ----------
    config : LocatorConfig
        Locator configuration controlling depth handling and pick filtering.
    data : DataHandler
        Shared event data container that receives initial locations.
    run_config : RunConfig
        Per-run configuration providing the RNG seed for reproducible seeding.
    propagation_engine : PropagationEngine or None
        Optional forward model used to score seed/depth candidates via travel
        times. Prefer a cheap precomputed table. When ``None``, seeding uses
        heuristic epicenter seeds and ``trial_depth_km`` without travel-time
        scoring.
    """

    def __init__(
        self,
        config: LocatorConfig,
        data: DataHandler,
        run_config: RunConfig,
        propagation_engine: Optional[PropagationEngine] = None,
    ) -> None:
        self.config = config
        self.data = data
        self.run_config = run_config
        self.propagation_engine = propagation_engine

    def seed(
        self,
        trial_depth_km: float,
        depth_step_km: Optional[float],
        max_depth_km: Optional[float],
    ) -> None:
        """Initialize event locations and origin-time seeds in place.

        Parameters
        ----------
        trial_depth_km : float
            Initial depth guess in kilometers.
        depth_step_km : float or None
            Minimum depth spacing used when building the candidate depth grid.
        max_depth_km : float or None
            Upper bound for candidate depths. When ``None``, a heuristic bound
            is derived from ``trial_depth_km``.

        Notes
        -----
        Uses a small set of deterministic epicenter seeds plus a lightweight
        randomized depth sample. Random depths are drawn from a local generator
        seeded with ``run_config.seed`` so a fixed seed makes the run
        reproducible. When ``propagation_engine`` is set, candidates are scored
        by weighted RMS of travel-time residuals; otherwise the first seed and
        trial depth are used with origin time at the earliest pick.
        """
        data = self.data
        config = self.config
        rng = np.random.default_rng(self.run_config.seed)

        for event_index, event in enumerate(data.events):
            if len(event.picks) == 0:
                continue

            known_picks = _known_station_picks(event.picks, data.stations)
            station_indices = _station_indices_for_picks(known_picks, data)
            seeds = _epicenter_seeds(known_picks, data, station_indices)
            candidate_depths = _depth_candidates(
                trial_depth_km,
                depth_step_km,
                max_depth_km,
                config.fix_depth_km,
                rng,
            )
            logger.debug(
                "Seeds: %s, Depth candidates: %s for event %d",
                seeds,
                candidate_depths,
                event_index,
            )

            if not seeds:
                logger.warning(
                    "Event %d has no picks from known stations; skipping seed.",
                    event_index,
                )
                continue

            if self.propagation_engine is None:
                depth_km = (
                    trial_depth_km
                    if config.fix_depth_km is None
                    else float(config.fix_depth_km)
                )
                self._apply_heuristic_seed(event_index, known_picks, seeds[0], depth_km)
                continue

            best_solution = self._score_candidates(event_index, seeds, candidate_depths)
            if best_solution is None:
                logger.warning(
                    "Event %d: no scored seed candidate; skipping seed.",
                    event_index,
                )
                continue

            _, lat_best, lon_best, depth_best_km, t0_best_rel = best_solution
            data.locations[event_index] = (lat_best, lon_best, depth_best_km)
            data.origin_times[event_index] = t0_best_rel
            t0_abs = t0_best_rel + data.time_refs[event_index]
            logger.debug(
                "[init] ev#%d: lat=%.4f lon=%.4f z=%.2f km, t0_rel=%.3fs, t0_abs=%.3fs",
                event_index,
                lat_best,
                lon_best,
                depth_best_km,
                t0_best_rel,
                t0_abs,
            )

    def _apply_heuristic_seed(
        self,
        event_index: int,
        known_picks: Sequence,
        seed: tuple[float, float],
        depth_km: float,
    ) -> None:
        """Assign first epicenter seed + depth without travel-time scoring."""
        lat, lon = seed
        self.data.locations[event_index] = (lat, lon, depth_km)
        first_pick = min(known_picks, key=lambda pk: pk.time)
        self.data.origin_times[event_index] = (
            float(first_pick.time) - self.data.time_refs[event_index]
        )
        t0_abs = self.data.origin_times[event_index] + self.data.time_refs[event_index]
        logger.debug(
            "[init:heuristic] ev#%d: lat=%.4f lon=%.4f z=%.2f km, t0_abs=%.3fs",
            event_index,
            lat,
            lon,
            depth_km,
            t0_abs,
        )

    def _score_candidates(
        self,
        event_index: int,
        seeds: list[tuple[float, float]],
        candidate_depths: list[float],
    ) -> tuple[float, float, float, float, float] | None:
        """Evaluate seed/depth pairs; return best ``(wrms, lat, lon, depth, t0)``."""
        assert self.propagation_engine is not None
        best_solution: tuple[float, float, float, float, float] | None = None

        for lat_seed, lon_seed in seeds:
            for depth_candidate_km in candidate_depths:
                logger.debug(
                    "Evaluating seed (lat=%.4f, lon=%.4f, depth=%.2f km) for event %d",
                    lat_seed,
                    lon_seed,
                    depth_candidate_km,
                    event_index,
                )
                scored = _estimate_t0_and_score(
                    self.data,
                    self.config,
                    self.propagation_engine,
                    event_index,
                    lat_seed,
                    lon_seed,
                    depth_candidate_km,
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
                        "New best solution for event %d: WRMS=%.3f, lat=%.4f, "
                        "lon=%.4f, depth=%.2f km, t0_rel=%.3fs",
                        event_index,
                        wrms,
                        lat_seed,
                        lon_seed,
                        depth_candidate_km,
                        t0_estimate_rel,
                    )
        return best_solution
