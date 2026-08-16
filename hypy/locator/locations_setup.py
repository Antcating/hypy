"""Seed and initialize event locations before inversion."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from obspy.geodetics import (  # type: ignore[import-untyped]
    degrees2kilometers,
    kilometer2degrees,
)

from hypy.logger import logger  # type: ignore[import-not-found]
from hypy.propagation import PropagationEngine
from hypy.utils.geo import gps2dist_azimuth_nb

from .config import LocatorConfig, RunConfig
from .data import DataHandler

_MIN_COS_LAT = 1e-6


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


def _config_search_bounds(
    config: LocatorConfig,
) -> Optional[tuple[float, float, float, float]]:
    """Return config search box ``(lat_min, lat_max, lon_min, lon_max)`` or ``None``."""
    return config.search_bounds


def _station_bbox_bounds(
    station_latlons: Sequence[tuple[float, float]],
    distance_step_km: float,
    max_distance_km: float,
) -> tuple[float, float, float, float]:
    """Return a padded geographic envelope around event stations."""
    arr = np.asarray(station_latlons, dtype=float)
    lat_min = float(arr[:, 0].min())
    lat_max = float(arr[:, 0].max())
    lon_min = float(arr[:, 1].min())
    lon_max = float(arr[:, 1].max())
    mean_lat = 0.5 * (lat_min + lat_max)
    cos_lat = max(abs(np.cos(np.radians(mean_lat))), _MIN_COS_LAT)

    lat_span_km = float(degrees2kilometers(lat_max - lat_min))
    lon_span_km = float(degrees2kilometers(lon_max - lon_min)) * cos_lat
    margin_km = 0.5 * max(lat_span_km, lon_span_km)
    if margin_km < distance_step_km:
        margin_km = float(distance_step_km)
    margin_km = min(margin_km, float(max_distance_km))

    pad_lat = float(kilometer2degrees(margin_km))
    pad_lon = float(kilometer2degrees(margin_km)) / cos_lat
    return (
        lat_min - pad_lat,
        lat_max + pad_lat,
        lon_min - pad_lon,
        lon_max + pad_lon,
    )


def _horizontal_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    distance_step_km: float,
) -> list[tuple[float, float]]:
    """Regular lat/lon grid over ``[lat_min, lat_max] × [lon_min, lon_max]``."""
    if lat_max < lat_min or lon_max < lon_min:
        return []

    mean_lat = 0.5 * (lat_min + lat_max)
    cos_lat = max(abs(np.cos(np.radians(mean_lat))), _MIN_COS_LAT)
    dlat = float(kilometer2degrees(distance_step_km))
    dlon = float(kilometer2degrees(distance_step_km)) / cos_lat

    lats = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
    lons = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
    if lats.size == 0:
        lats = np.asarray([mean_lat], dtype=float)
    if lons.size == 0:
        lons = np.asarray([0.5 * (lon_min + lon_max)], dtype=float)

    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def _slice_by_station_coverage(
    nodes: Sequence[tuple[float, float]],
    station_latlons: Sequence[tuple[float, float]],
    max_distance_km: float,
    n_phases: int,
    min_picks_per_event: int,
) -> list[tuple[float, float]]:
    """Keep nodes with ``n_stations_in_range * n_phases > min_picks_per_event``."""
    kept: list[tuple[float, float]] = []
    for lat, lon in nodes:
        n_in_range = 0
        for sta_lat, sta_lon in station_latlons:
            dist_m, _, _ = gps2dist_azimuth_nb(
                float(lat), float(lon), float(sta_lat), float(sta_lon)
            )
            if dist_m / 1000.0 <= max_distance_km:
                n_in_range += 1
        if n_in_range * n_phases > min_picks_per_event:
            kept.append((float(lat), float(lon)))
    return kept


def _depth_grid(
    depth_step_km: Optional[float],
    max_depth_km: Optional[float],
    fix_depth_km: Optional[float],
) -> list[float]:
    """Build a regular depth grid (or a single fixed depth)."""
    if fix_depth_km is not None:
        return [float(fix_depth_km)]

    if depth_step_km is None or max_depth_km is None:
        raise ValueError(
            "Grid-search seeding requires depth_step_km and max_depth_km "
            "on the initial propagation engine when depth is not fixed."
        )
    if depth_step_km <= 0.0 or max_depth_km <= 0.0:
        raise ValueError("depth_step_km and max_depth_km must be positive.")

    depths = np.arange(
        float(depth_step_km),
        float(max_depth_km) + 0.5 * float(depth_step_km),
        float(depth_step_km),
    )
    if depths.size == 0:
        return [float(depth_step_km)]
    return [float(d) for d in depths]


def _require_grid_search_engine_attrs(engine: PropagationEngine) -> tuple[float, float]:
    """Validate engine attrs needed for horizontal grid search.

    Returns
    -------
    tuple[float, float]
        ``(distance_step_km, max_distance_km)``. ``max_distance_km`` is the
        effective coverage limit (configured max capped by the last tabulated
        distance sample when a table grid is present).
    """
    distance_step_km = engine.distance_step_km
    max_distance_km = engine.max_distance_km
    if distance_step_km is None or distance_step_km <= 0.0:
        raise ValueError(
            "Grid-search seeding requires a positive distance_step_km "
            "on the initial propagation engine."
        )
    if max_distance_km is None or max_distance_km <= 0.0:
        raise ValueError(
            "Grid-search seeding requires a positive max_distance_km "
            "on the initial propagation engine."
        )
    distance_grid = getattr(engine, "distance_grid", None)
    if distance_grid is not None and len(distance_grid) > 0:
        max_distance_km = min(float(max_distance_km), float(distance_grid[-1]))
    return float(distance_step_km), float(max_distance_km)


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


def _earth_radius_m(engine: PropagationEngine) -> float:
    """Return planet radius in meters from the engine model when available."""
    model = getattr(engine, "velocity_model", None)
    if model is not None and hasattr(model, "radius_of_planet_km"):
        return float(model.radius_of_planet_km) * 1000.0
    return 6371.0e3


def _effective_max_distance_km(engine: PropagationEngine) -> float | None:
    """Return configured max distance capped by the last tabulated sample."""
    max_distance_km = engine.max_distance_km
    if max_distance_km is None:
        return None
    distance_grid = getattr(engine, "distance_grid", None)
    if distance_grid is not None and len(distance_grid) > 0:
        return min(float(max_distance_km), float(distance_grid[-1]))
    return float(max_distance_km)


def _table_travel_time(
    engine: PropagationEngine,
    distance_km: float,
    depth_km: float,
    phase: str,
    max_distance_km: float | None,
) -> float | None:
    """Interpolate a 1D table travel time, or ``None`` if out of range."""
    if max_distance_km is not None and distance_km > max_distance_km:
        return None
    depth_grid = getattr(engine, "depth_grid", None)
    if depth_grid is not None and depth_km > float(depth_grid[-1]):
        return None
    try:
        travel_time = engine.compute_travel_time(distance_km, depth_km, phase)
    except (ValueError, AttributeError):
        return None
    if travel_time is None or not np.isfinite(travel_time):
        return None
    return float(travel_time)


def _wrms_from_residuals(
    residual_arrival_times: list[float],
    weights: list[float],
    min_needed: int,
) -> tuple[float, float, int] | None:
    """Estimate ``t0`` and weighted RMS from arrival residuals."""
    if len(residual_arrival_times) < min_needed:
        return None

    residual_arr = np.asarray(residual_arrival_times, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    t0_rel = round(float(np.sum(weight_arr * residual_arr) / np.sum(weight_arr)), 4)
    wrms = float(
        np.sqrt(np.sum(weight_arr * (residual_arr - t0_rel) ** 2) / np.sum(weight_arr))
    )
    return t0_rel, wrms, len(residual_arrival_times)


def _score_depth_with_table(
    engine: PropagationEngine,
    pick_geometry: Sequence[tuple[float, float, str, float, float]],
    depth_km: float,
    max_distance_km: float | None,
    min_needed: int,
) -> tuple[float, float, int] | None:
    """Score one depth using precomputed epicentral distances (1D tables)."""
    residual_arrival_times: list[float] = []
    weights: list[float] = []
    for distance_km, sta_elev_m, phase, t_obs_rel, sigma in pick_geometry:
        travel_time = _table_travel_time(
            engine,
            distance_km,
            depth_km + sta_elev_m / 1000.0,
            phase,
            max_distance_km,
        )
        if travel_time is None:
            continue
        residual_arrival_times.append(t_obs_rel - travel_time)
        weights.append(1.0 / (sigma * sigma))
    return _wrms_from_residuals(residual_arrival_times, weights, min_needed)


def _score_depth_with_get_arrival(
    engine: PropagationEngine,
    lat: float,
    lon: float,
    depth_km: float,
    pick_geometry: Sequence[tuple[float, float, float, str, float, float]],
    max_distance_km: float | None,
    min_needed: int,
) -> tuple[float, float, int] | None:
    """Score one depth via ``get_arrival`` (no derivative evaluation)."""
    residual_arrival_times: list[float] = []
    weights: list[float] = []
    for sta_lat, sta_lon, sta_elev_m, phase, t_obs_rel, sigma in pick_geometry:
        _, travel_time, _, _ = engine.get_arrival(
            source_latitude=lat,
            source_longitude=lon,
            source_depth_km=depth_km,
            receiver_latitude=sta_lat,
            receiver_longitude=sta_lon,
            receiver_elevation_m=sta_elev_m,
            phase=phase,
            max_distance_km=max_distance_km,
        )
        if travel_time is None or not np.isfinite(travel_time):
            continue
        residual_arrival_times.append(t_obs_rel - float(travel_time))
        weights.append(1.0 / (sigma * sigma))
    return _wrms_from_residuals(residual_arrival_times, weights, min_needed)


class LocationSeeder:
    """Initialize event hypocenters and origin-time seeds before inversion.

    Parameters
    ----------
    config : LocatorConfig
        Locator configuration controlling depth handling and pick filtering.
    data : DataHandler
        Shared event data container that receives initial locations.
    run_config : RunConfig
        Per-run configuration (kept for API compatibility with the locator).
    propagation_engine : PropagationEngine or None
        Optional forward model used to score a lat/lon/depth grid via travel
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
            Initial depth guess in kilometers (heuristic path only).
        depth_step_km : float or None
            Depth spacing for the grid-search depth axis.
        max_depth_km : float or None
            Upper bound for the grid-search depth axis.

        Notes
        -----
        When ``propagation_engine`` is set, builds a regular lat/lon/depth grid
        (config search bounds if provided, else a padded station envelope),
        slices nodes that lack enough in-range stations, and keeps the
        lowest weighted-RMS hypocenter. When ``propagation_engine`` is
        ``None``, uses the first epicenter seed (clamped to config search
        bounds when set) and trial/fixed depth with origin time at the
        earliest pick.
        """
        data = self.data
        config = self.config

        for event_index, event in enumerate(data.events):
            if len(event.picks) == 0:
                continue

            known_picks = _known_station_picks(event.picks, data.stations)
            station_indices = _station_indices_for_picks(known_picks, data)

            if not station_indices:
                logger.warning(
                    "Event %d has no picks from known stations; skipping seed.",
                    event_index,
                )
                continue

            if self.propagation_engine is None:
                seeds = _epicenter_seeds(known_picks, data, station_indices)
                if not seeds:
                    logger.warning(
                        "Event %d has no picks from known stations; skipping seed.",
                        event_index,
                    )
                    continue
                depth_km = (
                    trial_depth_km
                    if config.fix_depth_km is None
                    else float(config.fix_depth_km)
                )
                self._apply_heuristic_seed(event_index, known_picks, seeds[0], depth_km)
                continue

            epicenter_nodes, candidate_depths = self._build_grid_search_candidates(
                station_indices,
                depth_step_km,
                max_depth_km,
            )
            logger.debug(
                "Grid-search nodes: %d, Depth candidates: %s for event %d",
                len(epicenter_nodes),
                candidate_depths,
                event_index,
            )

            if not epicenter_nodes:
                logger.warning(
                    "Event %d: empty coverage-sliced search grid; skipping seed.",
                    event_index,
                )
                continue

            best_solution = self._score_candidates(
                event_index, epicenter_nodes, candidate_depths
            )
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

    def _build_grid_search_candidates(
        self,
        station_indices: list[int],
        depth_step_km: Optional[float],
        max_depth_km: Optional[float],
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Build coverage-sliced epicenter nodes and depth candidates."""
        assert self.propagation_engine is not None
        distance_step_km, max_distance_km = _require_grid_search_engine_attrs(
            self.propagation_engine
        )

        station_latlons = [
            (
                float(self.data.stations_locations[i][0]),
                float(self.data.stations_locations[i][1]),
            )
            for i in station_indices
        ]

        cfg_bounds = _config_search_bounds(self.config)
        if cfg_bounds is not None:
            lat_min, lat_max, lon_min, lon_max = cfg_bounds
        else:
            lat_min, lat_max, lon_min, lon_max = _station_bbox_bounds(
                station_latlons,
                distance_step_km,
                max_distance_km,
            )

        nodes = _horizontal_grid(lat_min, lat_max, lon_min, lon_max, distance_step_km)
        nodes = _slice_by_station_coverage(
            nodes,
            station_latlons,
            max_distance_km,
            self.config.n_phases,
            self.config.min_picks_per_event,
        )
        depths = _depth_grid(
            depth_step_km,
            max_depth_km,
            self.config.fix_depth_km,
        )
        return nodes, depths

    def _apply_heuristic_seed(
        self,
        event_index: int,
        known_picks: Sequence,
        seed: tuple[float, float],
        depth_km: float,
    ) -> None:
        """Assign first epicenter seed + depth without travel-time scoring."""
        lat, lon = self.config.clamp_horizontal(*seed)
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
        """Evaluate epicenter/depth pairs; return best ``(wrms, lat, lon, depth, t0)``.

        Uses travel times only (no derivatives). For 1D table engines exposing
        ``compute_travel_time``, station distances are computed once per
        epicenter and reused across depths.
        """
        assert self.propagation_engine is not None
        engine = self.propagation_engine
        data = self.data

        picks = _usable_arrivals(data, event_index)
        min_needed = max(3, data.min_picks_per_event // 2)
        if len(picks) < min_needed:
            return None

        max_distance_km = _effective_max_distance_km(engine)
        use_table = callable(getattr(engine, "compute_travel_time", None))
        earth_radius_m = _earth_radius_m(engine) if use_table else None

        best_solution: tuple[float, float, float, float, float] | None = None

        def _consider(
            lat: float,
            lon: float,
            depth_km: float,
            scored: tuple[float, float, int] | None,
        ) -> None:
            nonlocal best_solution
            if scored is None:
                return
            t0_estimate_rel, wrms, _ = scored
            if np.isfinite(wrms) and (
                best_solution is None or wrms < best_solution[0]
            ):
                best_solution = (wrms, lat, lon, depth_km, t0_estimate_rel)

        for lat_seed, lon_seed in seeds:
            if use_table:
                assert earth_radius_m is not None
                pick_geometry_table: list[tuple[float, float, str, float, float]] = []
                for station_id, phase_id, t_obs_rel, sigma in picks:
                    sta_lat, sta_lon, sta_elev = data.stations_locations[station_id]
                    dist_m, _, _ = gps2dist_azimuth_nb(
                        lat_seed,
                        lon_seed,
                        float(sta_lat),
                        float(sta_lon),
                        R=earth_radius_m,
                    )
                    pick_geometry_table.append(
                        (
                            dist_m / 1000.0,
                            float(sta_elev),
                            data.phases[phase_id],
                            t_obs_rel,
                            sigma,
                        )
                    )
                for depth_candidate_km in candidate_depths:
                    _consider(
                        lat_seed,
                        lon_seed,
                        depth_candidate_km,
                        _score_depth_with_table(
                            engine,
                            pick_geometry_table,
                            depth_candidate_km,
                            max_distance_km,
                            min_needed,
                        ),
                    )
            else:
                pick_geometry_geo: list[
                    tuple[float, float, float, str, float, float]
                ] = [
                    (
                        float(data.stations_locations[station_id][0]),
                        float(data.stations_locations[station_id][1]),
                        float(data.stations_locations[station_id][2]),
                        data.phases[phase_id],
                        t_obs_rel,
                        sigma,
                    )
                    for station_id, phase_id, t_obs_rel, sigma in picks
                ]
                for depth_candidate_km in candidate_depths:
                    _consider(
                        lat_seed,
                        lon_seed,
                        depth_candidate_km,
                        _score_depth_with_get_arrival(
                            engine,
                            lat_seed,
                            lon_seed,
                            depth_candidate_km,
                            pick_geometry_geo,
                            max_distance_km,
                            min_needed,
                        ),
                    )

        if best_solution is not None:
            logger.debug(
                "Best seed for event %d: WRMS=%.3f, lat=%.4f, lon=%.4f, "
                "depth=%.2f km, t0_rel=%.3fs",
                event_index,
                best_solution[0],
                best_solution[1],
                best_solution[2],
                best_solution[3],
                best_solution[4],
            )
        return best_solution
