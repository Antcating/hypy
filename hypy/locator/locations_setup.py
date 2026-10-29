"""Seed and initialize event locations before inversion."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from obspy.geodetics import (  # type: ignore[import-untyped]
    degrees2kilometers,
    kilometer2degrees,
)

from hypy.propagation import PropagationEngine
from hypy.utils.geo import gps2dist_azimuth_batch_nb, gps2dist_azimuth_nb
from hypy.utils.seismic import normalize_phase

from ..logger import logger  # type: ignore[import-not-found]
from .config import LocatorConfig, RunConfig
from .data import DataHandler

_MIN_COS_LAT = 1e-6
# λ_min/λ_max below this: extra trials stay in the station strip.
_LINEAR_ASPECT = 0.1


def _early_p_weights(
    picks: Sequence,
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Return ``(latlon, weights, dt_span, time_scale)`` for the earliest P.

    ``None`` when fewer than two stations have a P or the weights vanish.
    """
    earliest_p: dict[tuple[str, str], float] = {}
    for pick in picks:
        if normalize_phase(pick.phase_hint or "") != "p":
            continue
        key = (pick.waveform_id.network_code, pick.waveform_id.station_code)
        if key not in event_stations:
            continue
        pick_time = float(pick.time)
        previous = earliest_p.get(key)
        if previous is None or pick_time < previous:
            earliest_p[key] = pick_time
    if len(earliest_p) < 2:
        return None
    p_times = np.asarray(list(earliest_p.values()), dtype=float)
    p_latlon = np.asarray(
        [event_stations[key][1:3] for key in earliest_p],
        dtype=float,
    )
    dt = p_times - np.min(p_times)
    positive_dt = dt[dt > 0.0]
    time_scale_s = max(float(np.median(positive_dt)), 0.5) if positive_dt.size else 1.0
    weights = np.exp(-dt / time_scale_s)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None
    return p_latlon, weights / weight_sum, float(np.max(dt)), time_scale_s


def _line_frame(
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
) -> tuple[float, float, float, float, float] | None:
    """Return line axes, or None when ``λ_min/λ_max`` is at least 0.1.

    ``(along_e, along_n, cross_e, cross_n, sigma_cross_km)``.
    ``cross`` is 90° counter-clockwise from ``along`` (``along_e >= 0``).
    ``sigma_cross_km`` is ``sqrt(λ_min)``, the station-strip half-width.
    """
    if len(event_stations) < 2:
        return None
    latlon = np.asarray(
        [(lat, lon) for _, lat, lon, _ in event_stations.values()],
        dtype=float,
    )
    center_lat = float(np.mean(latlon[:, 0]))
    center_lon = float(np.mean(latlon[:, 1]))
    cos_lat = max(abs(np.cos(np.radians(center_lat))), _MIN_COS_LAT)
    north = np.asarray(degrees2kilometers(latlon[:, 0] - center_lat), dtype=float)
    east = np.asarray(
        degrees2kilometers(latlon[:, 1] - center_lon) * cos_lat, dtype=float
    )
    count = float(east.size)
    cov_ee = float(np.dot(east, east) / count)
    cov_en = float(np.dot(east, north) / count)
    cov_nn = float(np.dot(north, north) / count)
    trace = cov_ee + cov_nn
    disc = float(
        np.sqrt(max(trace * trace - 4.0 * (cov_ee * cov_nn - cov_en * cov_en), 0.0))
    )
    lam_max = 0.5 * (trace + disc)
    lam_min = max(0.0, 0.5 * (trace - disc))
    if lam_max <= 1e-12 or lam_min / lam_max >= _LINEAR_ASPECT:
        return None
    scale = max(1.0, abs(cov_ee), abs(cov_nn))
    if abs(cov_en) <= 1e-12 * scale:
        if cov_ee >= cov_nn:
            along_e, along_n = 1.0, 0.0
        else:
            along_e, along_n = 0.0, 1.0
    else:
        along_e = cov_en
        along_n = lam_max - cov_ee
        norm = float(np.hypot(along_e, along_n))
        if norm <= 1e-15:
            return None
        along_e /= norm
        along_n /= norm
    if along_e < 0.0 or (abs(along_e) <= 1e-15 and along_n < 0.0):
        along_e, along_n = -along_e, -along_n
    cross_e, cross_n = -along_n, along_e
    return along_e, along_n, cross_e, cross_n, float(np.sqrt(lam_min))


def _weighted_epicenter(
    picks: Sequence,
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
) -> tuple[float, float, float, float, float]:
    """Return network mean, extrapolated early-P epicenter, and cloud radius.

    The station centroid is the network center. Stations with earlier P
    arrivals are weighted more strongly. The vector from the centroid to
    that early-arrival point is then extrapolated beyond the station
    cloud (factor 1.25–2.25) so a one-sided or offshore event can seed
    outside the convex hull. Without that step the inversion starts
    inside the network and cannot walk tens of kilometres back out.
    """
    if not event_stations:
        raise ValueError("Cannot estimate heuristic epicenter without stations.")

    station_latlon = np.asarray(
        [(lat, lon) for _, lat, lon, _ in event_stations.values()],
        dtype=float,
    )
    center_lat = float(np.mean(station_latlon[:, 0]))
    center_lon = float(np.mean(station_latlon[:, 1]))
    cos_lat = max(abs(np.cos(np.radians(center_lat))), _MIN_COS_LAT)
    radius_km = max(
        float(
            np.median(
                np.hypot(
                    degrees2kilometers(station_latlon[:, 0] - center_lat),
                    degrees2kilometers(station_latlon[:, 1] - center_lon) * cos_lat,
                )
            )
        ),
        1.0,
    )

    early = _early_p_weights(picks, event_stations)
    if early is None:
        return center_lat, center_lon, center_lat, center_lon, radius_km
    p_latlon, weights, dt_span, time_scale_s = early
    early_lat = float(np.sum(weights * p_latlon[:, 0]))
    early_lon = float(np.sum(weights * p_latlon[:, 1]))

    if dt_span <= 0.0:
        extrapolation_factor = 1.25
    else:
        extrapolation_factor = float(
            np.clip(1.25 + 0.25 * (dt_span / max(time_scale_s, 1e-6)), 1.25, 2.25)
        )
    return (
        center_lat,
        center_lon,
        center_lat + extrapolation_factor * (early_lat - center_lat),
        center_lon + extrapolation_factor * (early_lon - center_lon),
        radius_km,
    )


def _coarse_origin_time(
    event_index: int,
    picks: Sequence,
    data: DataHandler,
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
    lat: float,
    lon: float,
    depth_km: float,
    fixed_t0: Optional[float] = None,
) -> tuple[float, float]:
    """Estimate relative origin time and WRMS from a straight-line hypocenter."""
    phase_index = {phase: idx for idx, phase in enumerate(data.config.phases)}
    t0_estimates: list[float] = []
    weights: list[float] = []
    n_phases = data.config.n_phases

    for pick in picks:
        phase = normalize_phase(pick.phase_hint or "")
        if phase not in phase_index:
            continue
        rec = event_stations.get(
            (pick.waveform_id.network_code, pick.waveform_id.station_code)
        )
        if rec is None:
            continue
        station_id, sta_lat, sta_lon, sta_elev_m = rec
        phase_id = phase_index[phase]
        obs_id = phase_id + station_id * n_phases
        dist_m, _, _ = gps2dist_azimuth_nb(
            float(lat), float(lon), float(sta_lat), float(sta_lon)
        )
        correction = float(data.station_time_corrections[obs_id])
        t0_estimate = (
            float(pick.time)
            - float(data.time_refs[event_index])
            - float(
                np.hypot(
                    float(dist_m) / 1000.0,
                    max(0.0, float(depth_km) + float(sta_elev_m) / 1000.0),
                )
            )
            / (6.0 if phase == "p" else 6.0 / 1.75)
            - (correction if np.isfinite(correction) else 0.0)
        )
        sigma = data.arrival_time_errors[event_index, obs_id]
        if np.isfinite(t0_estimate):
            t0_estimates.append(float(t0_estimate))
            weights.append(
                1.0
                if sigma is None or not np.isfinite(sigma) or sigma <= 0.0
                else 1.0 / float(sigma) ** 2
            )

    if not t0_estimates:
        t0_rel = float(min(picks, key=lambda pk: pk.time).time) - float(
            data.time_refs[event_index]
        )
        if fixed_t0 is not None and np.isfinite(fixed_t0):
            t0_rel = float(fixed_t0)
        return t0_rel, float("inf")

    t0_arr = np.asarray(t0_estimates, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    abs_dev = np.abs(t0_arr - float(np.median(t0_arr)))
    mad = float(np.median(abs_dev))
    if mad > 0.0:
        keep = abs_dev <= 3.0 * (1.4826 * mad)
        if np.any(keep):
            t0_arr = t0_arr[keep]
            weight_arr = weight_arr[keep]
    weight_sum = float(np.sum(weight_arr))
    if fixed_t0 is not None and np.isfinite(fixed_t0):
        t0_rel = float(fixed_t0)
    elif not np.isfinite(weight_sum) or weight_sum <= 0.0:
        t0_rel = float(np.median(t0_arr))
    else:
        t0_rel = float(np.sum(weight_arr * t0_arr) / weight_sum)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return t0_rel, float("inf")
    residuals = t0_arr - t0_rel
    return t0_rel, float(np.sqrt(np.sum(weight_arr * residuals**2) / weight_sum))


def get_seed_from_origins(
    event, time_ref: float, config: LocatorConfig
) -> tuple[float, float, Optional[float], Optional[float]] | None:
    """Return ``(lat, lon, depth_km, t0_rel)`` from ``event.origins[-1]``."""
    if not event.origins:
        return None
    origin = event.origins[-1]
    if origin.latitude is None or origin.longitude is None:
        return None
    lat, lon = float(origin.latitude), float(origin.longitude)
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    lat, lon = config.clamp_horizontal(lat, lon)
    depth_km = (
        float(origin.depth) / 1000.0
        if origin.depth is not None and np.isfinite(origin.depth)
        else None
    )
    t0_rel = float(origin.time) - float(time_ref) if origin.time is not None else None
    return lat, lon, depth_km, t0_rel


def _slice_by_station_coverage(
    nodes: Sequence[tuple[float, float]],
    station_latlons: Sequence[tuple[float, float]],
    max_distance_km: float,
    n_phases: int,
    min_picks_per_event: int,
) -> list[tuple[float, float]]:
    """Keep nodes with enough potentially usable stations."""
    kept: list[tuple[float, float]] = []
    for lat, lon in nodes:
        n_in_range = 0
        for sta_lat, sta_lon in station_latlons:
            dist_m, _, _ = gps2dist_azimuth_nb(
                float(lat), float(lon), float(sta_lat), float(sta_lon)
            )
            if dist_m / 1000.0 <= max_distance_km:
                n_in_range += 1
        if n_in_range * n_phases >= min_picks_per_event:
            kept.append((float(lat), float(lon)))
    return kept


def _usable_arrivals(
    data: DataHandler, event_id: int
) -> list[tuple[int, int, float, float]]:
    """Collect ``(station_id, phase_id, corrected_t_obs_rel, sigma)``."""
    usable: list[tuple[int, int, float, float]] = []
    n_phases = data.config.n_phases
    for station_id in range(data.n_stations):
        for phase_id in range(n_phases):
            obs_id = phase_id + station_id * n_phases
            arrival_time = data.arrivals[event_id, obs_id]
            if np.isnan(arrival_time):
                continue
            sigma = data.arrival_time_errors[event_id, obs_id]
            if sigma is None or np.isnan(sigma) or sigma <= 0.0:
                continue
            correction = float(data.station_time_corrections[obs_id])
            usable.append(
                (
                    station_id,
                    phase_id,
                    float(arrival_time)
                    - (correction if np.isfinite(correction) else 0.0),
                    float(sigma),
                )
            )
    return usable


def _wrms_from_residuals(
    residual_arrival_times: list[float],
    weights: list[float],
    min_needed: int,
    fixed_t0: Optional[float] = None,
) -> tuple[float, float] | None:
    """Estimate t0 and weighted RMS from candidate arrival residuals."""
    if len(residual_arrival_times) < min_needed:
        return None
    residual_arr = np.asarray(residual_arrival_times, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weight_arr))
    if weight_sum <= 0.0:
        return None
    if fixed_t0 is not None and np.isfinite(fixed_t0):
        t0_rel = float(fixed_t0)
    else:
        t0_rel = round(float(np.sum(weight_arr * residual_arr) / weight_sum), 4)
    wrms = float(
        np.sqrt(np.sum(weight_arr * (residual_arr - t0_rel) ** 2) / weight_sum)
    )
    return t0_rel, wrms


def _score_hypocenter_with_engine(
    data: DataHandler,
    engine: PropagationEngine,
    event_index: int,
    lat: float,
    lon: float,
    depth_km: float,
    fixed_t0: Optional[float] = None,
) -> tuple[float, float]:
    """Score one hypocenter with the inversion forward model.

    Returns ``(t0_rel, wrms)``. Unusable geometry (too few in-range
    arrivals) is ``(nan, inf)`` so the caller can skip the candidate.
    """
    picks = _usable_arrivals(data, event_index)
    min_needed = max(3, data.config.min_picks_per_event // 2)
    if len(picks) < min_needed:
        return float("nan"), float("inf")

    max_distance_km = engine.max_distance_km
    if max_distance_km is not None:
        max_distance_km = float(max_distance_km)
    phases = data.config.phases
    residuals: list[float] = []
    weights: list[float] = []

    by_phase: dict[int, list[tuple[int, float, float]]] = {}
    for station_id, phase_id, t_obs_rel, sigma in picks:
        by_phase.setdefault(phase_id, []).append((station_id, t_obs_rel, sigma))

    get_batch = getattr(engine, "get_arrival_batch", None)
    for phase_id, rows in by_phase.items():
        station_ids = np.asarray([row[0] for row in rows], dtype=int)
        t_obs = np.asarray([row[1] for row in rows], dtype=float)
        sigmas = np.asarray([row[2] for row in rows], dtype=float)
        locs = data.stations_locations[station_ids]
        travel_times: np.ndarray
        if callable(get_batch):
            try:
                _, travel_times, _, _ = get_batch(
                    source_latitude=float(lat),
                    source_longitude=float(lon),
                    source_depth_km=float(depth_km),
                    receiver_latitude=np.asarray(locs[:, 0], dtype=float),
                    receiver_longitude=np.asarray(locs[:, 1], dtype=float),
                    receiver_elevation_m=np.asarray(locs[:, 2], dtype=float),
                    phase=phases[phase_id],
                    max_distance_km=max_distance_km,
                )
            except (TypeError, ValueError, NotImplementedError):
                travel_times = np.full(len(rows), np.nan, dtype=float)
                get_batch = None
        if not callable(get_batch):
            travel_times = np.full(len(rows), np.nan, dtype=float)
            for i, station_id in enumerate(station_ids):
                sta_lat, sta_lon, sta_elev = data.stations_locations[station_id]
                try:
                    _, travel_time, _, _ = engine.get_arrival(
                        source_latitude=float(lat),
                        source_longitude=float(lon),
                        source_depth_km=float(depth_km),
                        receiver_latitude=float(sta_lat),
                        receiver_longitude=float(sta_lon),
                        receiver_elevation_m=float(sta_elev),
                        phase=phases[phase_id],
                        max_distance_km=max_distance_km,
                    )
                except (TypeError, ValueError, NotImplementedError):
                    continue
                if travel_time is not None:
                    travel_times[i] = float(travel_time)
        keep = np.isfinite(travel_times)
        residuals.extend((t_obs[keep] - travel_times[keep]).tolist())
        weights.extend((1.0 / (sigmas[keep] * sigmas[keep])).tolist())

    scored = _wrms_from_residuals(residuals, weights, min_needed, fixed_t0=fixed_t0)
    if scored is None:
        return float("nan"), float("inf")
    return scored


def _score_candidates(
    data: DataHandler,
    engine: PropagationEngine,
    event_index: int,
    seeds: list[tuple[float, float]],
    candidate_depths: list[float],
) -> tuple[float, float, float, float, float] | None:
    """Evaluate grid-search epicenter/depth pairs; return best WRMS solution."""
    picks = _usable_arrivals(data, event_index)
    min_needed = max(3, data.config.min_picks_per_event // 2)
    if len(picks) < min_needed:
        return None

    best_solution: tuple[float, float, float, float, float] | None = None

    if not callable(getattr(engine, "compute_travel_time_batch", None)):
        for lat_seed, lon_seed in seeds:
            for depth_km in candidate_depths:
                t0_estimate_rel, wrms = _score_hypocenter_with_engine(
                    data, engine, event_index, lat_seed, lon_seed, depth_km
                )
                if np.isfinite(wrms) and (
                    best_solution is None or wrms < best_solution[0]
                ):
                    best_solution = (
                        wrms,
                        lat_seed,
                        lon_seed,
                        depth_km,
                        t0_estimate_rel,
                    )
    else:
        max_distance_km = float(engine.distance_grid[-1])
        if engine.max_distance_km is not None:
            max_distance_km = min(max_distance_km, float(engine.max_distance_km))
        earth_radius_m = float(engine.velocity_model.radius_of_planet_km) * 1000.0
        phases = data.config.phases

        station_ids = np.asarray([p[0] for p in picks], dtype=int)
        phase_ids = np.asarray([p[1] for p in picks], dtype=int)
        t_obs = np.asarray([p[2] for p in picks], dtype=float)
        pick_weights = 1.0 / np.asarray([p[3] for p in picks], dtype=float) ** 2
        locs = data.stations_locations[station_ids].astype(float)
        sta_lat = np.ascontiguousarray(locs[:, 0])
        sta_lon = np.ascontiguousarray(locs[:, 1])
        depths = np.asarray(candidate_depths, dtype=float)
        # (n_depths, n_picks): receiver elevation folded into the source depth.
        source_depths = depths[:, None] + locs[None, :, 2] / 1000.0
        depth_ok = (source_depths >= 0.0) & (
            source_depths <= float(engine.depth_grid[-1])
        )
        phase_cols = [
            (phases[phase_id], np.flatnonzero(phase_ids == phase_id))
            for phase_id in np.unique(phase_ids)
        ]

        # ponytail: fixed chunk bounds memory at ~chunk*n_depths*n_picks floats.
        chunk = 256
        for start in range(0, len(seeds), chunk):
            block = seeds[start : start + chunk]
            dist_km = (
                np.stack(
                    [
                        gps2dist_azimuth_batch_nb(
                            lat, lon, sta_lat, sta_lon, earth_radius_m
                        )[0]
                        for lat, lon in block
                    ]
                )
                / 1000.0
            )
            # (n_block, n_depths, n_picks)
            valid = depth_ok[None] & (dist_km <= max_distance_km)[:, None, :]
            travel_times = np.full(valid.shape, np.nan)
            for phase, cols in phase_cols:
                s, d, sub = np.nonzero(valid[:, :, cols])
                if s.size:
                    c = cols[sub]
                    travel_times[s, d, c] = engine.compute_travel_time_batch(
                        dist_km[s, c], source_depths[d, c], phase
                    )

            ok = np.isfinite(travel_times)
            w = np.where(ok, pick_weights, 0.0)
            r = np.where(ok, t_obs - travel_times, 0.0)
            w_sum = w.sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                t0_rel = np.round((w * r).sum(axis=-1) / w_sum, 4)
                wrms = np.sqrt((w * (r - t0_rel[..., None]) ** 2).sum(axis=-1) / w_sum)
            wrms[(ok.sum(axis=-1) < min_needed) | ~(w_sum > 0.0)] = np.inf
            i, k = np.unravel_index(int(np.argmin(wrms)), wrms.shape)
            if np.isfinite(wrms[i, k]) and (
                best_solution is None or wrms[i, k] < best_solution[0]
            ):
                best_solution = (
                    float(wrms[i, k]),
                    block[i][0],
                    block[i][1],
                    float(depths[k]),
                    float(t0_rel[i, k]),
                )

    if best_solution is not None:
        logger.debug(
            "Best seed for event %d: WRMS=%.3f, lat=%.4f, lon=%.4f, "
            "depth=%.2f km, t0_rel=%.3fs",
            event_index,
            *best_solution,
        )
    return best_solution


def _seed_heuristic(
    config: LocatorConfig,
    data: DataHandler,
    event,
    event_index: int,
    known_picks: Sequence,
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
    trial_depth_km: float,
    max_depth_km: Optional[float],
    run_config: RunConfig,
    scoring_engine: Optional[PropagationEngine] = None,
) -> None:
    """Seed one event from scored mean, extrapolated, origin, and Gaussian trials.

    The extrapolated early-P anchor can lie outside the station hull so
    offshore and edge-of-network events are not started inside the network.
    Candidates are scored with ``scoring_engine`` (the inversion forward
    model) when provided; otherwise a straight-line 6 km/s fallback.
    When ``λ_min/λ_max < 0.1``, the extra trials run along the line with
    cross-track scatter ``sqrt(λ_min)``. One anchor sits under the earliest
    P station. Both count toward ``num_trials``.
    """
    mean_lat, mean_lon, weighted_lat, weighted_lon, radius_km = _weighted_epicenter(
        known_picks, event_stations
    )
    mean_lat, mean_lon = config.clamp_horizontal(mean_lat, mean_lon)
    weighted_lat, weighted_lon = config.clamp_horizontal(weighted_lat, weighted_lon)
    depth_fixed = config.fix_depth_km is not None
    default_depth = float(config.fix_depth_km) if depth_fixed else float(trial_depth_km)

    # Origin first so ``num_trials=1`` with ``use_origins`` keeps only it.
    anchors: list[tuple[float, float, float, Optional[float]]] = []
    if run_config.use_origins:
        origin_seed = get_seed_from_origins(event, data.time_refs[event_index], config)
        if origin_seed is not None:
            o_lat, o_lon, o_depth, o_t0 = origin_seed
            o_z = default_depth
            if not depth_fixed and o_depth is not None and np.isfinite(o_depth):
                o_z = float(o_depth)
            o_t0_rel = float(o_t0) if o_t0 is not None and np.isfinite(o_t0) else None
            anchors.append((o_lat, o_lon, o_z, o_t0_rel))

    def _seen(lat: float, lon: float, depth: float) -> bool:
        return any(
            abs(lat - a_lat) <= 1e-8
            and abs(lon - a_lon) <= 1e-8
            and abs(depth - a_depth) <= 1e-6
            for a_lat, a_lon, a_depth, _ in anchors
        )

    if not _seen(mean_lat, mean_lon, default_depth):
        anchors.append((mean_lat, mean_lon, default_depth, None))
    if (
        abs(weighted_lat - mean_lat) > 1e-8 or abs(weighted_lon - mean_lon) > 1e-8
    ) and not _seen(weighted_lat, weighted_lon, default_depth):
        anchors.append((weighted_lat, weighted_lon, default_depth, None))
    first_p = None
    for pick in known_picks:
        if normalize_phase(pick.phase_hint or "") != "p":
            continue
        rec = event_stations.get(
            (pick.waveform_id.network_code, pick.waveform_id.station_code)
        )
        if rec is None:
            continue
        pick_time = float(pick.time)
        if first_p is None or pick_time < first_p[0]:
            first_p = (pick_time, float(rec[1]), float(rec[2]))
    if first_p is not None and not _seen(first_p[1], first_p[2], default_depth):
        anchors.append(
            (*config.clamp_horizontal(first_p[1], first_p[2]), default_depth, None)
        )

    num_trials = run_config.num_trials
    candidates = list(anchors[:num_trials])
    rng = np.random.default_rng(run_config.seed)
    sigma_z = min(5.0, 0.3 * float(trial_depth_km))
    line = _line_frame(event_stations)
    for extra_i in range(max(0, num_trials - len(candidates))):
        a_lat, a_lon, a_z, _ = anchors[extra_i % len(anchors)]
        if line is None:
            d_north, d_east = (float(v) for v in rng.normal(0.0, radius_km, size=2))
        else:
            along_e, along_n, cross_e, cross_n, sigma_cross = line
            along_km, cross_km = rng.normal(0.0, (radius_km, sigma_cross))
            d_north = float(along_km) * along_n + float(cross_km) * cross_n
            d_east = float(along_km) * along_e + float(cross_km) * cross_e
        cos_lat = max(abs(np.cos(np.radians(a_lat))), _MIN_COS_LAT)
        lat, lon = config.clamp_horizontal(
            a_lat + float(kilometer2degrees(float(d_north))),
            a_lon + float(kilometer2degrees(float(d_east))) / cos_lat,
        )
        if depth_fixed:
            depth_km = a_z
        else:
            depth_km = max(0.0, float(rng.normal(a_z, sigma_z)))
            if max_depth_km is not None:
                depth_km = min(depth_km, float(max_depth_km))
        candidates.append((lat, lon, depth_km, None))

    best: Optional[tuple[float, float, float, float, float]] = None
    fallback: Optional[tuple[float, float, float, float]] = None
    scored_all: list[tuple[int, float, float, float, float, float]] = []
    for cand_idx, (lat, lon, depth_km, fixed_t0) in enumerate(candidates):
        if scoring_engine is not None:
            t0_rel, wrms = _score_hypocenter_with_engine(
                data,
                scoring_engine,
                event_index,
                lat,
                lon,
                depth_km,
                fixed_t0=fixed_t0,
            )
        else:
            t0_rel, wrms = _coarse_origin_time(
                event_index,
                known_picks,
                data,
                event_stations,
                lat,
                lon,
                depth_km,
                fixed_t0=fixed_t0,
            )
        if fallback is None:
            fallback = (lat, lon, depth_km, t0_rel)
        if np.isfinite(wrms):
            scored_all.append((cand_idx, wrms, lat, lon, depth_km, t0_rel))
            if best is None or wrms < best[0]:
                best = (wrms, lat, lon, depth_km, t0_rel)

    if best is None and scoring_engine is not None:
        for cand_idx, (lat, lon, depth_km, fixed_t0) in enumerate(candidates):
            t0_rel, wrms = _coarse_origin_time(
                event_index,
                known_picks,
                data,
                event_stations,
                lat,
                lon,
                depth_km,
                fixed_t0=fixed_t0,
            )
            if fallback is None:
                fallback = (lat, lon, depth_km, t0_rel)
            if np.isfinite(wrms):
                scored_all.append((cand_idx, wrms, lat, lon, depth_km, t0_rel))
                if best is None or wrms < best[0]:
                    best = (wrms, lat, lon, depth_km, t0_rel)

    if best is not None:
        _, lat, lon, depth_km, t0_rel = best
        if event_index < len(data.fallback_seeds):
            n_anchors = len(anchors)
            anchor_part = [s for s in reversed(scored_all) if s[0] < n_anchors]
            extra_part = sorted(
                (s for s in scored_all if s[0] >= n_anchors), key=lambda s: s[1]
            )
            seen_starts = [(lat, lon)]
            fallbacks: list[tuple[float, float, float, float]] = []
            for _, _, fb_lat, fb_lon, fb_z, fb_t0 in anchor_part + extra_part:
                if any(
                    abs(fb_lat - s_lat) < 1e-4 and abs(fb_lon - s_lon) < 1e-4
                    for s_lat, s_lon in seen_starts
                ):
                    continue
                seen_starts.append((fb_lat, fb_lon))
                fallbacks.append((fb_lat, fb_lon, fb_z, fb_t0))
                if len(fallbacks) >= 3:
                    break
            data.fallback_seeds[event_index] = fallbacks
    else:
        lat, lon, depth_km, t0_rel = fallback  # type: ignore[misc]
    data.locations[event_index] = (lat, lon, depth_km)
    data.origin_times[event_index] = t0_rel
    logger.debug(
        "[init:heuristic] ev#%d: lat=%.4f lon=%.4f z=%.2f km t0_rel=%.3fs t0_abs=%.3fs",
        event_index,
        lat,
        lon,
        depth_km,
        t0_rel,
        t0_rel + data.time_refs[event_index],
    )


def _seed_grid_search(
    config: LocatorConfig,
    data: DataHandler,
    event,
    event_index: int,
    event_stations: dict[tuple[str, str], tuple[int, float, float, float]],
    depth_step_km: Optional[float],
    max_depth_km: Optional[float],
    engine: PropagationEngine,
    use_origins: bool,
) -> None:
    """Seed one event by scoring a travel-time grid."""
    distance_step_km = engine.distance_step_km
    max_distance_km = engine.max_distance_km
    if distance_step_km is None or distance_step_km <= 0.0:
        raise ValueError(
            "Grid-search seeding requires a positive "
            "distance_step_km on the initial propagation engine."
        )
    if max_distance_km is None or max_distance_km <= 0.0:
        raise ValueError(
            "Grid-search seeding requires a positive "
            "max_distance_km on the initial propagation engine."
        )
    distance_step_km = float(distance_step_km)
    max_distance_km = float(max_distance_km)
    if "propagation.table" in engine.feature_flags:
        max_distance_km = min(max_distance_km, float(engine.distance_grid[-1]))

    station_latlons = [
        (float(lat), float(lon)) for _, lat, lon, _ in event_stations.values()
    ]
    bounds = config.search_bounds
    if bounds is None:
        arr = np.asarray(station_latlons, dtype=float)
        lat_min, lat_max = float(arr[:, 0].min()), float(arr[:, 0].max())
        lon_min, lon_max = float(arr[:, 1].min()), float(arr[:, 1].max())
        cos_lat = max(abs(np.cos(np.radians(0.5 * (lat_min + lat_max)))), _MIN_COS_LAT)
        margin_km = min(
            max(
                0.5
                * max(
                    float(degrees2kilometers(lat_max - lat_min)),
                    float(degrees2kilometers(lon_max - lon_min)) * cos_lat,
                ),
                distance_step_km,
            ),
            max_distance_km,
        )
        pad_lat = float(kilometer2degrees(margin_km))
        bounds = (
            lat_min - pad_lat,
            lat_max + pad_lat,
            lon_min - pad_lat / cos_lat,
            lon_max + pad_lat / cos_lat,
        )
    lat_min, lat_max, lon_min, lon_max = bounds

    if lat_max < lat_min or lon_max < lon_min:
        epicenter_nodes: list[tuple[float, float]] = []
    else:
        mean_lat = 0.5 * (lat_min + lat_max)
        cos_lat = max(abs(np.cos(np.radians(mean_lat))), _MIN_COS_LAT)
        dlat = float(kilometer2degrees(distance_step_km))
        dlon = dlat / cos_lat
        lats = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
        lons = np.arange(lon_min, lon_max + 0.5 * dlon, dlon)
        if lats.size == 0:
            lats = np.asarray([mean_lat], dtype=float)
        if lons.size == 0:
            lons = np.asarray([0.5 * (lon_min + lon_max)], dtype=float)
        epicenter_nodes = _slice_by_station_coverage(
            [(float(lat), float(lon)) for lat in lats for lon in lons],
            station_latlons,
            max_distance_km,
            config.n_phases,
            config.min_picks_per_event,
        )

    if config.fix_depth_km is not None:
        candidate_depths = [float(config.fix_depth_km)]
    else:
        if depth_step_km is None or max_depth_km is None:
            raise ValueError(
                "Grid-search seeding requires depth_step_km and "
                "max_depth_km on the initial propagation engine "
                "when depth is not fixed."
            )
        if depth_step_km <= 0.0 or max_depth_km <= 0.0:
            raise ValueError("depth_step_km and max_depth_km must be positive.")
        depths = np.arange(
            float(depth_step_km),
            float(max_depth_km) + 0.5 * float(depth_step_km),
            float(depth_step_km),
        )
        candidate_depths = (
            [float(depth_step_km)] if depths.size == 0 else [float(d) for d in depths]
        )

    origin_seed = (
        get_seed_from_origins(event, data.time_refs[event_index], config)
        if use_origins
        else None
    )
    if origin_seed is not None:
        lat, lon, origin_depth_km, _ = origin_seed
        origin_key = (round(lat, 4), round(lon, 4))
        epicenter_nodes = [(lat, lon)] + [
            node
            for node in epicenter_nodes
            if (round(node[0], 4), round(node[1], 4)) != origin_key
        ]
        if origin_depth_km is not None and config.fix_depth_km is None:
            depth = float(origin_depth_km)
            candidate_depths = [depth] + [
                float(candidate)
                for candidate in candidate_depths
                if abs(float(candidate) - depth) > 1e-9
            ]
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
        return
    best_solution = _score_candidates(
        data, engine, event_index, epicenter_nodes, candidate_depths
    )
    if best_solution is None:
        logger.warning(
            "Event %d: no scored seed candidate; skipping seed.",
            event_index,
        )
        return
    _, lat_best, lon_best, depth_best_km, t0_best_rel = best_solution
    data.locations[event_index] = (lat_best, lon_best, depth_best_km)
    data.origin_times[event_index] = t0_best_rel
    logger.debug(
        "[init:grid] ev#%d: lat=%.4f lon=%.4f z=%.2f km, t0_rel=%.3fs, t0_abs=%.3fs",
        event_index,
        lat_best,
        lon_best,
        depth_best_km,
        t0_best_rel,
        t0_best_rel + data.time_refs[event_index],
    )


def seed_locations(
    config: LocatorConfig,
    data: DataHandler,
    run_config: RunConfig,
    trial_depth_km: float,
    depth_step_km: Optional[float],
    max_depth_km: Optional[float],
    propagation_engine: Optional[PropagationEngine] = None,
    scoring_engine: Optional[PropagationEngine] = None,
) -> None:
    """Initialize event hypocenters and origin-time seeds before inversion.

    ``propagation_engine`` (the optional initial table) runs travel-time
    grid search. Otherwise a multi-start heuristic proposes the catalog
    Origin (when requested), the network mean, an early-P epicenter
    extrapolated outside the station hull, and the earliest-P station.
    Extra trials, within ``num_trials``, are Gaussian samples about
    those anchors. A nearly linear array (``λ_min/λ_max < 0.1``) draws
    them along the array with cross-track scatter ``sqrt(λ_min)``.
    Candidates are scored with ``scoring_engine``, the inversion
    forward model. A 6 km/s straight-line time is used when that
    model is absent or scores none of them.
    """
    stations_by_key = {
        sta: (
            idx,
            float(data.stations_locations[idx][0]),
            float(data.stations_locations[idx][1]),
            float(data.stations_locations[idx][2]),
        )
        for idx, sta in enumerate(data.stations)
    }

    for event_index, event in enumerate(data.events):
        if len(event.picks) == 0:
            continue

        known_picks = []
        event_stations: dict[tuple[str, str], tuple[int, float, float, float]] = {}
        for pick in event.picks:
            key = (pick.waveform_id.network_code, pick.waveform_id.station_code)
            rec = stations_by_key.get(key)
            if rec is None:
                continue
            known_picks.append(pick)
            event_stations[key] = rec
        if not event_stations:
            logger.warning(
                "Event %d has no picks from known stations; skipping seed.",
                event_index,
            )
            continue

        if propagation_engine is None:
            _seed_heuristic(
                config,
                data,
                event,
                event_index,
                known_picks,
                event_stations,
                trial_depth_km,
                max_depth_km,
                run_config,
                scoring_engine=scoring_engine,
            )
        else:
            _seed_grid_search(
                config,
                data,
                event,
                event_index,
                event_stations,
                depth_step_km,
                max_depth_km,
                propagation_engine,
                run_config.use_origins,
            )
