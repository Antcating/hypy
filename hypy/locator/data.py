import copy
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import time_error_weight
from .types import Catalog, Event, Inventory


def allocate_arrays(
    *,
    n_events: int,
    n_stations: int,
    n_phases: int,
    dim: int,
) -> Dict[str, np.ndarray]:
    arrivals = np.full((n_events, n_phases * n_stations), np.nan)
    arrival_time_errors = np.full((n_events, n_phases * n_stations), np.nan)
    theoretical_travel_times = np.zeros((n_events, n_phases * n_stations))

    derivative_matrix = np.zeros(
        (n_events * n_phases * n_stations, n_events * dim),
        dtype=float,
    )

    n_observations = n_events * n_stations * n_phases
    time_error_W = np.ones(n_observations, dtype=float)
    distance_W = np.ones(n_observations, dtype=float)

    locations = np.zeros((n_events, dim - 1), dtype=float)
    origin_times = np.zeros(n_events)
    time_refs = np.zeros(n_events)

    invalid_events = np.zeros(n_events, dtype=bool)

    return {
        "arrivals": arrivals,
        "arrival_time_errors": arrival_time_errors,
        "theoretical_travel_times": theoretical_travel_times,
        "derivative_matrix": derivative_matrix,
        "time_error_W": time_error_W,
        "distance_W": distance_W,
        "locations": locations,
        "origin_times": origin_times,
        "time_refs": time_refs,
        "invalid_events": invalid_events,
    }


def fill_data(
    *,
    events: List[Event],
    stations: List[Tuple[str, str]],
    phases: Sequence[str],
    n_stations: int,
    n_phases: int,
    min_picks_per_event: int,
    time_error_floor: float,
    arrivals: np.ndarray,
    arrival_time_errors: np.ndarray,
    time_refs: np.ndarray,
    time_error_W: np.ndarray,
    invalid_events: np.ndarray,
) -> Dict[str, np.ndarray]:
    phase_set = {p.lower() for p in phases}
    active_station_set = set(stations)

    for event_idx, event in enumerate(events):
        if len(event.picks) < min_picks_per_event:
            logger.warning(
                f"Event {event.resource_id} has only {len(event.picks)} picks, skipping."
            )
            invalid_events[event_idx] = True
            continue

        picked_stations = {
            (pk.waveform_id.network_code, pk.waveform_id.station_code)
            for pk in event.picks
        }
        if len(picked_stations) * n_phases < min_picks_per_event:
            logger.warning(
                f"Event {event.resource_id} has picks at only {len(picked_stations)} stations, skipping."
            )
            invalid_events[event_idx] = True
            continue

        matching_pick_times: List[float] = []
        for pk in event.picks:
            key = (pk.waveform_id.network_code, pk.waveform_id.station_code)
            if key in active_station_set and (pk.phase_hint or "").lower() in phase_set:
                matching_pick_times.append(float(pk.time))

        if not matching_pick_times:
            invalid_events[event_idx] = True
            logger.warning(
                f"Event {event.resource_id} has no picks matching selected phases ({', '.join(phases)}) to solve for, skipping."
            )
            continue

        time_refs[event_idx] = float(np.mean(matching_pick_times))

        for station_idx, (network, station) in enumerate(stations):
            for phase_idx, phase in enumerate(phases):
                pick = next(
                    (
                        pk
                        for pk in event.picks
                        if pk.waveform_id.station_code == station
                        and pk.waveform_id.network_code == network
                        and (pk.phase_hint or "").lower() == phase.lower()
                    ),
                    None,
                )
                if pick is None:
                    continue

                arrival_time = float(pick.time) - time_refs[event_idx]
                arrival_time_error = (
                    time_error_floor
                    if (
                        pick.time_errors is None
                        or pick.time_errors.uncertainty is None
                        or pick.time_errors.uncertainty <= 0.0
                    )
                    else float(pick.time_errors.uncertainty)
                )
                if arrival_time_error == time_error_floor:
                    logger.warning(
                        f"Pick {network}.{station} {phase}: invalid error → using floor {time_error_floor}s."
                    )

                j = n_phases * station_idx + phase_idx
                arrivals[event_idx, j] = arrival_time
                arrival_time_errors[event_idx, j] = arrival_time_error
                time_error_W[event_idx * n_phases * n_stations + j] = time_error_weight(
                    arrival_time_error
                )

    return {
        "arrivals": arrivals,
        "arrival_time_errors": arrival_time_errors,
        "time_refs": time_refs,
        "time_error_W": time_error_W,
        "invalid_events": invalid_events,
    }


def parse_data(
    *,
    inventory: Inventory,
    events: Union[List[Event], Catalog],
    phases: Sequence[str],
    min_picks_per_event: int,
    time_error_floor: float,
    fix_depth: bool,
    dim: int,
) -> Dict[str, object]:
    """Load inventory/events and build all arrays.

    Returns a dict containing all values previously written to EventLocator.
    """
    events_list = copy.deepcopy(list(events))
    n_events = len(events_list)

    inventory_station_map: Dict[Tuple[str, str], Tuple[float, float, float]] = {
        (net.code, sta.code): (sta.latitude, sta.longitude, sta.elevation)
        for net in inventory
        for sta in net.stations
    }

    picked_station_keys = {
        (pick.waveform_id.network_code, pick.waveform_id.station_code)
        for ev in events_list
        for pick in ev.picks
    }

    missing_stations = picked_station_keys - set(inventory_station_map.keys())
    if missing_stations:
        logger.warning(
            "Stations present in picks but missing in inventory: %s",
            sorted(missing_stations),
        )

    active_station_keys = sorted(
        picked_station_keys & set(inventory_station_map.keys())
    )
    stations: List[Tuple[str, str]] = list(active_station_keys)
    n_stations = len(stations)
    stations_locations = np.array(
        [inventory_station_map[key] for key in stations],
        dtype=float,
    )

    n_phases = len(tuple(phases))
    if n_phases * n_stations < min_picks_per_event:
        raise ValueError(
            f"{n_stations} stations * {n_phases} phases < {min_picks_per_event} required picks/event."
        )

    fixed_depth_mask = [
        fix_depth
        or (len(ev.origins) > 0 and ev.origins[-1].depth_type == "operator assigned")
        for ev in events_list
    ]

    arrays = allocate_arrays(
        n_events=n_events,
        n_stations=n_stations,
        n_phases=n_phases,
        dim=dim,
    )

    filled = fill_data(
        events=events_list,
        stations=stations,
        phases=tuple(phases),
        n_stations=n_stations,
        n_phases=n_phases,
        min_picks_per_event=min_picks_per_event,
        time_error_floor=time_error_floor,
        arrivals=arrays["arrivals"],
        arrival_time_errors=arrays["arrival_time_errors"],
        time_refs=arrays["time_refs"],
        time_error_W=arrays["time_error_W"],
        invalid_events=arrays["invalid_events"],
    )

    arrays.update(filled)

    return {
        "inv": inventory,
        "events": events_list,
        "n_events": n_events,
        "stations": stations,
        "stations_locations": stations_locations,
        "n_stations": n_stations,
        "fixed_depth_mask": fixed_depth_mask,
        **arrays,
    }
