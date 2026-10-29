"""Data handler for seismic information and associated arrays used during inversion."""

import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from obspy.core.event import Catalog, Event  # type: ignore[import-untyped]
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]

from ..logger import logger
from ..utils.seismic import classify_phase
from ..utils.stations.corrections import (
    apply_station_time_corrections,
)
from ..utils.weighting import time_error_weight
from .config import LocatorConfig
from .types import DIM


class DataHandler:
    """Manage stations, events, and the arrays used during inversion.

    Parameters
    ----------
    config : LocatorConfig
        Shared, immutable construction-time configuration.
    """

    def __init__(self, config: LocatorConfig) -> None:
        self.config = config
        # Depth is fixed by the global setting, the per-event input mask, or
        # inversion when a free-depth step leaves the model bounds.
        self.fixed_depth_mask: list = []
        self.inventory: Inventory
        self.events: Union[List[Event], Catalog]
        self.n_events: int
        self.stations: list = []
        self.stations_locations = np.empty((0, 3), dtype=float)
        self.n_stations = 0
        self.invalid_events = np.zeros(0, dtype=bool)
        self.arrivals = np.empty((0, 0))
        self.arrival_time_errors = np.empty((0, 0))
        self.theoretical_travel_times = np.empty((0, 0))
        self.jacobian_matrix = np.empty((0, 0))
        self.time_error_W = np.empty((0,), dtype=float)
        self.distance_W = np.empty((0,), dtype=float)
        self.residual_W = np.empty((0,), dtype=float)
        self.locations = np.empty((0, 3), dtype=float)
        self.origin_times = np.empty((0,), dtype=float)
        # Reference time to calculate around (mean of all picks of the event).
        # Avoids big float imprecisions.
        self.time_refs = np.empty((0,), dtype=float)
        self.station_time_corrections = np.empty((0,), dtype=float)
        self.fallback_seeds: list[list[tuple[float, float, float, float]]] = []

    def resolve_fixed_depth_mask(self) -> np.ndarray:
        """Return the per-event depth-freeze flags as a boolean array.

        A global ``fix_depth_km`` freezes every event. A mask that is not
        sized for the loaded events has not been populated yet, so nothing
        is frozen.
        """
        if self.config.fix_depth_km is not None:
            return np.ones(self.n_events, dtype=bool)
        if len(self.fixed_depth_mask) != self.n_events:
            return np.zeros(self.n_events, dtype=bool)
        return np.asarray(self.fixed_depth_mask, dtype=bool)

    def load_data(
        self,
        inventory: Inventory,
        events: Union[List[Event], Catalog],
        station_time_corrections: Optional[pd.DataFrame] = None,
        fixed_depth_mask: Optional[Sequence[bool]] = None,
    ) -> None:
        """Load inventory and events, then build inversion arrays.

        Cross-references picked stations with the inventory, validates
        counts, allocates arrays, and fills arrival / time-error matrices.

        Parameters
        ----------
        inventory : Inventory
            ObsPy Inventory with station metadata.
        events : list[Event] | Catalog
            ObsPy Event objects (with picks) or Catalog.
        station_time_corrections : pd.DataFrame, optional
            Known station delays (seconds), with columns ``network``,
            ``station``, ``delay_s`` and an optional ``phase`` (omitted or
            ``*`` means station-wide). Positive delay means the observed
            arrival is late. Default is no corrections (all zeros).
        fixed_depth_mask : sequence of bool, optional
            Per-event flags selecting depths to hold fixed. Must match the
            event order and count. The global ``fix_depth_km`` setting
            overrides the values and fixes every event.

        Raises
        ------
        ValueError
            If (#stations * #phases) < min_picks_per_event.
        """
        self.inventory = inventory
        self.events = copy.deepcopy(list(events))
        self.n_events = len(self.events)
        if fixed_depth_mask is None:
            mask = np.zeros(self.n_events, dtype=bool)
        else:
            mask = np.asarray(fixed_depth_mask)
            if mask.shape != (self.n_events,) or mask.dtype != np.bool_:
                raise ValueError("fixed_depth_mask must contain one boolean per event.")
        if self.config.fix_depth_km is not None:
            mask = np.ones(self.n_events, dtype=bool)
        self.fixed_depth_mask = mask.tolist()

        allowed = set(self.config.phases)
        for event in self.events:
            kept = []
            for pick in event.picks:
                if classify_phase(pick.phase_hint) in allowed:
                    kept.append(pick)
                    continue
                logger.warning(
                    "Skipping pick %s.%s %r: phase not in %s.",
                    pick.waveform_id.network_code,
                    pick.waveform_id.station_code,
                    pick.phase_hint,
                    self.config.phases,
                )
            event.picks = kept

        inventory_map = {
            (net.code, sta.code): (sta.latitude, sta.longitude, sta.elevation)
            for net in inventory
            for sta in net.stations
        }
        picked = {
            (pick.waveform_id.network_code, pick.waveform_id.station_code)
            for event in self.events
            for pick in event.picks
        }
        missing = picked - inventory_map.keys()
        if missing:
            logger.warning(
                "Stations present in picks but missing in inventory: %s",
                sorted(missing),
            )
        self.stations = sorted(picked & inventory_map.keys())
        self.n_stations = len(self.stations)
        self.stations_locations = np.array(
            [inventory_map[key] for key in self.stations], dtype=float
        )
        logger.info(
            f"Loaded {self.n_events} events and {self.n_stations} stations with"
            f" picks for phases {self.config.phases}."
        )

        if self.config.n_phases * self.n_stations < self.config.min_picks_per_event:
            raise ValueError(
                f"{self.n_stations} stations * {self.config.n_phases} phases < "
                f"{self.config.min_picks_per_event} required picks/event."
            )

        n_fixed = sum(self.fixed_depth_mask)
        if n_fixed:
            logger.info(
                f"{n_fixed} out of {self.n_events} events"
                " will be depth-fixed during inversion."
            )

        self._allocate_arrays()
        self.station_time_corrections = apply_station_time_corrections(
            station_time_corrections,
            self.stations,
            self.config.phases,
        )
        self._fill_data()

    def _allocate_arrays(self) -> None:
        """Preallocate inversion arrays from the current event/station counts."""
        n_events = self.n_events
        n_cols = self.config.n_phases * self.n_stations
        n_obs = n_events * n_cols
        self.arrivals = np.full((n_events, n_cols), np.nan)
        self.arrival_time_errors = np.full((n_events, n_cols), np.nan)
        self.theoretical_travel_times = np.zeros((n_events, n_cols))
        self.jacobian_matrix = np.zeros((n_obs, DIM), dtype=float)
        self.time_error_W = np.ones(n_obs, dtype=float)
        self.distance_W = np.ones(n_obs, dtype=float)
        self.residual_W = np.ones(n_obs, dtype=float)
        self.locations = np.zeros((n_events, DIM - 1), dtype=float)
        self.origin_times = np.zeros(n_events)
        self.time_refs = np.zeros(n_events)
        self.invalid_events = np.zeros(n_events, dtype=bool)
        self.station_time_corrections = np.zeros(n_cols, dtype=float)
        self.fallback_seeds = [[] for _ in range(n_events)]

    def observation_residuals(self) -> np.ndarray:
        """Return ``(t_obs - t0) - T_pred - dt_sta`` for every observation."""
        return (
            self.arrivals
            - self.origin_times[:, np.newaxis]
            - self.theoretical_travel_times
            - self.station_time_corrections[np.newaxis, :]
        )

    def _fill_data(self) -> None:
        invalid_time_errors = skipped_min_picks = skipped_min_stations = 0
        phase_set = set(self.config.phases)
        station_set = set(self.stations)
        station_index = {key: i for i, key in enumerate(self.stations)}
        phase_index = {phase: i for i, phase in enumerate(self.config.phases)}
        n_obs_per_event = self.config.n_phases * self.n_stations
        min_picks = self.config.min_picks_per_event
        floor_s = self.config.time_error_floor_s

        # Reading all events and validating them
        for event_idx, event in enumerate(self.events):
            # Remove events with less than min_picks_per_event picks
            if len(event.picks) < min_picks:
                logger.debug(
                    f"Skipping event {event.resource_id}: "
                    f"only {len(event.picks)} picks < {min_picks} required."
                )
                self.invalid_events[event_idx] = True
                skipped_min_picks += 1
                continue

            times: list[float] = []
            cols: list[int] = []
            sigmas: list[float] = []
            seen: set[tuple[str, str, str]] = set()
            for pk in event.picks:
                key = (pk.waveform_id.network_code, pk.waveform_id.station_code)
                phase = classify_phase(pk.phase_hint)
                # Keep first P/S arrival per station
                if key not in station_set or phase not in phase_set:
                    continue
                pair = (key[0], key[1], phase)
                if pair in seen:
                    logger.warning(
                        "Duplicate pick for %s.%s phase %s in event %s; "
                        "using the first occurrence.",
                        key[0],
                        key[1],
                        phase,
                        event.resource_id,
                    )
                    continue
                seen.add(pair)

                # Correct picks with pick uncertainty is None or below floor_s
                errors = pk.time_errors
                if (
                    errors is None
                    or errors.uncertainty is None
                    or errors.uncertainty <= floor_s
                ):
                    logger.debug(
                        f"Pick {key[0]}.{key[1]} {phase}: invalid time error."
                        f" Using floor {floor_s} s."
                    )
                    invalid_time_errors += 1
                    sigma = floor_s
                else:
                    sigma = errors.uncertainty
                times.append(float(pk.time))
                cols.append(
                    self.config.n_phases * station_index[key] + phase_index[phase]
                )
                sigmas.append(sigma)

            # Unique station-phase pairs be too few to constrain the event.
            if len(times) < min_picks:
                logger.debug(
                    "Skipping event %s: only %d usable station-phase "
                    "picks < %d required.",
                    event.resource_id,
                    len(times),
                    min_picks,
                )
                self.invalid_events[event_idx] = True
                skipped_min_stations += 1
                continue

            # Reference time is the mean pick time, so later values stay well scaled.
            times_arr = np.asarray(times, dtype=float)
            t_ref = float(times_arr.mean())
            self.time_refs[event_idx] = t_ref
            self.arrivals[event_idx, cols] = times_arr - t_ref
            self.arrival_time_errors[event_idx, cols] = sigmas
            self.time_error_W[
                event_idx * n_obs_per_event + np.asarray(cols, dtype=int)
            ] = time_error_weight(np.asarray(sigmas, dtype=float))

        if invalid_time_errors:
            logger.warning(
                f"{invalid_time_errors} invalid time errors found and "
                f"clamped to {floor_s} s. "
                "Use DEBUG level to see the details."
            )
        if skipped_min_picks:
            logger.warning(
                f"{skipped_min_picks} events skipped due to insufficient "
                "picks per event in provided dataset. "
                "Use DEBUG level to see the details."
            )
        if skipped_min_stations:
            logger.warning(
                f"{skipped_min_stations} events skipped due to insufficient "
                "unique station-phase pairs after phase deduplication. "
                "Use DEBUG level to see the details."
            )
