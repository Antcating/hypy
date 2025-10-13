import copy

import numpy as np

from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import time_error_weight
from .types import TYPE_CHECKING, Catalog, Event, Inventory, List, Tuple

if TYPE_CHECKING:
    from .state import LocatorState
else:

    class LocatorState:
        pass


class DataMixin:
    def __init__(self, *args, **kwargs) -> None:
        # per-instance defaults
        self.inv = None

        self.stations: list = []  # list[tuple[str, str]]
        self.stations_locations = np.empty((0, 3), dtype=float)
        self.n_stations = 0

        self.phases: tuple = tuple()
        self.n_phases = 0
        self.min_picks_per_event = 0
        self.time_error_floor = 0.0

        self.events: list = []  # Catalog | list[Event]
        self.n_events = 0

        self.fixed_depth_mask: list = []
        self.invalid_events = np.zeros(0, dtype=bool)

        self.arrivals = np.empty((0, 0))
        self.arrival_time_errors = np.empty((0, 0))
        self.theoretical_travel_times = np.empty((0, 0))
        self.derivative_matrix = np.empty((0, 0))
        self.time_error_W = np.empty((0,), dtype=float)
        self.distance_W = np.empty((0,), dtype=float)
        self.locations = np.empty((0, 3), dtype=float)
        self.origin_times = np.empty((0,), dtype=float)

        super().__init__(*args, **kwargs)  # cooperative MRO

    def load_data(
        self: "LocatorState", inventory: Inventory, events: list[Event] | Catalog
    ) -> None:
        """
        Load inventory and events, extract the set of stations present in picks,
        validate counts, allocate arrays, and populate arrival/time-error matrices.

        Parameters
        ----------
        inventory : Inventory
            ObsPy Inventory with station metadata.
        events : list[Event] | Catalog
            ObsPy Event objects (with picks) or Catalog.

        Raises
        ------
        ValueError
            If (#stations * #phases) < min_picks_per_event.
        """
        # -------------------- 1) Basic assignments --------------------
        self.inv = inventory
        self.events = copy.deepcopy(list(events))
        self.n_events = len(events)

        # -------------------- 2) Build a fast lookup for all inventory stations --------------------
        # Map ("NET", "STA") → (latitude, longitude, elevation)
        inventory_station_map: dict[tuple[str, str], tuple[float, float, float]] = {
            (net.code, sta.code): (sta.latitude, sta.longitude, sta.elevation)
            for net in inventory
            for sta in net.stations
        }

        # -------------------- 3) Collect all stations referenced in picks --------------------
        picked_station_keys: set[tuple[str, str]] = {
            (pick.waveform_id.network_code, pick.waveform_id.station_code)
            for event in events
            for pick in event.picks
        }

        # -------------------- 4) Validate & determine the active subset --------------------
        # Stations present in picks but missing in inventory
        missing_stations = picked_station_keys - inventory_station_map.keys()
        if missing_stations:
            logger.warning(
                "Stations present in picks but missing in inventory: %s",
                sorted(missing_stations),
            )

        # Keep only those that exist in both picks and inventory
        active_station_keys = sorted(picked_station_keys & inventory_station_map.keys())

        # -------------------- 5) Assign to instance attributes --------------------
        self.stations = active_station_keys
        self.n_stations = len(active_station_keys)
        self.stations_locations = np.array(
            [inventory_station_map[key] for key in active_station_keys],
            dtype=float,
        )

        # -------------------- 3) Validate minimal data sufficiency --------------------
        if self.n_phases * self.n_stations < self.min_picks_per_event:
            raise ValueError(
                f"{self.n_stations} stations * {self.n_phases} phases < "
                f"{self.min_picks_per_event} required picks/event."
            )

        # -------------------- 4) Build fixed-depth mask --------------------
        # An event is depth-fixed if global `_fix_depth` is set OR its last origin has operator-assigned depth.
        self.fixed_depth_mask = [
            self._fix_depth
            or (
                len(ev.origins) > 0 and ev.origins[-1].depth_type == "operator assigned"
            )
            for ev in events
        ]

        # -------------------- 5) Allocate working arrays and populate them --------------------
        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self._fill_data()

    def _allocate_arrays(
        self: "LocatorState", n_events: int, n_stations: int, n_phases: int
    ) -> None:
        """
        Allocate all working arrays used during event location.

        Parameters
        ----------
        n_events : int
            Number of events being processed.
        n_stations : int
            Number of active stations (with metadata and picks).
        n_phases : int
            Number of seismic phases considered (e.g., P, S).

        Notes
        -----
        This method preallocates all major data containers for efficiency.
        Array shapes are chosen to allow fast vectorized operations later
        in the inversion and residual calculations.
        """

        # --- Observed and theoretical arrival-time matrices ---
        # Shape: (n_events, n_stations * n_phases)
        # Each row corresponds to an event; each column to a station–phase pair.
        self.arrivals = np.full((n_events, n_phases * n_stations), np.nan)
        self.arrival_time_errors = np.full((n_events, n_phases * n_stations), np.nan)
        self.theoretical_travel_times = np.zeros((n_events, n_phases * n_stations))

        # --- Derivative (design) matrix ---
        # Shape: (n_events * n_stations * n_phases, n_events * self.dim)
        # Each event has `self.dim` unknowns. Usually 4: (x, y, z, t0).
        self.derivative_matrix = np.zeros(
            (n_events * n_phases * n_stations, n_events * self.dim),
            dtype=float,
        )

        # --- Weights ---
        # Flattened across all event–station–phase observations.
        # - time_error_W: weighting from pick-time uncertainty (1/sigma^2)
        # - distance_W: optional logistic distance-based weighting
        n_observations = n_events * n_stations * n_phases
        self.time_error_W = np.ones(n_observations, dtype=float)
        self.distance_W = np.ones(n_observations, dtype=float)

        # --- Event locations and origin times ---
        # Shape: (n_events, 3) → (latitude, longitude, depth_km)
        self.locations = np.zeros((n_events, self.dim - 1), dtype=float)
        # Origin times t0 per event (seconds since reference epoch)
        self.origin_times = np.zeros(n_events)

        # --- Event validity mask ---
        # Marks events excluded from inversion due to insufficient data or bad geometry.
        self.invalid_events = np.zeros(n_events, dtype=bool)

    def _fill_data(self: "LocatorState") -> None:
        for event_idx, event in enumerate(self.events):
            if len(event.picks) < self.min_picks_per_event:
                logger.warning(
                    f"Event {event.resource_id} has only {len(event.picks)} picks, skipping."
                )
                self.invalid_events[event_idx] = True
                continue

            picked_stations = {
                (pk.waveform_id.network_code, pk.waveform_id.station_code)
                for pk in event.picks
            }
            if len(picked_stations) * self.n_phases < self.min_picks_per_event:
                logger.warning(
                    f"Event {event.resource_id} has picks at only {len(picked_stations)} stations, skipping."
                )
                self.invalid_events[event_idx] = True
                continue

            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    pick = next(
                        (
                            pk
                            for pk in event.picks
                            if pk.waveform_id.station_code == station
                            and pk.waveform_id.network_code == network
                            and pk.phase_hint.lower() == phase.lower()
                        ),
                        None,
                    )
                    if pick is None:
                        continue

                    arrival_time = pick.time
                    arrival_time_error = (
                        self.time_error_floor
                        if (
                            pick.time_errors is None
                            or pick.time_errors.uncertainty is None
                            or pick.time_errors.uncertainty <= 0.0
                        )
                        else pick.time_errors.uncertainty
                    )
                    if arrival_time_error == self.time_error_floor:
                        logger.warning(
                            f"Pick {network}.{station} {phase}: invalid error → using floor {self.time_error_floor}s."
                        )

                    j = self.n_phases * station_idx + phase_idx
                    self.arrivals[event_idx, j] = arrival_time
                    self.arrival_time_errors[event_idx, j] = arrival_time_error
                    self.time_error_W[
                        event_idx * self.n_phases * self.n_stations + j
                    ] = time_error_weight(arrival_time_error)
