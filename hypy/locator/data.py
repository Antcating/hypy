from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import time_error_weight
from .types import Catalog, Event, Inventory, List, Tuple, np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state import LocatorState

class DataMixin:
    inv: Inventory
    stations: List[Tuple[str, str]]
    stations_locations: np.ndarray
    n_stations: int
    phases: tuple[str, ...]
    n_phases: int
    min_picks_per_event: int
    time_error_floor: float

    events: Catalog | list[Event]
    n_events: int

    _fix_depth: bool

    def load_data(self: "LocatorState", inv: Inventory, events: list[Event] | Catalog) -> None:
        """Load station inventory and events. Must be called before locate()."""
        # Stations from inventory
        self.inv = inv
        all_stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        all_stations_locations = [
            (sta.latitude, sta.longitude, sta.elevation)
            for net in inv for sta in net.stations
        ]

        # Events & basic checks
        self.events = events
        self.n_events = len(events)

        # Stations present in picks
        ev_stations = list({
            (pk.waveform_id.network_code, pk.waveform_id.station_code)
            for ev in events for pk in ev.picks
        })

        missing = [sta for sta in ev_stations if sta not in all_stations]
        if missing:
            logger.warning(
                f"Stations present in picks but missing in inventory: {missing}"
            )

        stations = list(set(all_stations) & set(ev_stations))
        self.stations = stations
        self.n_stations = len(stations)
        self.stations_locations = np.array(
            [all_stations_locations[all_stations.index(sta)] for sta in stations],
            dtype=float
        )

        if self.n_phases * self.n_stations < self.min_picks_per_event:
            raise ValueError(
                f"{self.n_stations} stations * {self.n_phases} phases < "
                f"{self.min_picks_per_event} required picks/event."
            )

        # Fixed-depth mask (global flag or operator-assigned depth)
        self.fixed_depth_mask = [
            self._fix_depth or (
                len(ev.origins) > 0 and ev.origins[-1].depth_type == "operator assigned"
            ) for ev in events
        ]

        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self._fill_data()

    def _allocate_arrays(self: "LocatorState", n_events: int, n_stations: int, n_phases: int) -> None:
        self.origin_times = np.zeros(n_events)
        self.arrivals = np.full((n_events, n_phases * n_stations), np.nan)
        self.arrival_time_errors = np.full((n_events, n_phases * n_stations), np.nan)
        self.theoretical_arrivals = np.zeros((n_events, n_phases * n_stations))
        self.derivative_matrix = np.zeros((n_events * n_phases * n_stations, n_events * self.dim))
        self.time_error_W = np.ones(n_events * n_phases * n_stations)
        self.distance_W = np.ones(n_events * n_phases * n_stations)
        self.locations = np.zeros((n_events, self.dim - 1))
        self.invalid_events = np.zeros(n_events, dtype=bool)

    def _fill_data(self) -> None:
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
                        (pk for pk in event.picks
                         if pk.waveform_id.station_code == station
                         and pk.waveform_id.network_code == network
                         and pk.phase_hint.lower() == phase.lower()),
                        None,
                    )
                    if pick is None:
                        continue

                    arrival_time = pick.time
                    arrival_time_error = (
                        self.time_error_floor if
                        (pick.time_errors is None or
                         pick.time_errors.uncertainty is None or
                         pick.time_errors.uncertainty <= 0.0)
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
