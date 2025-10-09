from ..logger import logger  # type: ignore[import-not-found]
from ..utils.weighting import time_error_weight
from .types import Catalog, Event, Inventory, List, Tuple, np
from obspy.core.event import Pick

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
        self.events = events
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
        """
        Populate observed arrivals, their uncertainties, and time-based weights
        for each event-station-phase observation. Events with insufficient or
        poorly distributed picks are marked invalid early.
        """
        for event_index, event in enumerate(self.events):
            # ---- Early exclusion: too few picks overall ----
            # Avoids underdetermined inversions even if geometry is fine.
            num_event_picks = len(event.picks)
            if num_event_picks < self.min_picks_per_event:
                logger.warning(
                    "Event %s has only %d picks; marking invalid.",
                    getattr(event, "resource_id", "<no-id>"),
                    num_event_picks,
                )
                self.invalid_events[event_index] = True
                continue

            # ---- Quick coverage check: unique stations represented for this event ----
            # Prevents unstable solutions (many picks from few stations).
            station_keys_for_event = {
                (pk.waveform_id.network_code, pk.waveform_id.station_code)
                for pk in event.picks
            }
            if len(station_keys_for_event) * self.n_phases < self.min_picks_per_event:
                logger.warning(
                    "Event %s has picks at only %d stations; marking invalid.",
                    getattr(event, "resource_id", "<no-id>"),
                    len(station_keys_for_event),
                )
                self.invalid_events[event_index] = True
                continue

            # ---- Build a lookup: (net, sta, phase_lower) -> pick ----
            pick_lookup: dict[tuple[str, str, str], Pick] = {}
            for pk in event.picks:
                if pk is None or pk.waveform_id is None:
                    continue  # Skip invalid picks
                net = pk.waveform_id.network_code
                sta = pk.waveform_id.station_code
                ph = (pk.phase_hint or "").lower()
                if net and sta and ph:
                    pick_lookup[(net, sta, ph)] = pk

            # ---- Fill arrivals, uncertainties, and per-observation weights ----
            for station_index, (network_code, station_code) in enumerate(self.stations):
                for phase_index, phase_name in enumerate(self.phases):
                    key = (network_code, station_code, phase_name.lower())
                    pick = pick_lookup.get(key)
                    if pick is None:
                        continue  # No pick for this station/phase → leave NaN, no weight

                    # Observed arrival time
                    observed_time = pick.time

                    # Uncertainty (sigma): use floor if missing/invalid (<= 0)
                    if (
                        pick.time_errors is None
                        or pick.time_errors.uncertainty is None
                        or pick.time_errors.uncertainty <= 0.0
                    ):
                        sigma = self.time_error_floor
                        logger.warning(
                            "Pick %s.%s %s: invalid/undefined time error → using floor %.3fs.",
                            network_code,
                            station_code,
                            phase_name,
                            self.time_error_floor,
                        )
                    else:
                        sigma = pick.time_errors.uncertainty

                    # Column index within event's (station×phase) axis
                    obs_col = self.n_phases * station_index + phase_index

                    # Write observed time & its uncertainty
                    self.arrivals[event_index, obs_col] = observed_time
                    self.arrival_time_errors[event_index, obs_col] = sigma

                    # Global flattened index for weight vectors (event-major)
                    obs_flat_index = (
                        event_index * (self.n_phases * self.n_stations) + obs_col
                    )

                    # Time-based weight derived from uncertainty (your mapping)
                    self.time_error_W[obs_flat_index] = time_error_weight(sigma)
