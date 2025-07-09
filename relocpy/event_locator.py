"""
event_locator.py
----------------
Contains the EventLocator class for earthquake location using travel-time tables.
"""
from typing import List, Tuple, Optional
import numpy as np
from obspy.taup import TauPyModel
from obspy.core.inventory import Inventory
from obspy.core.event import Event
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from .travel_time_tables import TravelTimeTables

class EventLocator:
    """
    Earthquake event locator using precomputed travel-time tables.
    """
    def __init__(
        self,
        model: TauPyModel,
        dim: int = 3,
        phases: List[str] = ["P"],
        max_depth: float = 20.0,
        max_distance: float = 100.0,
        depth_step: float = 0.2,
        distance_step: float = 0.5,
    ):
        """
        Initialize the EventLocator with a velocity model and configure travel-time tables.
        """
        self.model = model
        self.dim = dim
        self.phases = phases
        self.n_phases = len(phases)
        self.tables = TravelTimeTables(
            model, phases, max_depth, max_distance, depth_step, distance_step
        )
        self.inv: Optional[Inventory] = None
        self.networks: List[str] = []
        self.stations: List[Tuple[str, str]] = []
        self.stations_locations: List[Tuple[float, float, float]] = []
        self.n_stations: int = 0
        self.events: List[Event] = []
        self.n_events: int = 0
        self.origin_times: Optional[np.ndarray] = None
        self.arrivals: Optional[np.ndarray] = None
        self.theoretical_arrivals: Optional[np.ndarray] = None
        self.derivative_matrix: Optional[lil_matrix] = None
        self.locations: Optional[np.ndarray] = None

    def load_data(self, inv: Inventory, events: List[Event]) -> None:
        """
        Load station inventory (station metadata) and events (with picks) into the locator.
        This must be called before running locate().
        """
        self.inv = inv
        self.networks = [net.code for net in inv]
        self.stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        self.stations_locations = [
            (sta.latitude, sta.longitude, sta.elevation)
            for net in inv
            for sta in net.stations
        ]
        self.n_stations = len(self.stations)
        self.events = events
        self.n_events = len(events)
        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self._fill_data()

    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int) -> None:
        """
        Allocate arrays for origin times, observed arrivals, theoretical arrivals, derivative matrix, and locations.
        """
        self.origin_times = np.zeros(n_events)
        self.arrivals = np.full((n_events, n_phases * n_stations), np.nan)
        self.theoretical_arrivals = np.zeros((n_events, n_phases * n_stations))
        self.derivative_matrix = lil_matrix((n_events * n_phases * n_stations, n_events * self.dim))
        self.locations = np.zeros((n_events, self.dim))

    def _fill_data(self) -> None:
        """
        Populate the arrivals matrix with observed pick times from each event.
        """
        for event_idx, event in enumerate(self.events):
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
                    arrival_time = np.nan if pick is None else pick.time
                    self.arrivals[event_idx, self.n_phases * station_idx + phase_idx] = arrival_time

    def _interpolate_travel_time(self, phase: str, depth: float, distance: float) -> float:
        return self.tables.interpolate_travel_time(phase, depth, distance)

    def _interpolate_takeoff_angle(self, phase: str, depth: float, distance: float) -> float:
        return self.tables.interpolate_takeoff_angle(phase, depth, distance)

    def _interpolate_ray_param(self, phase: str, depth: float, distance: float) -> float:
        return self.tables.interpolate_ray_param(phase, depth, distance)

    def _derive_travel_time_param(self, event_location, station_location, phase):
        from obspy.geodetics import gps2dist_azimuth
        ev_lat, ev_lon, ev_depth = event_location
        st_lat, st_lon, st_elev = station_location
        dist_m, az_deg, _ = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
        distance_km = dist_m / 1000.0
        travel_time = self._interpolate_travel_time(phase, ev_depth + st_elev / 1000, distance_km)
        takeoff_angle = self._interpolate_takeoff_angle(phase, ev_depth + st_elev / 1000, distance_km)
        vel = self.model.model.s_mod.v_mod.evaluate_below(ev_depth, phase.lower())[0]
        takeoff_rad = np.deg2rad(takeoff_angle)
        dt_dr = np.sin(takeoff_rad) / vel
        dt_dz = -np.cos(takeoff_rad) / vel
        az_rad = np.deg2rad(az_deg)
        dt_dx = -dt_dr * np.cos(az_rad)
        dt_dy = -dt_dr * np.sin(az_rad)
        return travel_time, dt_dx, dt_dy, dt_dz

    def init_locations(self, trial_depth: float = 10.0, trial_time: float = 2.0) -> np.ndarray:
        """
        Initialize the event locations and origin times for iterative location.
        """
        for event_idx, event in enumerate(self.events):
            if len(event.picks) == 0:
                continue
            first_pick = min(event.picks, key=lambda pk: pk.time)
            net_code = first_pick.waveform_id.network_code
            sta_code = first_pick.waveform_id.station_code
            station = self.inv.select(network=net_code, station=sta_code)[0][0]
            self.locations[event_idx] = (station.latitude, station.longitude, trial_depth)
            self.origin_times[event_idx] = first_pick.time - trial_time
        return self.locations

    def locate(
        self,
        max_iterations: int = 10,
        conv_threshold: float = 1e-4,
        solve_airquakes: bool = False,
        max_depth: float = 20.0,
        max_depth_iterations: int = 5,
        damping_list: Optional[List[float]] = None,
    ):
        """
        Perform iterative earthquake location for all loaded events using a linearized least-squares approach.
        """
        from obspy.geodetics import kilometer2degrees
        self.init_locations()
        prev_max_update = np.inf
        if damping_list is None:
            damping_list = [0.0] * max_iterations
        self.invalid_events = np.zeros(self.n_events, dtype=bool)
        for it in range(max_iterations):
            valid_rows = []
            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                ev_lat, ev_lon, ev_depth = self.locations[event_idx]
                for station_idx in range(self.n_stations):
                    sta_lat, sta_lon, sta_elev = self.stations_locations[station_idx]
                    for phase_idx, phase in enumerate(self.phases):
                        col_idx = event_idx * self.dim
                        arr_index = self.n_phases * station_idx + phase_idx
                        observed_time = self.arrivals[event_idx, arr_index]
                        if np.isnan(observed_time):
                            continue
                        travel_time, dt_dx, dt_dy, dt_dz = self._derive_travel_time_param(
                            (ev_lat, ev_lon, ev_depth), (sta_lat, sta_lon, sta_elev), phase
                        )
                        self.theoretical_arrivals[event_idx, arr_index] = travel_time
                        row_idx = event_idx * (self.n_stations * self.n_phases) + arr_index
                        if self.dim >= 2:
                            self.derivative_matrix[row_idx, col_idx + 0] = dt_dx
                            self.derivative_matrix[row_idx, col_idx + 1] = dt_dy
                        if self.dim == 3:
                            self.derivative_matrix[row_idx, col_idx + 2] = dt_dz
                        valid_rows.append(row_idx)
            travel_times_obs = self.arrivals - self.origin_times[:, np.newaxis]
            residuals = self.theoretical_arrivals - travel_times_obs
            residuals_flat = residuals.flatten()
            valid_rows = np.array(valid_rows, dtype=int)
            b = residuals_flat[valid_rows]
            A = self.derivative_matrix.tocsr()[valid_rows, :]
            if self.n_events > 0:
                residual_mean = np.zeros(self.n_events)
                for event_idx in range(self.n_events):
                    event_rows = [
                        idx
                        for idx, row in enumerate(valid_rows)
                        if (row // (self.n_stations * self.n_phases)) == event_idx
                    ]
                    if event_rows:
                        mean_res = np.mean(b[event_rows])
                        residual_mean[event_idx] = mean_res
                        b[event_rows] -= mean_res
                self.origin_times -= residual_mean
            from scipy.sparse.linalg import lsqr
            damping = damping_list[it] if it < len(damping_list) else damping_list[-1]
            dx_solution = lsqr(A, b, damp=damping, atol=1e-6, btol=1e-6, iter_lim=1000)[0]
            dx = dx_solution.reshape((self.n_events, self.dim))
            max_step_xy = 3.0
            max_step_z = 2.0
            for d in range(min(self.dim, 3)):
                if d < 2:
                    dx[:, d] = np.clip(dx[:, d], -max_step_xy, max_step_xy)
                else:
                    dx[:, d] = np.clip(dx[:, d], -max_step_z, max_step_z)
            for event_idx in range(self.n_events):
                if self.invalid_events[event_idx]:
                    continue
                new_depth = self.locations[event_idx, 2] - (dx[event_idx, 2] if self.dim == 3 else 0.0)
                if (not solve_airquakes and new_depth < 0.0) or (
                    it > max_depth_iterations and new_depth > max_depth
                ):
                    print(f"Event {event_idx} would have depth {new_depth:.2f} km out of bounds. Marking as invalid.")
                    self.invalid_events[event_idx] = True
                    if self.dim == 3:
                        dx[event_idx, 2] = 0.0
            if self.dim >= 2:
                dx[:, 0] = kilometer2degrees(dx[:, 0])
                dx[:, 1] = kilometer2degrees(dx[:, 1])
            max_update = 0.0
            if dx.size > 0:
                valid_mask = ~self.invalid_events
                if np.any(valid_mask):
                    max_update = np.max(np.abs(dx[valid_mask]))
            self.locations -= dx
            if max_update < conv_threshold:
                print(f"Converged after {it + 1} iterations (max update {max_update:.3f} km < {conv_threshold} km).")
                break
            if (
                prev_max_update != 0
                and abs(prev_max_update - max_update) / prev_max_update < 1e-5
                and max_update < 0.5
            ):
                print(f"Converged (minor change) after {it + 1} iterations (max update {max_update:.3f} km).")
                break
            prev_max_update = max_update
        return self.locations, self.origin_times, max_update

    def report_invalid_events(self) -> np.ndarray:
        """
        Return an array of indices for events that were marked as invalid (e.g., requiring non-physical depths).
        """
        if hasattr(self, "invalid_events"):
            return np.where(self.invalid_events)[0]
        return np.array([])
