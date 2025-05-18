from typing import List, Tuple
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees, gps2dist_azimuth
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.event import Pick, Event
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

class EventLocator:
    def __init__(self, model: TauPyModel, dim: int=2, phases: List[str]=["P"]):

        self.model = model
        # Dimension of the problem (2 or 3) for 2D or 3D location
        self.dim = dim
        # Phases to use for location
        self.phases = phases
        self.n_phases = len(phases)

        self.inv: Inventory = None

        self.events: List[Event] = None
        self.n_events: int = None

        self.networks: List[str] = None
        self.stations: List[Tuple[str, str]] = None
        self.stations_locations: List[Tuple[float, float, float]] = None
        self.n_stations: int = None

        self.origin_times: np.ndarray = None
        self.arrivals: np.ndarray = None
        self.theoretical_arrivals: np.ndarray = None
        self.derivative_matrix: lil_matrix = None
        self.locations: np.ndarray = None

    def load_data(self, inv: Inventory, events: List[Event]):
        self.inv = inv
        self.networks = [net.code for net in inv]
        self.stations = [(net.code, sta.code) for net in inv for sta in net.stations]
        self.stations_locations = [(sta.latitude, sta.longitude, sta.elevation) for net in inv for sta in net.stations]
        self.n_stations = len(self.stations)
        self.events = events
        self.n_events = len(self.events)
        self._allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self._fill_data()

    def _allocate_arrays(self, n_events: int, n_stations: int, n_phases: int):
        # Origin times for all events
        self.origin_times = np.zeros(n_events)

        # Picks recorded for all events
        self.arrivals = np.zeros((n_events, (n_phases * n_stations)))
        # Theoretical arrivals for all events per station
        self.theoretical_arrivals = np.zeros((n_events, (n_phases * n_stations)))
        # Derivative matrix for all events per station and per coordinate
        self.derivative_matrix = lil_matrix((n_events * (n_phases * n_stations), n_events * self.dim))
        # Locations of the events (solution)
        self.locations = np.zeros((n_events, self.dim))

    def _fill_data(self):
        for event_idx, event in enumerate(self.events):
            for station_idx, (network, station) in enumerate(self.stations):
                for phase_idx, phase in enumerate(self.phases):
                    arrival = next(
                        (pick for pick in event.picks
                        if pick.waveform_id.station_code == station and pick.waveform_id.network_code == network and pick.phase_hint.lower() == phase.lower()),
                        None
                    )
                    if arrival is None:
                        arrival_time = np.nan
                    else:
                        arrival_time = arrival.time
                    self.arrivals[event_idx, self.n_phases * station_idx + phase_idx] = arrival_time

    def analyt_dt(self, arrival, phase: str):
        """
        Calculate the derivatives of travel time with respect to radial distance and depth.

        Parameters:
        ----------
        arrival (Arrival): An object containing information about the arrival, including the path and takeoff angle.
        
        Returns:
        --------
        tuple: A tuple containing two elements:
            - dt_dr (float): The derivative of travel time with respect to radial distance.
            - dt_dz (float): The derivative of travel time with respect to depth.
        """
        
        path = arrival.path
        depth = path[np.argmax([p[-1] for p in path])][-1]
        vel = self.model.model.s_mod.v_mod.evaluate_below(depth , phase.lower())[0]
        dt_dr = np.sin(np.deg2rad(arrival.takeoff_angle)) / vel
        dt_dz = -np.cos(np.deg2rad(arrival.takeoff_angle)) / vel
        return dt_dr, dt_dz
    
    def get_arrival(self, event_location, station_location, phases: List[str]):
        arrivals = self.model.get_ray_paths_geo(
            source_latitude_in_deg=event_location[0], source_longitude_in_deg=event_location[1], source_depth_in_km=event_location[2],
            receiver_latitude_in_deg=station_location[0], receiver_longitude_in_deg=station_location[1],
            phase_list=phases)
        arrival = arrivals[0]
        return arrival
    
    def _derive_travel_time_param(self, event_location, station_location, phase: str):
        if phase == "P":
            phase_list = ["P", "p"]
        elif phase == "S":
            phase_list = ["S", "s"]
        else:
            raise ValueError("Invalid phase")
        
        arrival = self.get_arrival(event_location, station_location, phase_list)
        time = arrival.time
        dt_dr, dt_dz = self.analyt_dt(arrival, phase)
        # Azimuth is measured from the north
        # dx is latitude, dy is longitude
        _, az, _ = gps2dist_azimuth(event_location[0], event_location[1], station_location[0], station_location[1])
        dt_dx = -dt_dr * np.cos(np.deg2rad(az))
        dt_dy = -dt_dr * np.sin(np.deg2rad(az))
        return time, dt_dx, dt_dy, dt_dz
                        
    
    def _iterate_locate_3d(self, clip_range: float = 5, solve_airquakes: bool = False, max_depth: float = 0, iteration: int = 0, max_depth_iterations: int = 0):
        if not hasattr(self, 'invalid_events'):
            self.invalid_events = np.zeros(self.n_events, dtype=bool)
        valid_rows = []
        for event_idx in range(self.n_events):
            if getattr(self, 'invalid_events', np.zeros(self.n_events, dtype=bool))[event_idx]:
                continue  # skip invalid events
            for station_idx in range(self.n_stations):
                event_location = self.locations[event_idx]
                station_location = self.stations_locations[station_idx]
                for phase_idx, phase in enumerate(self.phases):
                    arr_idx = self.n_phases * station_idx + phase_idx
                    if np.isnan(self.arrivals[event_idx, arr_idx]):
                        continue  # skip if no arrival for this event/station/phase
                    arrival_time, dt_dx, dt_dy, dt_dz = self._derive_travel_time_param(event_location, station_location, phase)
                    self.theoretical_arrivals[event_idx, arr_idx] = arrival_time
                    row_idx = event_idx * (self.n_stations * self.n_phases) + station_idx * self.n_phases + phase_idx
                    self.derivative_matrix[row_idx, event_idx * 3] = dt_dx
                    self.derivative_matrix[row_idx, event_idx * 3 + 1] = dt_dy
                    self.derivative_matrix[row_idx, event_idx * 3 + 2] = dt_dz
                    valid_rows.append(row_idx)
        # Only use valid rows (where arrivals are not NaN), per event
        travel_times = self.arrivals - self.origin_times[:, np.newaxis]
        dt = self.theoretical_arrivals - travel_times
        dt_flat = dt.flatten()
        valid_rows = np.array(valid_rows)
        valid_dt = dt_flat[valid_rows]
        valid_deriv = self.derivative_matrix[valid_rows, :]
        # Remove mean for each event (only for valid arrivals)
        dt_m = np.zeros(self.n_events)
        for event_idx in range(self.n_events):
            event_rows = [i for i, row in enumerate(valid_rows) if row // (self.n_stations * self.n_phases) == event_idx]
            if event_rows:
                dt_m[event_idx] = np.mean(valid_dt[event_rows])
                valid_dt[event_rows] -= dt_m[event_idx]
        self.origin_times -= dt_m
        dx = lsqr(valid_deriv, valid_dt)[0]
        dx = dx.reshape((self.n_events, self.dim))
        # Constrain all location changes (lat, lon, depth) to clip_range
        for i in range(min(3, self.dim)):
            if np.any(np.abs(dx[:, i]) > clip_range):
                print(f"Warning: large update in dimension {i}, clipping")
                dx[:, i] = np.clip(dx[:, i], -clip_range, clip_range)
        # Check for negative depth and mark events as invalid
        for event_idx in range(self.n_events):
            if self.invalid_events[event_idx]:
                continue
            new_depth = self.locations[event_idx, 2] - dx[event_idx, 2]
            if (not solve_airquakes and new_depth < 0) or (iteration > max_depth_iterations and new_depth > max_depth):
                    print(f"Event {event_idx} would have depth {new_depth} outside [0, {max_depth}]. Marking as invalid.")
                    self.invalid_events[event_idx] = True
                    dx[event_idx, :] = 0  # No update for this event
        return dx
    
    def init_locations(self, trial_depth: float = 10.0, trial_time: float = 2.0):
        for event in range(self.n_events):
            # Find the station with the minimum arrival time for each event
            closest_pick = sorted(self.events[event].picks, key=lambda x: x.time)[0]
            closest_station = self.inv.select(
                network=closest_pick.waveform_id.network_code,
                station=closest_pick.waveform_id.station_code)[0][0]

            # Set the location of the event to the location of the closest station and the trial depth
            self.locations[event] = (closest_station.latitude, closest_station.longitude, trial_depth)
            # Offset the origin time by the difference between the closest pick and the trial time
            self.origin_times[event] = closest_pick.time - trial_time

        return self.locations

    def locate(self, max_iterations: int = 10, conv: float = 1e-2, solve_airquakes: bool = False, max_depth: float = 20.0, max_depth_iterations: int = 5):
        self.init_locations()
        max_dx = np.inf
        prev_max_dx = np.inf
        for i in range(max_iterations):
            dx = self._iterate_locate_3d(solve_airquakes=solve_airquakes, max_depth=max_depth, max_depth_iterations=max_depth_iterations, iteration=i)
            max_dx = np.max(np.abs(dx))
            # Location of maximum update
            max_dx_idx = np.unravel_index(np.argmax(np.abs(dx)), dx.shape)
            print(f"Event {max_dx_idx[0]}, Station {max_dx_idx[1]}, max_dx = {max_dx}")
            dx[:, 0:2] = kilometer2degrees(dx[:, 0:2])
            
            # Track the maximum update for convergence criteria
            print(f"iteration {i}: max_dx = {max_dx}")
            if max_dx < conv:
                print("Solution converged before max iterations")
                break
            elif np.abs(prev_max_dx - max_dx) / prev_max_dx < 0.05 and max_dx < 0.5:
                print("Solution is converged to unstable point")
                break
            prev_max_dx = max_dx
            # Update the locations for the next iteration
            self.locations -= dx
        return self.locations, self.origin_times, max_dx

    def report_invalid_events(self):
        """Return indices of events marked as invalid (negative depth)."""
        if hasattr(self, 'invalid_events'):
            return np.where(self.invalid_events)[0]
        return np.array([])