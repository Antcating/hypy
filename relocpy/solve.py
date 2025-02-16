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
        self.allocate_arrays(self.n_events, self.n_stations, self.n_phases)
        self.fill_data()

    def allocate_arrays(self, n_events: int, n_stations: int, n_phases: int):
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

    def fill_data(self):
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

        
    def _dt_dr_num(self, distance_km, source_depth, receiver_depth, epsilon=0.01):
        """
        Calculate the numerical derivative of travel time with respect to distance.

        This method computes the numerical derivative of the travel time with respect to 
        the distance between the source and receiver using a central difference method.

        WARNING: This method is not efficient (because it computes the travel times twice) and should be used for testing purposes only.

        Parameters:
        distance_km (float): The distance between the source and receiver in kilometers.
        source_depth (float): The depth of the source in kilometers.
        receiver_depth (float): The depth of the receiver in kilometers.
        epsilon (float, optional): A small value to perturb the distance for numerical 
                                   differentiation. Default is 0.01 km.

        Returns:
        float: The numerical derivative of the travel time with respect to distance.
        """
        left_distance_km = distance_km - epsilon
        right_distance_km = distance_km + epsilon
        left_distance_deg = kilometer2degrees(left_distance_km)
        right_distance_deg = kilometer2degrees(right_distance_km)
        tt_left = self.model.get_travel_times(source_depth_in_km=source_depth, 
                                              receiver_depth_in_km=receiver_depth,
                                              distance_in_degree=left_distance_deg, 
                                              phase_list=["P", "p"])[0].time
        tt_right = self.model.get_travel_times(source_depth_in_km=source_depth, 
                                               receiver_depth_in_km=receiver_depth,
                                               distance_in_degree=right_distance_deg, 
                                               phase_list=["P", "p"])[0].time
        dt_dr = (tt_right - tt_left) / (2 * epsilon)
        return dt_dr
    
    def _dt_dz_num(self, distance_km, source_depth, receiver_depth, epsilon=0.01):
        """
        Calculate the numerical derivative of travel time with respect to source depth.

        This function computes the numerical derivative of the travel time with respect to the source depth
        using a central difference method. It perturbs the source depth by a small value epsilon and calculates
        the travel times for the perturbed depths to estimate the derivative.

        WARNING: This method is not efficient (because it computes the travel times twice) and should be used for testing purposes only.

        Parameters:
        -----------
        distance_km : float
            The distance between the source and receiver in kilometers.
        source_depth : float
            The depth of the source in kilometers.
        receiver_depth : float
            The depth of the receiver in kilometers.
        epsilon : float, optional
            The small perturbation value for the source depth (default is 0.01 km).

        Returns:
        --------
        float
            The numerical derivative of the travel time with respect to the source depth.
        """
        distance_deg = kilometer2degrees(distance_km)
        left_depth = source_depth - epsilon
        right_depth = source_depth + epsilon
        tt_left = self.model.get_travel_times(source_depth_in_km=left_depth, 
                                              receiver_depth_in_km=receiver_depth,
                                              distance_in_degree=distance_deg, 
                                              phase_list=["P", "p"])[0].time
        tt_right = self.model.get_travel_times(source_depth_in_km=right_depth, 
                                               receiver_depth_in_km=receiver_depth,
                                               distance_in_degree=distance_deg, 
                                               phase_list=["P", "p"])[0].time
        dt_dz = (tt_right - tt_left) / (2 * epsilon)
        return dt_dz
    
    def _num_dt(self, distance_km, source_depth, receiver_depth, epsilon=0.01):
        """
        Calculate the numerical derivative of travel time with respect to distance and source depth.

        This function computes the numerical derivative of the travel time with respect to the distance between the source and receiver
        and the source depth using a central difference method. It perturbs the distance and source depth by a small value epsilon
        and calculates the travel times for the perturbed distances and depths to estimate the derivatives.

        WARNING: This method is not efficient (because it computes the travel times twice) and should be used for testing purposes only.

        Parameters:
        -----------
        distance_km : float
            The distance between the source and receiver in kilometers.
        source_depth : float
            The depth of the source in kilometers.
        receiver_depth : float
            The depth of the receiver in kilometers.
        epsilon : float, optional
            The small perturbation value for the distance and source depth (default is 0.01 km).

        Returns:
        --------
        tuple
            A tuple containing the numerical derivatives of the travel time with respect to distance and source depth.
        """
        dt_dr = self._dt_dr_num(distance_km, source_depth, receiver_depth, epsilon=epsilon)
        dt_dz = self._dt_dz_num(distance_km, source_depth, receiver_depth, epsilon=epsilon)
        return dt_dr, dt_dz

    def _analyt_dt(self, arrival, phase: str):
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
        vel_p = self.model.model.s_mod.v_mod.evaluate_below(depth , phase.lower())[0]
        dt_dr = np.sin(np.deg2rad(arrival.takeoff_angle)) / vel_p
        dt_dz = -np.cos(np.deg2rad(arrival.takeoff_angle)) / vel_p
        return dt_dr, dt_dz
    
    def _iterate_locate_3d(self, method: str ="analytical", epsilon: float = 0.01, clip_range: float = 5):
        for event_idx in range(self.n_events):
            for station_idx, (network, station) in enumerate(self.stations):
                event_location = self.locations[event_idx]
                station_location = self.stations_locations[station_idx]
                distance_m, az, _ = gps2dist_azimuth(event_location[0], event_location[1], station_location[0], station_location[1])
                distance_km = distance_m / 1000
                for phase_idx, phase in enumerate(self.phases):
                    if phase == "P":
                        phase_list = ["P", "p"]
                    elif phase == "S":
                        phase_list = ["S", "s"]
                    else:
                        raise ValueError("Invalid phase")
                    arrivals = self.model.get_ray_paths_geo(
                        source_latitude_in_deg=event_location[0], source_longitude_in_deg=event_location[1], source_depth_in_km=event_location[2],
                        receiver_latitude_in_deg=station_location[0], receiver_longitude_in_deg=station_location[1],
                        phase_list=phase_list)
                    arrival = arrivals[0]
                    if method == "numerical":
                        dt_dr, dt_dz = self._num_dt(distance_km, event_location[2], station_location[2], epsilon=epsilon)
                    elif method == "analytical":
                        dt_dr, dt_dz = self._analyt_dt(arrival, phase)
                    else:
                        raise ValueError("Invalid method")
                    
                    # Azimuth is measured from the north
                    # dx is latitude, dy is longitude
                    dt_dx = -dt_dr * np.cos(np.deg2rad(az))
                    dt_dy = -dt_dr * np.sin(np.deg2rad(az))

                    self.theoretical_arrivals[event_idx, self.n_phases * station_idx + phase_idx] = arrival.time
                    # The derivative matrix is a sparse matrix
                    # The row index is the event index times the number of stations plus the station index
                    # The column index is the event index times 3 (for 3D) plus the coordinate index:
                    # 0 for latitude, 1 for longitude, 2 for depth
                    self.derivative_matrix[event_idx * len(self.stations) + (self.n_phases * station_idx + phase_idx), event_idx * 3] = dt_dx
                    self.derivative_matrix[event_idx * len(self.stations) + (self.n_phases * station_idx + phase_idx), event_idx * 3 + 1] = dt_dy
                    self.derivative_matrix[event_idx * len(self.stations) + (self.n_phases * station_idx + phase_idx), event_idx * 3 + 2] = dt_dz

        travel_times = self.arrivals - self.origin_times[:, np.newaxis]

        dt = self.theoretical_arrivals - travel_times

        dt_m = np.mean(dt, axis=1)
        dt = dt - dt_m[:, np.newaxis]
        self.origin_times -= dt_m

        dx = lsqr(self.derivative_matrix, dt.flatten())[0]
        dx = dx.reshape((self.n_events, self.dim))

        if np.any(dx[:, 2] > clip_range):
            print("Warning: large depth update")
            print("Clipping depth update")
            dx[:, 2] = np.clip(dx[:, 2], -clip_range, clip_range)
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

    def locate(self, epsilon: float = 0.01, max_iterations: int = 10, conv: float = 1e-2):
        self.init_locations()
        max_dx = np.inf
        for i in range(max_iterations):
            if self.dim == 2:
                dx = self._iterate_locate_2d(epsilon=epsilon)
                max_dx = np.max(np.abs(dx))
            elif self.dim == 3:
                dx = self._iterate_locate_3d(epsilon=epsilon)
                max_dx = np.max(np.abs(dx))
                dx[:, 0:2] = kilometer2degrees(dx[:, 0:2])
            else:
                raise ValueError("Invalid dimension")
            
            # Track the maximum update for convergence criteria
            print(f"iteration {i}: max_dx = {max_dx}")
            if max_dx < conv:
                print("Solution converged before max iterations")
                break
            # Update the locations for the next iteration
            self.locations -= dx
        return self.locations, self.origin_times