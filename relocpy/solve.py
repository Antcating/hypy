import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees, gps2dist_azimuth
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

class EventLocator:
    def __init__(self, events: list, stations: list, model: TauPyModel, dim: int=2):
        self.events = events
        self.stations = stations
        self.model = model
        self.dim = dim
        self.theoretical_arrivals = np.zeros((len(events), len(stations)))
        self.derivative_matrix = lil_matrix((len(events) * len(stations), len(events) * self.dim))
        self.locations = np.zeros((len(events), self.dim))
    
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

    def _analyt_dt(self, arrival):
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
        vel_p = self.model.model.s_mod.v_mod.evaluate_below(depth , "p")[0]
        dt_dr = np.sin(np.deg2rad(arrival.takeoff_angle)) / vel_p
        dt_dz = -np.cos(np.deg2rad(arrival.takeoff_angle)) / vel_p
        return dt_dr, dt_dz

    def _iterate_locate_2d(self, event_initials, arrivals_times, method="analytical", epsilon=0.01):
        for event_idx, event_initial in enumerate(event_initials):
            for station_idx, station in enumerate(self.stations):
                distance_km = np.sqrt((event_initial[1] - station[1]) ** 2)
                distance_deg = kilometer2degrees(distance_km)
                arrivals = self.model.get_ray_paths(source_depth_in_km=event_initial[0], 
                                                    receiver_depth_in_km=station[0],
                                                    distance_in_degree=distance_deg, 
                                                    phase_list=["P", "p"])
                arrival = arrivals[0]
                if method == "numerical":
                    dt_dr, dt_dz = self._num_dt(distance_km, event_initial[0], station[0], epsilon=epsilon)
                elif method == "analytical":
                    dt_dr, dt_dz = self._analyt_dt(arrival)
                else:
                    raise ValueError("Invalid method")

                self.theoretical_arrivals[event_idx, station_idx] = arrival.time
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 2] = dt_dz
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 2 + 1] = dt_dr
        
        dx = lsqr(self.derivative_matrix, (self.theoretical_arrivals - arrivals_times).flatten())[0]
        return dx.reshape((len(event_initials), self.dim))
    
    def _iterate_locate_3d(self, event_initials, arrivals_times, method="analytical", epsilon=0.01):
        for event_idx, event_initial in enumerate(event_initials):
            for station_idx, station in enumerate(self.stations):
                distance_m, az, _ = gps2dist_azimuth(event_initial[0], event_initial[1], station[0], station[1])
                
                distance_km = distance_m / 1000
                arrivals = self.model.get_ray_paths_geo(
                    source_latitude_in_deg=event_initial[0], source_longitude_in_deg=event_initial[1], source_depth_in_km=event_initial[2],
                    receiver_latitude_in_deg=station[0], receiver_longitude_in_deg=station[1],
                    phase_list=["P", "p"]
                )
                arrival = arrivals[0]
                if method == "numerical":
                    dt_dr, dt_dz = self._num_dt(distance_km, event_initial[2], station[2], epsilon=epsilon)
                elif method == "analytical":
                    dt_dr, dt_dz = self._analyt_dt(arrival)
                else:
                    raise ValueError("Invalid method")
                
                # Azimuth is measured from the north
                # dx is latitude, dy is longitude
                dt_dx = -dt_dr * np.cos(np.deg2rad(az))
                dt_dy = -dt_dr * np.sin(np.deg2rad(az))

                self.theoretical_arrivals[event_idx, station_idx] = arrival.time
                # The derivative matrix is a sparse matrix
                # The row index is the event index times the number of stations plus the station index
                # The column index is the event index times 3 (for 3D) plus the coordinate index:
                # 0 for latitude, 1 for longitude, 2 for depth
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 3] = dt_dx
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 3 + 1] = dt_dy
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 3 + 2] = dt_dz

        dx = lsqr(self.derivative_matrix, (self.theoretical_arrivals - arrivals_times).flatten())[0]
        dx = dx.reshape((len(event_initials), self.dim))
        if np.any(dx[:, 2] > 2):
            print("Warning: large depth update")
            print("Clamping depth update")
            dx[:, 2] = np.clip(dx[:, 2], -2, 2)
        return dx
    def locate(self, initial_locations, arrivals_times, epsilon: float = 0.01, max_iterations: int = 10):
        locations = initial_locations.copy()
        max_dx = np.inf
        for i in range(max_iterations):
            if self.dim == 2:
                dx = self._iterate_locate_2d(locations, arrivals_times, epsilon=epsilon)
            elif self.dim == 3:
                dx = self._iterate_locate_3d(locations, arrivals_times, epsilon=epsilon)
                dx[:, 0:2] = kilometer2degrees(dx[:, 0:2])
            max_dx = np.max(np.abs(dx))
            print(f"iteration {i}: max_dx = {max_dx}")
            locations -= dx
        return locations