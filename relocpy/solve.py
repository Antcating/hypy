import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

class EventLocator:
    def __init__(self, events: list, stations: list, model: TauPyModel, dim: int=2):
        self.events = events
        self.stations = stations
        self.model = model
        self.theoretical_arrivals = np.zeros((len(events), len(stations)))
        self.derivative_matrix = lil_matrix((len(events) * len(stations), len(events) * dim))
        self.locations = np.zeros((len(events), dim))
    
    def _dt_dr(self, distance_km, source_depth, receiver_depth, epsilon=0.01):
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
    
    def _dt_dz(self, distance_km, source_depth, receiver_depth, epsilon=0.01):
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
    
    def _iterate_locate(self, event_initials, arrivals_times, epsilon=0.01):
        for event_idx, event_initial in enumerate(event_initials):
            for station_idx, station in enumerate(self.stations):
                distance_km = np.sqrt((event_initial[1] - station[1]) ** 2)
                distance_deg = kilometer2degrees(distance_km)
                arrivals = self.model.get_ray_paths(source_depth_in_km=event_initial[0], 
                                                    receiver_depth_in_km=station[0],
                                                    distance_in_degree=distance_deg, 
                                                    phase_list=["P", "p"])
                arrival = arrivals[0]
                dt_dx = self._dt_dr(distance_km, event_initial[0], station[0], epsilon)
                dt_dz = self._dt_dz(distance_km, event_initial[0], station[0], epsilon)
                self.theoretical_arrivals[event_idx, station_idx] = arrival.time
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 2] = dt_dz
                self.derivative_matrix[event_idx * len(self.stations) + station_idx, event_idx * 2 + 1] = dt_dx
        
        dx = np.linalg.lstsq(self.derivative_matrix.toarray(), (self.theoretical_arrivals - arrivals_times).flatten(), rcond=None)[0]
        return dx.reshape((len(event_initials), 2))
    
    def locate(self, initial_locations, arrivals_times, epsilon=0.01, max_iterations=10):
        locations = initial_locations.copy()
        max_dx = np.inf
        for i in range(max_iterations):
            dx = self._iterate_locate(locations, arrivals_times, epsilon=min(epsilon, 0.01*max_dx))
            max_dx = np.max(np.abs(dx))
            print(f"iteration {i}: max_dx = {max_dx}")
            locations -= dx
        return locations