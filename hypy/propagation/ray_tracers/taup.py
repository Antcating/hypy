from typing import List
from hypy.propagation import PropagationModel

from obspy.taup import TauPyModel
from obspy.taup.tau import Arrivals, Arrival
from obspy.geodetics import kilometer2degrees

class TauPTracer(PropagationModel):
    """
    Ray tracer using Pyrocko's cake module for seismic ray tracing.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = TauPyModel(self.model_name)

    def get_arrival(self, source_depth: float, distance: float, phase: str):
        """
        Get the arrival time for a given source depth, distance, and seismic phase.
        """
        if phase.lower() == 'p':
            phases = ["p", "P"]
        elif phase.lower() == 's':
            phases = ["s", "S"]
        else:
            raise ValueError(f"Unsupported phase: {phase}. Supported phases are 'P' and 'S'.")
        
         # Convert distance from km to degrees
        arrivals: List[Arrival] = self.model.get_ray_paths(
            source_depth_in_km=source_depth,
            distance_in_degree=kilometer2degrees(distance),
            phase_list=phases,
        )
        if len(arrivals) == 0:
            raise ValueError(f"No arrivals found for phase {phase} at distance {distance} and depth {source_depth}.")
        

        return arrivals[0].time, arrivals[0].takeoff_angle